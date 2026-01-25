#!/usr/bin/env python3
"""
REAP V12 - Corrected VLM Expert Pruning
========================================

CRITICAL FIXES from V11:
1. Correct REAP formula: normalize by ACTIVE token count, not total tokens
2. Only compute outputs for TOP-K selected experts (not all 128)
3. Use actual gate values from TOP-K selection
4. Proper VLM calibration with mixed data
5. Observer-style activation collection

REAP Score (from CerebrasResearch paper, Equation 9):
    Sⱼ = (1/|𝒳ⱼ|) × Σ(x∈𝒳ⱼ) gⱼ(x) · ‖fⱼ(x)‖₂

Where:
    - 𝒳ⱼ = set of tokens where expert j is in TOP-K
    - gⱼ(x) = gate value (after softmax + top-k selection)
    - ‖fⱼ(x)‖₂ = L2 norm of expert output

Based on: https://github.com/CerebrasResearch/reap
Paper: https://arxiv.org/abs/2510.13999
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from collections import defaultdict
from tqdm import tqdm
import shutil
from datetime import datetime
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ReapConfig:
    """REAP V12 Configuration - Corrected implementation."""
    
    # Paths
    model_path: str = os.environ.get(
        "REAP_MODEL_PATH",
        "/app/data/models/Qwen3-VL-235B-A22B-Thinking-AWQ"
    )
    output_path: str = os.environ.get(
        "REAP_OUTPUT_PATH", 
        "/app/output/reap-v12"
    )
    calibration_data_path: str = os.environ.get(
        "REAP_CALIBRATION_PATH",
        "/app/data/calibration/calibration_data.json"
    )
    
    # Pruning settings
    prune_ratio: float = float(os.environ.get("REAP_PRUNE_RATIO", "0.40"))
    max_samples: int = int(os.environ.get("REAP_MAX_SAMPLES", "300"))
    max_seq_len: int = int(os.environ.get("REAP_MAX_SEQ_LEN", "2048"))
    random_seed: int = int(os.environ.get("REAP_SEED", "42"))
    group_size: int = 128  # AWQ group size
    
    # Memory settings
    total_vram_gb: float = float(os.environ.get("REAP_VRAM_GB", "140.0"))
    vram_buffer_ratio: float = float(os.environ.get("REAP_VRAM_BUFFER", "0.10"))
    batch_size: int = int(os.environ.get("REAP_BATCH_SIZE", "1"))
    
    device: str = os.environ.get("REAP_DEVICE", "cuda:0")
    
    # Analysis
    analyze_distribution: bool = True
    save_per_layer_scores: bool = True


CONFIG = ReapConfig()
device = torch.device(CONFIG.device)


def print_banner():
    print("\n" + "="*70)
    print("  REAP V12 - Corrected VLM Expert Pruning")
    print("  Based on CerebrasResearch/reap")
    print("="*70)
    print(f"\n  Model: {CONFIG.model_path}")
    print(f"  Output: {CONFIG.output_path}")
    print(f"  Prune Ratio: {CONFIG.prune_ratio*100:.0f}%")
    print(f"  Max Samples: {CONFIG.max_samples}")
    print(f"\n  CRITICAL FIXES:")
    print(f"    ✓ Normalize by ACTIVE token count (not total)")
    print(f"    ✓ Only compute TOP-K expert outputs")
    print(f"    ✓ Use actual gate values from selection")
    print("="*70 + "\n")


# =============================================================================
# REAP SCORE ACCUMULATOR
# =============================================================================

class ReapScoreAccumulator:
    """
    Accumulates REAP scores correctly per the paper formula.
    
    REAP Score = (1/|𝒳ⱼ|) × Σ(x∈𝒳ⱼ) gⱼ(x) · ‖fⱼ(x)‖₂
    
    Where |𝒳ⱼ| is the number of tokens where expert j was SELECTED (in top-k),
    NOT the total number of tokens.
    """
    
    def __init__(self, num_experts: int, num_layers: int, device: torch.device):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.device = device
        
        # Per-layer accumulators
        # weighted_sum[layer][expert] = Σ gⱼ(x) · ‖fⱼ(x)‖₂
        self.weighted_sum = {
            layer: torch.zeros(num_experts, dtype=torch.float64, device=device)
            for layer in range(num_layers)
        }
        
        # active_count[layer][expert] = |𝒳ⱼ| (number of times expert was selected)
        self.active_count = {
            layer: torch.zeros(num_experts, dtype=torch.long, device=device)
            for layer in range(num_layers)
        }
        
    def update(self, layer_idx: int, expert_idx: int, gate_value: float, output_norm: float):
        """
        Update score for a single expert activation.
        
        Args:
            layer_idx: Layer index
            expert_idx: Expert index  
            gate_value: gⱼ(x) - the gate value for this expert on this token
            output_norm: ‖fⱼ(x)‖₂ - L2 norm of expert output
        """
        self.weighted_sum[layer_idx][expert_idx] += gate_value * output_norm
        self.active_count[layer_idx][expert_idx] += 1
        
    def update_batch(self, layer_idx: int, expert_indices: torch.Tensor, 
                     gate_values: torch.Tensor, output_norms: torch.Tensor):
        """
        Batch update for efficiency.
        
        Args:
            layer_idx: Layer index
            expert_indices: [N] expert indices that were selected
            gate_values: [N] corresponding gate values
            output_norms: [N] corresponding output norms
        """
        for i in range(len(expert_indices)):
            exp_idx = expert_indices[i].item()
            self.weighted_sum[layer_idx][exp_idx] += gate_values[i].item() * output_norms[i].item()
            self.active_count[layer_idx][exp_idx] += 1
    
    def get_scores(self) -> Dict[int, torch.Tensor]:
        """
        Compute final REAP scores: mean over ACTIVE tokens only.
        
        Returns:
            Dict mapping layer_idx -> [num_experts] tensor of REAP scores
        """
        scores = {}
        for layer_idx in range(self.num_layers):
            counts = self.active_count[layer_idx].float().clamp(min=1)
            scores[layer_idx] = (self.weighted_sum[layer_idx] / counts).cpu()
        return scores
    
    def get_counts(self) -> Dict[int, torch.Tensor]:
        """Get active counts per layer."""
        return {k: v.cpu() for k, v in self.active_count.items()}
    
    def get_statistics(self) -> Dict:
        """Get summary statistics for analysis."""
        stats = {
            'total_activations': 0,
            'zero_count_experts': 0,
            'per_layer': {}
        }
        
        for layer_idx in range(self.num_layers):
            counts = self.active_count[layer_idx]
            active = (counts > 0).sum().item()
            total = counts.sum().item()
            
            stats['per_layer'][layer_idx] = {
                'active_experts': active,
                'total_activations': total,
                'min_count': counts.min().item(),
                'max_count': counts.max().item(),
            }
            stats['total_activations'] += total
            stats['zero_count_experts'] += (self.num_experts - active)
            
        return stats


# =============================================================================
# AWQ DEQUANTIZATION
# =============================================================================

def dequantize_awq(qweight: torch.Tensor, qzeros: torch.Tensor, 
                   scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Dequantize AWQ 4-bit weights to FP16."""
    in_features = qweight.shape[0]
    out_features = scales.shape[1]
    num_groups = scales.shape[0]
    
    # Unpack 4-bit values
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    weight_int = ((qweight.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF)
    weight_int = weight_int.reshape(in_features, out_features)
    
    zeros_int = ((qzeros.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF)
    zeros_int = zeros_int.reshape(num_groups, out_features)
    
    # Expand to full size
    zeros_expanded = zeros_int.repeat_interleave(group_size, dim=0)[:in_features]
    scales_expanded = scales.repeat_interleave(group_size, dim=0)[:in_features]
    
    # Dequantize
    weight = (weight_int.float() - zeros_expanded.float()) * scales_expanded.float()
    return weight.T.half()


def get_weight(weights: dict, prefix: str) -> Optional[torch.Tensor]:
    """Get weight tensor, handling AWQ quantization if present."""
    qw = weights.get(f"{prefix}.qweight")
    qz = weights.get(f"{prefix}.qzeros")
    sc = weights.get(f"{prefix}.scales")
    
    if qw is not None and qz is not None and sc is not None:
        return dequantize_awq(qw, qz, sc, CONFIG.group_size)
    
    return weights.get(f"{prefix}.weight") or weights.get(prefix)


# =============================================================================
# ROTARY EMBEDDINGS
# =============================================================================

class RotaryEmbedding:
    """RoPE implementation for Qwen3-VL."""
    
    def __init__(self, dim: int, max_position: int = 2048, base: float = 1000000.0):
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.inv_freq = inv_freq.to(device)
        self._build_cache(max_position)
        
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        freqs = freqs.repeat_interleave(2, dim=-1)
        self.cos_cache = freqs.cos().half()
        self.sin_cache = freqs.sin().half()
        
    def apply(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)
            
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_reshape = x.reshape(*x.shape[:-1], -1, 2)
        x0, x1 = x_reshape[..., 0], x_reshape[..., 1]
        cos_r = cos.reshape(*cos.shape[:-1], -1, 2)[..., 0]
        sin_r = sin.reshape(*sin.shape[:-1], -1, 2)[..., 0]
        out0 = x0 * cos_r - x1 * sin_r
        out1 = x0 * sin_r + x1 * cos_r
        return torch.stack([out0, out1], dim=-1).reshape(x.shape)


# =============================================================================
# RMS NORM
# =============================================================================

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(orig_dtype)


# =============================================================================
# ATTENTION FORWARD
# =============================================================================

def forward_attention(hidden_states: torch.Tensor, weights: dict, prefix: str,
                      rope: RotaryEmbedding, num_heads: int, num_kv_heads: int,
                      head_dim: int) -> torch.Tensor:
    """Forward pass through attention layer."""
    B, S, H = hidden_states.shape
    
    q_proj = get_weight(weights, f"{prefix}self_attn.q_proj")
    k_proj = get_weight(weights, f"{prefix}self_attn.k_proj")
    v_proj = get_weight(weights, f"{prefix}self_attn.v_proj")
    o_proj = get_weight(weights, f"{prefix}self_attn.o_proj")
    
    if q_proj is None:
        return hidden_states
    
    q_bias = weights.get(f"{prefix}self_attn.q_proj.bias")
    k_bias = weights.get(f"{prefix}self_attn.k_proj.bias")
    v_bias = weights.get(f"{prefix}self_attn.v_proj.bias")
    o_bias = weights.get(f"{prefix}self_attn.o_proj.bias")
    
    q = F.linear(hidden_states, q_proj, q_bias).view(B, S, num_heads, head_dim).transpose(1, 2)
    k = F.linear(hidden_states, k_proj, k_bias).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    v = F.linear(hidden_states, v_proj, v_bias).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    
    # QK norm if present
    q_norm = weights.get(f"{prefix}self_attn.q_norm.weight")
    k_norm = weights.get(f"{prefix}self_attn.k_norm.weight")
    if q_norm is not None:
        q = rms_norm(q, q_norm.view(1, 1, 1, -1))
    if k_norm is not None:
        k = rms_norm(k, k_norm.view(1, 1, 1, -1))
        
    # Apply RoPE
    q, k = rope.apply(q, k)
    
    # GQA expansion
    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    
    # Attention
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, num_heads * head_dim)
    
    return F.linear(attn_output, o_proj, o_bias)


# =============================================================================
# MOE FORWARD WITH CORRECT REAP SCORING
# =============================================================================

def forward_moe_with_reap(
    hidden_states: torch.Tensor,
    weights: dict,
    prefix: str,
    num_experts: int,
    top_k: int,
    layer_idx: int,
    score_accumulator: ReapScoreAccumulator
) -> torch.Tensor:
    """
    Forward pass through MoE layer with CORRECT REAP score collection.
    
    Key differences from V11:
    1. Only compute outputs for TOP-K selected experts
    2. Use actual gate values (after top-k selection and normalization)
    3. Accumulate scores only for ACTIVE tokens
    
    Args:
        hidden_states: [B, S, H] input tensor
        weights: Layer weights dict
        prefix: Weight name prefix
        num_experts: Total number of experts
        top_k: Number of experts per token
        layer_idx: Current layer index
        score_accumulator: REAP score accumulator
        
    Returns:
        MoE output tensor [B, S, H]
    """
    B, S, H = hidden_states.shape
    total_tokens = B * S
    hidden_flat = hidden_states.view(total_tokens, H)
    
    # Get router/gate weight
    gate_weight = get_weight(weights, f"{prefix}mlp.gate")
    if gate_weight is None:
        return hidden_states
    
    # Compute router logits and probabilities
    router_logits = F.linear(hidden_flat, gate_weight)
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    
    # TOP-K selection
    topk_weights, topk_indices = torch.topk(router_probs, top_k, dim=-1)
    
    # Normalize top-k weights (this is what actually gets used)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.half()
    
    # Initialize output
    moe_output = torch.zeros_like(hidden_flat)
    
    # Cache for expert outputs (avoid recomputation)
    # Key: expert_idx, Value: dict of token_idx -> output tensor
    expert_cache: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
    
    # Find which experts are selected and for which tokens
    expert_to_tokens: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    # expert_to_tokens[expert_idx] = [(token_idx, k_position), ...]
    
    for token_idx in range(total_tokens):
        for k_pos in range(top_k):
            exp_idx = topk_indices[token_idx, k_pos].item()
            expert_to_tokens[exp_idx].append((token_idx, k_pos))
    
    # Process each selected expert
    for exp_idx, token_info_list in expert_to_tokens.items():
        if not token_info_list:
            continue
            
        # Get expert weights
        gate_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.gate_proj")
        up_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.up_proj")
        down_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.down_proj")
        
        if gate_proj is None:
            continue
        
        # Get all tokens that use this expert
        token_indices = [t[0] for t in token_info_list]
        tokens_tensor = hidden_flat[token_indices]  # [N, H]
        
        # Compute expert output for all tokens at once
        with torch.amp.autocast('cuda'):
            gate_out = F.silu(F.linear(tokens_tensor, gate_proj))
            up_out = F.linear(tokens_tensor, up_proj)
            expert_out = F.linear(gate_out * up_out, down_proj)  # [N, H]
        
        # Compute L2 norms for REAP scoring
        output_norms = torch.norm(expert_out.float(), p=2, dim=-1)  # [N]
        
        # Process each token
        for i, (token_idx, k_pos) in enumerate(token_info_list):
            # Get actual gate value for this expert on this token
            gate_value = topk_weights[token_idx, k_pos].float().item()
            output_norm = output_norms[i].item()
            
            # Update REAP score accumulator
            score_accumulator.update(layer_idx, exp_idx, gate_value, output_norm)
            
            # Add weighted output to MoE result
            moe_output[token_idx] += topk_weights[token_idx, k_pos] * expert_out[i]
    
    return moe_output.view(B, S, H)


# =============================================================================
# LAYER PROCESSOR
# =============================================================================

class LayerProcessor:
    """Processes model layers with weight loading/unloading."""
    
    def __init__(self, model_path: str, config: dict, num_layers: int):
        self.model_path = model_path
        self.config = config
        self.num_layers = num_layers
        
        # Load weight index
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_index = json.load(f)
        
        self.weight_map = weight_index['weight_map']
        self.shard_to_weights = defaultdict(list)
        for name, shard in self.weight_map.items():
            self.shard_to_weights[shard].append(name)
    
    def _load_layer_weights(self, layer_idx: int) -> dict:
        """Load weights for a specific layer."""
        prefix = f"model.language_model.layers.{layer_idx}."
        weights = {}
        
        # Find shards containing this layer's weights
        needed_shards = set()
        for name, shard in self.weight_map.items():
            if prefix in name:
                needed_shards.add(shard)
        
        # Load from each shard
        for shard in needed_shards:
            shard_path = os.path.join(self.model_path, shard)
            with safe_open(shard_path, framework="pt", device="cuda:0") as f:
                for name in self.shard_to_weights[shard]:
                    if prefix in name:
                        weights[name] = f.get_tensor(name)
        
        return weights
    
    def process_layer(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        weights: dict,
        rope: RotaryEmbedding,
        score_accumulator: ReapScoreAccumulator
    ) -> torch.Tensor:
        """Process a single layer with REAP scoring."""
        prefix = f"model.language_model.layers.{layer_idx}."
        
        # Pre-attention norm
        residual = hidden_states
        input_ln = weights.get(f"{prefix}input_layernorm.weight")
        if input_ln is not None:
            hidden_states = rms_norm(hidden_states, input_ln)
        
        # Attention
        hidden_states = forward_attention(
            hidden_states, weights, prefix, rope,
            self.config['num_attention_heads'],
            self.config['num_key_value_heads'],
            self.config['head_dim']
        )
        hidden_states = residual + hidden_states
        
        # Post-attention norm
        residual = hidden_states
        post_ln = weights.get(f"{prefix}post_attention_layernorm.weight")
        if post_ln is not None:
            hidden_normed = rms_norm(hidden_states, post_ln)
        else:
            hidden_normed = hidden_states
        
        # MoE with REAP scoring
        moe_out = forward_moe_with_reap(
            hidden_normed,
            weights,
            prefix,
            self.config['num_experts'],
            self.config['num_experts_per_tok'],
            layer_idx,
            score_accumulator
        )
        
        hidden_states = residual + moe_out
        
        return hidden_states
    
    def process_all_layers(
        self,
        hidden_states_list: List[torch.Tensor],
        rope: RotaryEmbedding,
        score_accumulator: ReapScoreAccumulator,
        progress_callback=None
    ) -> List[torch.Tensor]:
        """Process all layers for all samples."""
        
        for layer_idx in range(self.num_layers):
            layer_start = time.time()
            
            # Load layer weights
            weights = self._load_layer_weights(layer_idx)
            
            # Process each sample
            for sample_idx, hidden in enumerate(hidden_states_list):
                hidden_batched = hidden.unsqueeze(0)
                
                with torch.no_grad():
                    output = self.process_layer(
                        hidden_batched,
                        layer_idx,
                        weights,
                        rope,
                        score_accumulator
                    )
                
                hidden_states_list[sample_idx] = output.squeeze(0)
            
            # Cleanup
            del weights
            torch.cuda.empty_cache()
            
            # Progress callback
            if progress_callback:
                layer_time = time.time() - layer_start
                stats = score_accumulator.get_statistics()
                layer_stats = stats['per_layer'].get(layer_idx, {})
                progress_callback(layer_idx, layer_time, layer_stats)
        
        return hidden_states_list


# =============================================================================
# VISION ENCODER
# =============================================================================

def load_vision_encoder(model_path: str):
    """Load vision encoder for VLM models."""
    try:
        from transformers import AutoConfig
        
        # Try to load Qwen3-VL vision encoder
        try:
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLVisionEncoder
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            visual = Qwen3VLVisionEncoder(config.vision_config).to(device).half()
        except ImportError:
            # Fallback to generic
            from transformers import AutoModel
            visual = AutoModel.from_pretrained(
                model_path, 
                trust_remote_code=True,
                subfolder="visual"
            ).to(device).half()
        
        # Load vision weights
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_index = json.load(f)
        
        vision_weights = {}
        for name, shard in weight_index['weight_map'].items():
            if 'visual' in name:
                shard_path = os.path.join(model_path, shard)
                with safe_open(shard_path, framework="pt", device="cuda:0") as f:
                    if name in f.keys():
                        clean_name = name.replace("model.visual.", "")
                        vision_weights[clean_name] = f.get_tensor(name)
        
        if vision_weights:
            visual.load_state_dict(vision_weights, strict=False)
        
        visual.eval()
        print(f"  Vision encoder loaded: {sum(p.numel() for p in visual.parameters())/1e6:.1f}M params")
        return visual
        
    except Exception as e:
        print(f"  Vision encoder not available: {e}")
        return None


# =============================================================================
# CALIBRATION DATASET
# =============================================================================

class CalibrationDataset:
    """
    Calibration dataset for VLM REAP scoring.
    
    Important: Mix of vision+text and text-only samples for balanced calibration.
    """
    
    VISION_TEMPLATES = [
        'Gorselde "{region}" nesnesini bul.',
        'Locate "{region}" in the image.',
        '"{region}" konumunu tespit et.',
        'Find the bounding box of "{region}".',
        'Where is "{region}" located?',
    ]
    
    TEXT_TEMPLATES = [
        "Explain the concept of {topic}.",
        "What are the key aspects of {topic}?",
        "Describe {topic} in detail.",
    ]
    
    def __init__(self, json_path: str):
        self.base_dir = os.path.dirname(json_path)
        self.vision_samples = []
        self.text_samples = []
        
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # Vision samples
            image_path = os.path.join(self.base_dir, item.get('image', ''))
            if os.path.exists(image_path):
                for region in item.get('regions', []):
                    if region:
                        self.vision_samples.append({
                            'type': 'vision',
                            'image_path': image_path,
                            'region': region,
                        })
            
            # Text samples
            if 'prompt' in item:
                self.text_samples.append({
                    'type': 'text',
                    'prompt': item['prompt'],
                })
        
        print(f"  Loaded {len(self.vision_samples)} vision samples")
        print(f"  Loaded {len(self.text_samples)} text samples")
    
    def generate_samples(self, max_samples: Optional[int] = None, 
                         vision_ratio: float = 0.7) -> List[dict]:
        """
        Generate balanced calibration samples.
        
        Args:
            max_samples: Maximum total samples
            vision_ratio: Ratio of vision samples (default 0.7 = 70% vision)
        """
        random.seed(CONFIG.random_seed)
        
        if max_samples is None:
            max_samples = len(self.vision_samples) + len(self.text_samples)
        
        # Calculate split
        n_vision = min(int(max_samples * vision_ratio), len(self.vision_samples))
        n_text = min(max_samples - n_vision, len(self.text_samples))
        
        # Sample
        vision_selected = random.sample(self.vision_samples, n_vision) if n_vision > 0 else []
        text_selected = random.sample(self.text_samples, n_text) if n_text > 0 else []
        
        # Generate prompts
        results = []
        
        for s in vision_selected:
            template = random.choice(self.VISION_TEMPLATES)
            results.append({
                'type': 'vision',
                'image_path': s['image_path'],
                'prompt': template.format(region=s['region']),
            })
        
        for s in text_selected:
            results.append({
                'type': 'text',
                'prompt': s['prompt'],
            })
        
        # Shuffle
        random.shuffle(results)
        
        print(f"  Generated {len(results)} samples ({n_vision} vision, {n_text} text)")
        return results


# =============================================================================
# EXPERT SELECTION
# =============================================================================

def select_experts_to_keep(
    reap_scores: Dict[int, torch.Tensor],
    expert_counts: Dict[int, torch.Tensor],
    num_experts: int,
    prune_ratio: float,
    min_active_threshold: int = 10
) -> Tuple[Dict[str, List[int]], int]:
    """
    Select experts to keep based on REAP scores.
    
    Args:
        reap_scores: Dict[layer_idx -> [num_experts] scores]
        expert_counts: Dict[layer_idx -> [num_experts] activation counts]
        num_experts: Total experts per layer
        prune_ratio: Fraction to prune (e.g., 0.4 = prune 40%)
        min_active_threshold: Minimum activations to be considered
        
    Returns:
        (experts_to_keep dict, num_experts_kept)
    """
    experts_to_keep = int(num_experts * (1 - prune_ratio))
    
    result = {}
    
    for layer_idx in sorted(reap_scores.keys()):
        scores = reap_scores[layer_idx].numpy()
        counts = expert_counts[layer_idx].numpy()
        
        # Penalize experts with very few activations (might be unreliable scores)
        # But don't completely zero them - they might still be important
        reliability_weight = np.minimum(counts / min_active_threshold, 1.0)
        adjusted_scores = scores * reliability_weight
        
        # Sort by adjusted score (descending)
        sorted_indices = np.argsort(adjusted_scores)[::-1]
        
        # Keep top experts
        kept = sorted_indices[:experts_to_keep].tolist()
        result[str(layer_idx)] = sorted(kept)  # Sort for consistency
    
    return result, experts_to_keep


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_results(
    reap_scores: Dict[int, torch.Tensor],
    expert_counts: Dict[int, torch.Tensor],
    experts_to_keep: Dict[str, List[int]],
    num_experts: int,
    num_layers: int
):
    """Analyze and print REAP scoring results."""
    
    print("\n" + "="*70)
    print("  REAP SCORING ANALYSIS")
    print("="*70)
    
    # Overall statistics
    total_active = sum(
        (expert_counts[i] > 0).sum().item() 
        for i in range(num_layers)
    )
    total_possible = num_experts * num_layers
    
    print(f"\n  Active expert-layer combinations: {total_active}/{total_possible}")
    print(f"  Zero-activation experts: {total_possible - total_active}")
    
    # Per-layer analysis (sample layers)
    print(f"\n  Sample Layer Analysis:")
    sample_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
    
    for layer_idx in sample_layers:
        scores = reap_scores[layer_idx].numpy()
        counts = expert_counts[layer_idx].numpy()
        kept = experts_to_keep[str(layer_idx)]
        
        active = (counts > 0).sum()
        kept_scores = scores[kept]
        pruned_indices = [i for i in range(num_experts) if i not in kept]
        pruned_scores = scores[pruned_indices] if pruned_indices else np.array([0])
        
        print(f"\n  Layer {layer_idx}:")
        print(f"    Active experts: {active}/{num_experts}")
        print(f"    Kept experts score range: [{kept_scores.min():.4f}, {kept_scores.max():.4f}]")
        print(f"    Pruned experts score range: [{pruned_scores.min():.4f}, {pruned_scores.max():.4f}]")
        print(f"    Score ratio (kept/pruned): {kept_scores.mean()/(pruned_scores.mean()+1e-8):.2f}x")
    
    # Score distribution
    all_scores = np.concatenate([reap_scores[i].numpy() for i in range(num_layers)])
    
    print(f"\n  Global Score Distribution:")
    print(f"    Min: {all_scores.min():.6f}")
    print(f"    Max: {all_scores.max():.6f}")
    print(f"    Mean: {all_scores.mean():.6f}")
    print(f"    Std: {all_scores.std():.6f}")
    print(f"    Median: {np.median(all_scores):.6f}")
    
    print("\n" + "="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_banner()
    
    # 1. Load model config
    print("\n[1/7] Loading model configuration...")
    
    config_path = os.path.join(CONFIG.model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    text_config = config.get('text_config', config)
    
    model_config = {
        'num_hidden_layers': text_config['num_hidden_layers'],
        'num_attention_heads': text_config.get('num_attention_heads', 64),
        'num_key_value_heads': text_config.get('num_key_value_heads', 4),
        'hidden_size': text_config['hidden_size'],
        'head_dim': text_config.get('head_dim', text_config['hidden_size'] // 64),
        'num_experts': text_config.get('num_experts', 128),
        'num_experts_per_tok': text_config.get('num_experts_per_tok', 8),
        'rope_theta': text_config.get('rope_theta', 1000000.0),
    }
    
    NUM_LAYERS = model_config['num_hidden_layers']
    NUM_EXPERTS = model_config['num_experts']
    TOP_K = model_config['num_experts_per_tok']
    EXPERTS_TO_KEEP = NUM_EXPERTS - int(NUM_EXPERTS * CONFIG.prune_ratio)
    
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Experts per layer: {NUM_EXPERTS}")
    print(f"  Top-K: {TOP_K}")
    print(f"  Experts to keep: {EXPERTS_TO_KEEP} ({100*(1-CONFIG.prune_ratio):.0f}%)")
    
    # 2. Load calibration data
    print("\n[2/7] Loading calibration data...")
    dataset = CalibrationDataset(CONFIG.calibration_data_path)
    samples = dataset.generate_samples(CONFIG.max_samples)
    
    # 3. Load processor
    print("\n[3/7] Loading tokenizer and embeddings...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(CONFIG.model_path, trust_remote_code=True)
    
    # Load embedding weights
    index_path = os.path.join(CONFIG.model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_index = json.load(f)
    
    embed_shard = weight_index['weight_map']["model.language_model.embed_tokens.weight"]
    with safe_open(os.path.join(CONFIG.model_path, embed_shard), framework="pt", device="cuda:0") as f:
        embed_weight = f.get_tensor("model.language_model.embed_tokens.weight")
    
    embed_tokens = nn.Embedding(embed_weight.shape[0], embed_weight.shape[1], device=device)
    embed_tokens.weight.data = embed_weight.half()
    
    print(f"  Vocabulary size: {embed_weight.shape[0]}")
    print(f"  Hidden size: {embed_weight.shape[1]}")
    
    # 4. Load vision encoder
    print("\n[4/7] Loading vision encoder...")
    visual = load_vision_encoder(CONFIG.model_path)
    use_vision = visual is not None
    
    IMAGE_TOKEN_ID = config.get('image_token_id', 151655)
    
    # 5. Process calibration samples
    print("\n[5/7] Processing calibration samples...")
    all_hidden = []
    
    for sample in tqdm(samples, desc="  Encoding"):
        try:
            if sample.get('type') == 'vision' and use_vision:
                # Vision sample
                image = Image.open(sample['image_path']).convert('RGB')
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': sample['prompt']}
                ]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text], images=[image], 
                    return_tensors='pt', padding=True
                )
                
                input_ids = inputs['input_ids'].to(device)
                pixel_values = inputs['pixel_values'].to(device).half()
                image_grid_thw = inputs['image_grid_thw'].to(device)
                
                with torch.no_grad():
                    vision_embeds = visual(pixel_values, grid_thw=image_grid_thw)
                    text_embeds = embed_tokens(input_ids)
                
                # Combine vision and text embeddings
                image_mask = (input_ids[0] == IMAGE_TOKEN_ID)
                image_pos = torch.where(image_mask)[0]
                
                combined = text_embeds.clone()
                if len(image_pos) > 0:
                    n_vision = min(len(image_pos), vision_embeds.shape[1])
                    for i in range(n_vision):
                        combined[0, image_pos[i]] = vision_embeds[0, i]
                
                hidden = combined.squeeze(0)
                
            else:
                # Text-only sample
                inputs = processor.tokenizer(
                    sample['prompt'], 
                    return_tensors="pt",
                    truncation=True, 
                    max_length=CONFIG.max_seq_len
                ).to(device)
                
                with torch.no_grad():
                    hidden = embed_tokens(inputs['input_ids']).squeeze(0)
            
            # Truncate if needed
            if hidden.shape[0] > CONFIG.max_seq_len:
                hidden = hidden[:CONFIG.max_seq_len]
            
            all_hidden.append(hidden)
            
        except Exception as e:
            print(f"  Warning: Failed to process sample: {e}")
            continue
    
    # Cleanup
    del embed_tokens
    if visual is not None:
        del visual
    torch.cuda.empty_cache()
    
    print(f"  Successfully processed: {len(all_hidden)} samples")
    
    if len(all_hidden) == 0:
        print("ERROR: No samples processed!")
        sys.exit(1)
    
    # 6. Run REAP scoring
    print("\n[6/7] Running REAP scoring...")
    
    # Initialize
    rope = RotaryEmbedding(
        model_config['head_dim'], 
        CONFIG.max_seq_len, 
        model_config['rope_theta']
    )
    
    score_accumulator = ReapScoreAccumulator(NUM_EXPERTS, NUM_LAYERS, device)
    layer_processor = LayerProcessor(CONFIG.model_path, model_config, NUM_LAYERS)
    
    # Progress callback
    def progress_cb(layer_idx, layer_time, layer_stats):
        active = layer_stats.get('active_experts', 0)
        total = layer_stats.get('total_activations', 0)
        print(f"  Layer {layer_idx+1:3d}/{NUM_LAYERS}: "
              f"{active:3d}/{NUM_EXPERTS} active, "
              f"{total:6d} activations, "
              f"{layer_time:.1f}s")
    
    # Process
    start_time = time.time()
    layer_processor.process_all_layers(all_hidden, rope, score_accumulator, progress_cb)
    total_time = time.time() - start_time
    
    print(f"\n  Total scoring time: {total_time/60:.1f} minutes")
    
    # Get results
    reap_scores = score_accumulator.get_scores()
    expert_counts = score_accumulator.get_counts()
    
    # 7. Select experts and save
    print("\n[7/7] Selecting experts and saving results...")
    
    experts_to_keep, experts_kept = select_experts_to_keep(
        reap_scores, expert_counts, NUM_EXPERTS, CONFIG.prune_ratio
    )
    
    # Analysis
    if CONFIG.analyze_distribution:
        analyze_results(
            reap_scores, expert_counts, experts_to_keep, 
            NUM_EXPERTS, NUM_LAYERS
        )
    
    # Create output directory
    os.makedirs(CONFIG.output_path, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        'reap_version': 'v12_corrected',
        'description': 'Corrected REAP implementation matching CerebrasResearch paper',
        'critical_fixes': [
            'Normalize by ACTIVE token count, not total tokens',
            'Only compute outputs for TOP-K selected experts',
            'Use actual gate values from selection',
        ],
        'formula': 'Sⱼ = (1/|𝒳ⱼ|) × Σ(x∈𝒳ⱼ) gⱼ(x) · ‖fⱼ(x)‖₂',
        'prune_ratio': CONFIG.prune_ratio,
        'experts_kept': experts_kept,
        'num_experts_original': NUM_EXPERTS,
        'num_layers': NUM_LAYERS,
        'top_k': TOP_K,
        'calibration_samples': len(all_hidden),
        'max_seq_len': CONFIG.max_seq_len,
        'experts_to_keep': experts_to_keep,
        'timestamp': datetime.now().isoformat(),
        'model_path': CONFIG.model_path,
    }
    
    # Add per-layer scores if requested
    if CONFIG.save_per_layer_scores:
        metadata['reap_scores'] = {
            str(k): v.tolist() for k, v in reap_scores.items()
        }
        metadata['expert_counts'] = {
            str(k): v.tolist() for k, v in expert_counts.items()
        }
    
    # Save metadata
    metadata_path = os.path.join(CONFIG.output_path, 'reap_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy config files
    for fname in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                  'preprocessor_config.json', 'generation_config.json']:
        src = os.path.join(CONFIG.model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, CONFIG.output_path)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  REAP V12 SCORING COMPLETE!")
    print(f"{'='*70}")
    print(f"  Output: {CONFIG.output_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Experts: {NUM_EXPERTS} -> {experts_kept} ({100*(1-CONFIG.prune_ratio):.0f}% kept)")
    print(f"\n  Next steps:")
    print(f"    1. python prune_model.py --metadata {metadata_path}")
    print(f"    2. python test_pruned_model.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()