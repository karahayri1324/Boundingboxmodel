#!/usr/bin/env python3
"""
REAP V12.4 - Qwen3-VL-235B-A22B-Thinking-AWQ
=============================================

FIXES in V12.4:
1. Gate weight slicing - auto-detect expert dimension
2. Config-aware normalization (norm_topk_prob)
3. Native transformers vision encoder

Requirements:
    pip install transformers>=4.57.0
    pip install qwen-vl-utils
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
from typing import Dict, List, Optional, Tuple
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# CHECK TRANSFORMERS VERSION
# =============================================================================

def check_transformers_version():
    """Check if transformers supports Qwen3VLMoe."""
    try:
        import transformers
        version = transformers.__version__
        
        major, minor = int(version.split('.')[0]), int(version.split('.')[1])
        
        if major < 4 or (major == 4 and minor < 57):
            print(f"ERROR: transformers {version} detected. Need >= 4.57.0")
            print("Run: pip install transformers>=4.57.0")
            sys.exit(1)
        
        print(f"✓ transformers {version} OK")
        
        from transformers import Qwen3VLMoeForConditionalGeneration
        print("✓ Qwen3VLMoeForConditionalGeneration available")
        
        return True
        
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Run: pip install transformers>=4.57.0")
        sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Configuration for Qwen3-VL-235B-A22B-Thinking-AWQ."""
    
    # Paths
    model_path: str = os.environ.get(
        "REAP_MODEL_PATH",
        "/app/data/models/Qwen3-VL-235B-A22B-Thinking-AWQ"
    )
    output_path: str = os.environ.get(
        "REAP_OUTPUT_PATH", 
        "/app/output/reap-v12.4-qwen3vl"
    )
    calibration_data_path: str = os.environ.get(
        "REAP_CALIBRATION_PATH",
        "/app/data/calibration/calibration_data.json"
    )
    
    # Pruning
    prune_ratio: float = float(os.environ.get("REAP_PRUNE_RATIO", "0.40"))
    max_samples: int = int(os.environ.get("REAP_MAX_SAMPLES", "300"))
    max_seq_len: int = int(os.environ.get("REAP_MAX_SEQ_LEN", "2048"))
    random_seed: int = int(os.environ.get("REAP_SEED", "42"))
    
    # Memory
    device: str = os.environ.get("REAP_DEVICE", "cuda:0")
    
    # Text config (from config.json) - will be loaded dynamically
    num_layers: int = 94
    num_experts: int = 128
    num_experts_per_tok: int = 8
    hidden_size: int = 4096
    head_dim: int = 128
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    rope_theta: float = 5000000.0
    awq_group_size: int = 128
    norm_topk_prob: bool = True  # NEW: from config
    
    # MRoPE
    mrope_interleaved: bool = True
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    
    # Token IDs
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    
    # Weight prefixes
    layer_prefix: str = "model.language_model.layers"
    embed_prefix: str = "model.language_model.embed_tokens"
    vision_prefix: str = "model.visual"


def load_config_from_model(config: Config) -> Config:
    """Load config values from model's config.json."""
    config_path = os.path.join(config.model_path, "config.json")
    
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    text_config = model_config.get('text_config', {})
    
    # Update config with actual values
    config.num_layers = text_config.get('num_hidden_layers', config.num_layers)
    config.num_experts = text_config.get('num_experts', config.num_experts)
    config.num_experts_per_tok = text_config.get('num_experts_per_tok', config.num_experts_per_tok)
    config.hidden_size = text_config.get('hidden_size', config.hidden_size)
    config.head_dim = text_config.get('head_dim', config.head_dim)
    config.num_attention_heads = text_config.get('num_attention_heads', config.num_attention_heads)
    config.num_key_value_heads = text_config.get('num_key_value_heads', config.num_key_value_heads)
    config.rope_theta = text_config.get('rope_theta', config.rope_theta)
    
    # CRITICAL: norm_topk_prob
    config.norm_topk_prob = text_config.get('norm_topk_prob', True)
    
    # MRoPE
    rope_scaling = text_config.get('rope_scaling', {})
    config.mrope_interleaved = rope_scaling.get('mrope_interleaved', True)
    config.mrope_section = rope_scaling.get('mrope_section', [24, 20, 20])
    
    # Token IDs
    config.image_token_id = model_config.get('image_token_id', 151655)
    config.video_token_id = model_config.get('video_token_id', 151656)
    
    # AWQ group size
    quant_config = model_config.get('quantization_config', {})
    config.awq_group_size = quant_config.get('group_size', 128)
    
    return config


CONFIG = Config()
device = None  # Will be set after config load


def print_banner(config: Config):
    print("\n" + "="*70)
    print("  REAP V12.4 - Qwen3-VL-235B-A22B-Thinking-AWQ")
    print("  WITH GATE SLICING FIX + NORM_TOPK_PROB SUPPORT")
    print("="*70)
    print(f"\n  Model: {config.model_path}")
    print(f"  Output: {config.output_path}")
    print(f"  Prune Ratio: {config.prune_ratio*100:.0f}%")
    print(f"\n  Config (from model):")
    print(f"    Layers: {config.num_layers}, Experts: {config.num_experts} (top-{config.num_experts_per_tok})")
    print(f"    Hidden: {config.hidden_size}, Head: {config.head_dim}")
    print(f"    rope_theta: {config.rope_theta:,.0f}")
    print(f"    MRoPE: interleaved={config.mrope_interleaved}, sections={config.mrope_section}")
    print(f"    norm_topk_prob: {config.norm_topk_prob}")  # NEW
    print("="*70 + "\n")


# =============================================================================
# REAP SCORE ACCUMULATOR
# =============================================================================

class ReapScoreAccumulator:
    """
    REAP score accumulator with correct formula.
    
    REAP Score = (1/|𝒳ⱼ|) × Σ(x∈𝒳ⱼ) gⱼ(x) · ‖fⱼ(x)‖₂
    """
    
    def __init__(self, num_experts: int, num_layers: int, device: torch.device):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.device = device
        
        self.weighted_sum = {
            layer: torch.zeros(num_experts, dtype=torch.float64, device=device)
            for layer in range(num_layers)
        }
        self.active_count = {
            layer: torch.zeros(num_experts, dtype=torch.long, device=device)
            for layer in range(num_layers)
        }
    
    def update(self, layer_idx: int, expert_idx: int, gate_value: float, output_norm: float):
        self.weighted_sum[layer_idx][expert_idx] += gate_value * output_norm
        self.active_count[layer_idx][expert_idx] += 1
    
    def update_batch(self, layer_idx: int, expert_indices: torch.Tensor, 
                     gate_values: torch.Tensor, output_norms: torch.Tensor):
        """Batch update for efficiency."""
        for i in range(len(expert_indices)):
            exp_idx = expert_indices[i].item()
            self.weighted_sum[layer_idx][exp_idx] += gate_values[i].item() * output_norms[i].item()
            self.active_count[layer_idx][exp_idx] += 1
    
    def get_scores(self) -> Dict[int, torch.Tensor]:
        scores = {}
        for layer_idx in range(self.num_layers):
            counts = self.active_count[layer_idx].float().clamp(min=1)
            scores[layer_idx] = (self.weighted_sum[layer_idx] / counts).cpu()
        return scores
    
    def get_counts(self) -> Dict[int, torch.Tensor]:
        return {k: v.cpu() for k, v in self.active_count.items()}


# =============================================================================
# AWQ DEQUANTIZATION
# =============================================================================

def dequantize_awq(qweight: torch.Tensor, qzeros: torch.Tensor, 
                   scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Dequantize AWQ 4-bit weights."""
    in_features = qweight.shape[0]
    out_features = scales.shape[1]
    num_groups = scales.shape[0]
    
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    weight_int = ((qweight.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF)
    weight_int = weight_int.reshape(in_features, out_features)
    
    zeros_int = ((qzeros.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF)
    zeros_int = zeros_int.reshape(num_groups, out_features)
    
    zeros_expanded = zeros_int.repeat_interleave(group_size, dim=0)[:in_features]
    scales_expanded = scales.repeat_interleave(group_size, dim=0)[:in_features]
    
    weight = (weight_int.float() - zeros_expanded.float()) * scales_expanded.float()
    return weight.T.half()


def get_weight(weights: dict, prefix: str, is_awq: bool = True, 
               group_size: int = 128) -> Optional[torch.Tensor]:
    """Get weight tensor with AWQ support."""
    fp16_weight = weights.get(f"{prefix}.weight")
    if fp16_weight is not None:
        return fp16_weight
    
    if is_awq:
        qw = weights.get(f"{prefix}.qweight")
        qz = weights.get(f"{prefix}.qzeros")
        sc = weights.get(f"{prefix}.scales")
        
        if qw is not None and qz is not None and sc is not None:
            return dequantize_awq(qw, qz, sc, group_size)
    
    return None


# =============================================================================
# MROPE
# =============================================================================

class MRoPE:
    """Multi-Resolution RoPE for Qwen3-VL."""
    
    def __init__(self, head_dim: int, max_position: int, 
                 rope_theta: float, mrope_section: List[int],
                 interleaved: bool = True, device: torch.device = None):
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section
        self.interleaved = interleaved
        self.device = device or torch.device('cuda:0')
        
        self.inv_freqs = []
        for section_dim in mrope_section:
            inv_freq = 1.0 / (rope_theta ** (
                torch.arange(0, section_dim * 2, 2, dtype=torch.float32) / (section_dim * 2)
            ))
            self.inv_freqs.append(inv_freq.to(self.device))
        
        self._build_cache(max_position)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        
        cos_parts = []
        sin_parts = []
        
        for inv_freq in self.inv_freqs:
            freqs = torch.outer(t, inv_freq)
            if self.interleaved:
                freqs_interleaved = freqs.repeat_interleave(2, dim=-1)
            else:
                freqs_interleaved = torch.cat([freqs, freqs], dim=-1)
            
            cos_parts.append(freqs_interleaved.cos())
            sin_parts.append(freqs_interleaved.sin())
        
        self.cos_cache = torch.cat(cos_parts, dim=-1).half()
        self.sin_cache = torch.cat(sin_parts, dim=-1).half()
    
    def apply(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)
        
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.interleaved:
            x_reshape = x.reshape(*x.shape[:-1], -1, 2)
            x0, x1 = x_reshape[..., 0], x_reshape[..., 1]
            
            cos_r = cos.reshape(*cos.shape[:-1], -1, 2)[..., 0]
            sin_r = sin.reshape(*sin.shape[:-1], -1, 2)[..., 0]
            
            out0 = x0 * cos_r - x1 * sin_r
            out1 = x0 * sin_r + x1 * cos_r
            
            return torch.stack([out0, out1], dim=-1).reshape(x.shape)
        else:
            half_dim = x.shape[-1] // 2
            x_rot = torch.cat([-x[..., half_dim:], x[..., :half_dim]], dim=-1)
            return x * cos + x_rot * sin


# =============================================================================
# RMS NORM
# =============================================================================

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(orig_dtype)


# =============================================================================
# ATTENTION
# =============================================================================

def forward_attention(
    hidden_states: torch.Tensor, 
    weights: dict, 
    layer_idx: int,
    mrope: MRoPE,
    config: Config
) -> torch.Tensor:
    B, S, H = hidden_states.shape
    prefix = f"{config.layer_prefix}.{layer_idx}."
    
    q_proj = get_weight(weights, f"{prefix}self_attn.q_proj", is_awq=True, group_size=config.awq_group_size)
    k_proj = get_weight(weights, f"{prefix}self_attn.k_proj", is_awq=True, group_size=config.awq_group_size)
    v_proj = get_weight(weights, f"{prefix}self_attn.v_proj", is_awq=True, group_size=config.awq_group_size)
    o_proj = get_weight(weights, f"{prefix}self_attn.o_proj", is_awq=True, group_size=config.awq_group_size)
    
    if q_proj is None:
        return hidden_states
    
    q_bias = weights.get(f"{prefix}self_attn.q_proj.bias")
    k_bias = weights.get(f"{prefix}self_attn.k_proj.bias")
    v_bias = weights.get(f"{prefix}self_attn.v_proj.bias")
    o_bias = weights.get(f"{prefix}self_attn.o_proj.bias")
    
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    
    q = F.linear(hidden_states, q_proj, q_bias).view(B, S, num_heads, head_dim).transpose(1, 2)
    k = F.linear(hidden_states, k_proj, k_bias).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    v = F.linear(hidden_states, v_proj, v_bias).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    
    q_norm = weights.get(f"{prefix}self_attn.q_norm.weight")
    k_norm = weights.get(f"{prefix}self_attn.k_norm.weight")
    if q_norm is not None:
        q = rms_norm(q, q_norm.view(1, 1, 1, -1))
    if k_norm is not None:
        k = rms_norm(k, k_norm.view(1, 1, 1, -1))
    
    q, k = mrope.apply(q, k)
    
    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, num_heads * head_dim)
    
    return F.linear(attn_output, o_proj, o_bias)


# =============================================================================
# MOE WITH REAP SCORING
# =============================================================================

def forward_moe_with_reap(
    hidden_states: torch.Tensor,
    weights: dict,
    layer_idx: int,
    score_accumulator: ReapScoreAccumulator,
    config: Config
) -> torch.Tensor:
    """
    MoE forward with correct REAP scoring.
    
    FIXES in V12.4:
    - Config-aware normalization (norm_topk_prob)
    - Gate weight is FP16 (not AWQ)
    """
    B, S, H = hidden_states.shape
    total_tokens = B * S
    hidden_flat = hidden_states.view(total_tokens, H)
    
    prefix = f"{config.layer_prefix}.{layer_idx}."
    
    # Gate is FP16 (modules_to_not_convert includes "mlp.gate")
    gate_weight = get_weight(weights, f"{prefix}mlp.gate", is_awq=False)
    
    if gate_weight is None:
        return hidden_states
    
    # Router computation
    router_logits = F.linear(hidden_flat, gate_weight)
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    
    # Top-K selection
    top_k = config.num_experts_per_tok
    topk_weights, topk_indices = torch.topk(router_probs, top_k, dim=-1)
    
    # FIX: Config-aware normalization
    if config.norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    topk_weights = topk_weights.half()
    
    # Output tensor
    moe_output = torch.zeros_like(hidden_flat)
    
    # Group tokens by expert
    expert_to_tokens: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for token_idx in range(total_tokens):
        for k_pos in range(top_k):
            exp_idx = topk_indices[token_idx, k_pos].item()
            expert_to_tokens[exp_idx].append((token_idx, k_pos))
    
    # Process each selected expert
    for exp_idx, token_info in expert_to_tokens.items():
        if not token_info:
            continue
        
        exp_prefix = f"{prefix}mlp.experts.{exp_idx}"
        gate_proj = get_weight(weights, f"{exp_prefix}.gate_proj", is_awq=True, group_size=config.awq_group_size)
        up_proj = get_weight(weights, f"{exp_prefix}.up_proj", is_awq=True, group_size=config.awq_group_size)
        down_proj = get_weight(weights, f"{exp_prefix}.down_proj", is_awq=True, group_size=config.awq_group_size)
        
        if gate_proj is None:
            continue
        
        token_indices = [t[0] for t in token_info]
        tokens_tensor = hidden_flat[token_indices]
        
        with torch.amp.autocast('cuda'):
            gate_out = F.silu(F.linear(tokens_tensor, gate_proj))
            up_out = F.linear(tokens_tensor, up_proj)
            expert_out = F.linear(gate_out * up_out, down_proj)
        
        output_norms = torch.norm(expert_out.float(), p=2, dim=-1)
        
        for i, (token_idx, k_pos) in enumerate(token_info):
            gate_value = topk_weights[token_idx, k_pos].float().item()
            output_norm = output_norms[i].item()
            
            score_accumulator.update(layer_idx, exp_idx, gate_value, output_norm)
            moe_output[token_idx] += topk_weights[token_idx, k_pos] * expert_out[i]
    
    return moe_output.view(B, S, H)


# =============================================================================
# LAYER PROCESSOR
# =============================================================================

class LayerProcessor:
    def __init__(self, model_path: str, config: Config):
        self.model_path = model_path
        self.config = config
        
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_index = json.load(f)
        
        self.weight_map = weight_index['weight_map']
        self.shard_to_weights = defaultdict(list)
        for name, shard in self.weight_map.items():
            self.shard_to_weights[shard].append(name)
    
    def _load_layer_weights(self, layer_idx: int) -> dict:
        prefix = f"{self.config.layer_prefix}.{layer_idx}."
        weights = {}
        
        needed_shards = set()
        for name, shard in self.weight_map.items():
            if prefix in name:
                needed_shards.add(shard)
        
        for shard in needed_shards:
            shard_path = os.path.join(self.model_path, shard)
            with safe_open(shard_path, framework="pt", device=str(device)) as f:
                for name in self.shard_to_weights[shard]:
                    if prefix in name:
                        weights[name] = f.get_tensor(name)
        
        return weights
    
    def process_layer(self, hidden_states: torch.Tensor, layer_idx: int,
                      weights: dict, mrope: MRoPE, 
                      score_accumulator: ReapScoreAccumulator) -> torch.Tensor:
        prefix = f"{self.config.layer_prefix}.{layer_idx}."
        
        residual = hidden_states
        input_ln = weights.get(f"{prefix}input_layernorm.weight")
        if input_ln is not None:
            hidden_states = rms_norm(hidden_states, input_ln)
        
        hidden_states = forward_attention(hidden_states, weights, layer_idx, mrope, self.config)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        post_ln = weights.get(f"{prefix}post_attention_layernorm.weight")
        if post_ln is not None:
            hidden_normed = rms_norm(hidden_states, post_ln)
        else:
            hidden_normed = hidden_states
        
        moe_out = forward_moe_with_reap(hidden_normed, weights, layer_idx, score_accumulator, self.config)
        
        return residual + moe_out
    
    def process_all_layers(self, hidden_states_list: List[torch.Tensor],
                           mrope: MRoPE, score_accumulator: ReapScoreAccumulator,
                           progress_callback=None):
        for layer_idx in range(self.config.num_layers):
            layer_start = time.time()
            
            weights = self._load_layer_weights(layer_idx)
            
            for sample_idx, hidden in enumerate(hidden_states_list):
                hidden_batched = hidden.unsqueeze(0)
                
                with torch.no_grad():
                    output = self.process_layer(
                        hidden_batched, layer_idx, weights,
                        mrope, score_accumulator
                    )
                
                hidden_states_list[sample_idx] = output.squeeze(0)
            
            del weights
            torch.cuda.empty_cache()
            
            if progress_callback:
                layer_time = time.time() - layer_start
                counts = score_accumulator.active_count[layer_idx]
                active = (counts > 0).sum().item()
                progress_callback(layer_idx, layer_time, active)


# =============================================================================
# CALIBRATION
# =============================================================================

class CalibrationDataset:
    TEMPLATES = [
        'Gorselde "{region}" nesnesini bul.',
        'Locate "{region}" in the image.',
        'Find the bounding box of "{region}".',
        'Where is "{region}" located?',
        '"{region}" nerede?',
    ]
    
    def __init__(self, json_path: str):
        self.base_dir = os.path.dirname(json_path)
        self.vision_samples = []
        self.text_samples = []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            image_path = os.path.join(self.base_dir, item.get('image', ''))
            if os.path.exists(image_path):
                for region in item.get('regions', []):
                    if region:
                        self.vision_samples.append({
                            'image_path': image_path,
                            'region': region,
                        })
            
            if 'prompt' in item:
                self.text_samples.append({'prompt': item['prompt']})
        
        print(f"  Loaded {len(self.vision_samples)} vision samples")
        print(f"  Loaded {len(self.text_samples)} text samples")
    
    def generate_samples(self, max_samples: int, vision_ratio: float = 0.7) -> List[dict]:
        random.seed(CONFIG.random_seed)
        
        n_vision = min(int(max_samples * vision_ratio), len(self.vision_samples))
        n_text = min(max_samples - n_vision, len(self.text_samples))
        
        vision_selected = random.sample(self.vision_samples, n_vision) if n_vision > 0 else []
        text_selected = random.sample(self.text_samples, n_text) if n_text > 0 else []
        
        results = []
        
        for s in vision_selected:
            template = random.choice(self.TEMPLATES)
            results.append({
                'type': 'vision',
                'image_path': s['image_path'],
                'prompt': template.format(region=s['region']),
            })
        
        for s in text_selected:
            results.append({'type': 'text', 'prompt': s['prompt']})
        
        random.shuffle(results)
        
        print(f"  Generated {len(results)} samples ({n_vision} vision, {n_text} text)")
        return results


# =============================================================================
# VISION ENCODER LOADING
# =============================================================================

def load_vision_encoder(model_path: str, config: Config):
    """Load vision encoder using transformers native implementation."""
    from transformers import AutoConfig
    
    print("  Loading vision encoder from transformers...")
    
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeVisionEncoder
        )
        
        vision_encoder = Qwen3VLMoeVisionEncoder(model_config.vision_config)
        
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_index = json.load(f)
        
        vision_weights = {}
        needed_shards = set()
        
        for name, shard in weight_index['weight_map'].items():
            if config.vision_prefix in name:
                needed_shards.add(shard)
        
        for shard in needed_shards:
            shard_path = os.path.join(model_path, shard)
            with safe_open(shard_path, framework="pt", device=str(device)) as f:
                for name in f.keys():
                    if config.vision_prefix in name:
                        clean_name = name.replace(f"{config.vision_prefix}.", "")
                        vision_weights[clean_name] = f.get_tensor(name)
        
        missing, unexpected = vision_encoder.load_state_dict(vision_weights, strict=False)
        
        if missing:
            print(f"  Warning: Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Warning: Unexpected keys: {len(unexpected)}")
        
        vision_encoder = vision_encoder.to(device).half()
        vision_encoder.eval()
        
        param_count = sum(p.numel() for p in vision_encoder.parameters()) / 1e6
        print(f"  ✓ Vision encoder loaded: {param_count:.1f}M params")
        
        return vision_encoder
        
    except ImportError as e:
        print(f"  Vision encoder import failed: {e}")
        print("  Make sure transformers >= 4.57.0 is installed")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    global CONFIG, device
    
    # Load config from model
    CONFIG = load_config_from_model(CONFIG)
    device = torch.device(CONFIG.device)
    
    print_banner(CONFIG)
    
    # Check transformers version
    check_transformers_version()
    
    # 1. Load calibration data
    print("\n[1/7] Loading calibration data...")
    dataset = CalibrationDataset(CONFIG.calibration_data_path)
    samples = dataset.generate_samples(CONFIG.max_samples)
    
    # 2. Load processor
    print("\n[2/7] Loading processor...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(CONFIG.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    print("  ✓ Processor loaded")
    
    # 3. Load embeddings
    print("\n[3/7] Loading embeddings...")
    
    index_path = os.path.join(CONFIG.model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_index = json.load(f)
    
    embed_key = f"{CONFIG.embed_prefix}.weight"
    embed_shard = weight_index['weight_map'].get(embed_key)
    
    if embed_shard is None:
        for k, v in weight_index['weight_map'].items():
            if 'embed_tokens' in k and 'weight' in k:
                embed_shard = v
                embed_key = k
                break
    
    with safe_open(os.path.join(CONFIG.model_path, embed_shard), framework="pt", device=str(device)) as f:
        embed_weight = f.get_tensor(embed_key)
    
    embed_tokens = nn.Embedding(embed_weight.shape[0], embed_weight.shape[1], device=device)
    embed_tokens.weight.data = embed_weight.half()
    print(f"  ✓ Embeddings: {embed_weight.shape}")
    
    # 4. Load vision encoder
    print("\n[4/7] Loading vision encoder...")
    vision_encoder = load_vision_encoder(CONFIG.model_path, CONFIG)
    
    # 5. Process samples
    print("\n[5/7] Processing calibration samples...")
    all_hidden = []
    vision_count = 0
    text_count = 0
    
    for sample in tqdm(samples, desc="  Encoding"):
        try:
            if sample.get('type') == 'vision' and vision_encoder is not None:
                image = Image.open(sample['image_path']).convert('RGB')
                
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': sample['prompt']}
                ]}]
                
                try:
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(
                        text=[text], images=[image],
                        return_tensors='pt', padding=True
                    )
                    
                    input_ids = inputs['input_ids'].to(device)
                    pixel_values = inputs.get('pixel_values')
                    image_grid_thw = inputs.get('image_grid_thw')
                    
                    with torch.no_grad():
                        text_embeds = embed_tokens(input_ids)
                    
                    if pixel_values is not None:
                        pixel_values = pixel_values.to(device).half()
                        
                        with torch.no_grad():
                            if image_grid_thw is not None:
                                image_grid_thw = image_grid_thw.to(device)
                                vision_embeds = vision_encoder(pixel_values, grid_thw=image_grid_thw)
                            else:
                                vision_embeds = vision_encoder(pixel_values)
                        
                        image_mask = (input_ids[0] == CONFIG.image_token_id)
                        image_positions = torch.where(image_mask)[0]
                        
                        if len(image_positions) > 0 and vision_embeds is not None:
                            if hasattr(vision_embeds, 'last_hidden_state'):
                                v_embeds = vision_embeds.last_hidden_state
                            else:
                                v_embeds = vision_embeds
                            
                            combined = text_embeds.clone()
                            num_vision = min(len(image_positions), v_embeds.shape[1])
                            
                            for i in range(num_vision):
                                combined[0, image_positions[i]] = v_embeds[0, i]
                            
                            hidden = combined.squeeze(0)
                            vision_count += 1
                        else:
                            hidden = text_embeds.squeeze(0)
                            text_count += 1
                    else:
                        hidden = text_embeds.squeeze(0)
                        text_count += 1
                        
                except Exception as e:
                    inputs = tokenizer(
                        sample['prompt'], return_tensors="pt",
                        truncation=True, max_length=CONFIG.max_seq_len
                    ).to(device)
                    
                    with torch.no_grad():
                        hidden = embed_tokens(inputs['input_ids']).squeeze(0)
                    text_count += 1
            else:
                prompt = sample.get('prompt', 'Hello')
                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=CONFIG.max_seq_len
                ).to(device)
                
                with torch.no_grad():
                    hidden = embed_tokens(inputs['input_ids']).squeeze(0)
                text_count += 1
            
            if hidden.shape[0] > CONFIG.max_seq_len:
                hidden = hidden[:CONFIG.max_seq_len]
            
            all_hidden.append(hidden)
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    del embed_tokens
    if vision_encoder is not None:
        del vision_encoder
    torch.cuda.empty_cache()
    
    print(f"  ✓ Processed {len(all_hidden)} samples ({vision_count} vision, {text_count} text)")
    
    if len(all_hidden) == 0:
        print("ERROR: No samples processed!")
        sys.exit(1)
    
    # 6. REAP scoring
    print("\n[6/7] Running REAP scoring...")
    
    mrope = MRoPE(
        head_dim=CONFIG.head_dim,
        max_position=CONFIG.max_seq_len,
        rope_theta=CONFIG.rope_theta,
        mrope_section=CONFIG.mrope_section,
        interleaved=CONFIG.mrope_interleaved,
        device=device
    )
    
    score_accumulator = ReapScoreAccumulator(CONFIG.num_experts, CONFIG.num_layers, device)
    layer_processor = LayerProcessor(CONFIG.model_path, CONFIG)
    
    def progress_cb(layer_idx, layer_time, active_experts):
        if layer_idx % 10 == 0 or layer_idx == CONFIG.num_layers - 1:
            print(f"  Layer {layer_idx+1:3d}/{CONFIG.num_layers}: "
                  f"{active_experts:3d}/{CONFIG.num_experts} active, "
                  f"{layer_time:.1f}s")
    
    start_time = time.time()
    layer_processor.process_all_layers(all_hidden, mrope, score_accumulator, progress_cb)
    total_time = time.time() - start_time
    
    print(f"\n  Total time: {total_time/60:.1f} minutes")
    
    reap_scores = score_accumulator.get_scores()
    expert_counts = score_accumulator.get_counts()
    
    # 7. Save
    print("\n[7/7] Saving results...")
    
    experts_to_keep_count = int(CONFIG.num_experts * (1 - CONFIG.prune_ratio))
    experts_to_keep = {}
    
    for layer_idx in range(CONFIG.num_layers):
        scores = reap_scores[layer_idx].numpy()
        sorted_idx = np.argsort(scores)[::-1]
        experts_to_keep[str(layer_idx)] = sorted(sorted_idx[:experts_to_keep_count].tolist())
    
    os.makedirs(CONFIG.output_path, exist_ok=True)
    
    metadata = {
        'reap_version': 'v12.4_qwen3vl_awq',
        'model': 'Qwen3-VL-235B-A22B-Thinking-AWQ',
        'formula': 'Sⱼ = (1/|𝒳ⱼ|) × Σ(x∈𝒳ⱼ) gⱼ(x) · ‖fⱼ(x)‖₂',
        'prune_ratio': CONFIG.prune_ratio,
        'experts_kept': experts_to_keep_count,
        'num_experts_original': CONFIG.num_experts,
        'num_layers': CONFIG.num_layers,
        'top_k': CONFIG.num_experts_per_tok,
        'rope_theta': CONFIG.rope_theta,
        'mrope_interleaved': CONFIG.mrope_interleaved,
        'mrope_section': CONFIG.mrope_section,
        'norm_topk_prob': CONFIG.norm_topk_prob,
        'calibration_samples': len(all_hidden),
        'vision_samples': vision_count,
        'text_samples': text_count,
        'experts_to_keep': experts_to_keep,
        'reap_scores': {str(k): v.tolist() for k, v in reap_scores.items()},
        'expert_counts': {str(k): v.tolist() for k, v in expert_counts.items()},
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(CONFIG.output_path, 'reap_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    for fname in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 
                  'chat_template.json', 'preprocessor_config.json']:
        src = os.path.join(CONFIG.model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, CONFIG.output_path)
    
    # Analysis
    print(f"\n  === Analysis ===")
    all_scores = np.concatenate([reap_scores[i].numpy() for i in range(CONFIG.num_layers)])
    print(f"  Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"  Score mean: {all_scores.mean():.4f}")
    
    zero_experts = sum((expert_counts[i] == 0).sum().item() for i in range(CONFIG.num_layers))
    print(f"  Zero-activation experts: {zero_experts}/{CONFIG.num_experts * CONFIG.num_layers}")
    
    print(f"\n{'='*70}")
    print(f"  COMPLETE!")
    print(f"  Output: {CONFIG.output_path}")
    print(f"  Experts: {CONFIG.num_experts} -> {experts_to_keep_count}")
    print(f"\n  Next: python prune_model_v12.4.py --metadata {CONFIG.output_path}/reap_metadata.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()