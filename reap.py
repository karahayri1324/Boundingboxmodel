#!/usr/bin/env python3
"""
REAP V11 - VLM Expert Pruning (Fixed Version)
==============================================

CRITICAL FIXES from V10:
1. compute_all_experts=True by default (was False - WRONG!)
2. Router-only approximation mode for speed (optional)
3. Expert distribution analysis before pruning
4. Better numerical stability

REAP Score = mean(router_prob * ||expert_output||_L2) per expert

Based on CerebrasResearch/reap methodology.
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
from datetime import datetime, timedelta
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

sys.stdout.reconfigure(line_buffering=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# CONFIGURATION
# =============================================================================

class ScoringMode(Enum):
    FULL = "full"           # Compute ALL expert outputs (accurate, slower)
    ROUTER_ONLY = "router"  # Use router probabilities only (fast approx)


@dataclass
class ReapConfig:
    """REAP V11 Configuration - Fixed version."""
    # Paths
    model_path: str = os.environ.get(
        "REAP_MODEL_PATH",
        "/mnt/vault/boundingboxtest/Qwen3-VL-235B-A22B-Thinking-AWQ"
    )
    output_path: str = os.environ.get(
        "REAP_OUTPUT_PATH",
        "/mnt/vault/boundingboxtest/qwen3vl-235b-reap-v11"
    )
    calibration_data_path: str = os.environ.get(
        "REAP_CALIBRATION_PATH",
        "/mnt/vault/boundingboxtest/reap_calibration_data/calibration_data.json"
    )
    
    # Pruning settings
    prune_ratio: float = float(os.environ.get("REAP_PRUNE_RATIO", "0.40"))
    max_samples: int = int(os.environ.get("REAP_MAX_SAMPLES", "300"))
    max_seq_len: int = int(os.environ.get("REAP_MAX_SEQ_LEN", "2048"))
    random_seed: int = int(os.environ.get("REAP_SEED", "42"))
    group_size: int = 128
    
    # CRITICAL: Scoring mode - "full" computes ALL expert outputs
    scoring_mode: str = os.environ.get("REAP_SCORING_MODE", "full")
    
    # Memory settings
    total_vram_gb: float = float(os.environ.get("REAP_VRAM_GB", "140.0"))
    vram_buffer_ratio: float = float(os.environ.get("REAP_VRAM_BUFFER", "0.10"))
    batch_size: int = int(os.environ.get("REAP_BATCH_SIZE", "1"))
    
    device: str = os.environ.get("REAP_DEVICE", "cuda:0")
    analyze_distribution: bool = True


CONFIG = ReapConfig()
device = torch.device(CONFIG.device)


def print_config():
    print("\n" + "="*70)
    print("  REAP V11 Configuration (FIXED)")
    print("="*70)
    print(f"  Model: {CONFIG.model_path}")
    print(f"  Output: {CONFIG.output_path}")
    print(f"  Prune Ratio: {CONFIG.prune_ratio*100:.0f}%")
    print(f"  Scoring Mode: {CONFIG.scoring_mode.upper()}")
    if CONFIG.scoring_mode == "full":
        print(f"    → Computing ALL 128 expert outputs (CORRECT method)")
    else:
        print(f"    → Router-only approximation (fast but less accurate)")
    print("="*70 + "\n")


# =============================================================================
# ONLINE STATISTICS TRACKER
# =============================================================================

class OnlineStatsTracker:
    """Welford's algorithm for numerically stable online mean."""
    
    def __init__(self, shape: tuple, device: torch.device):
        self.count = torch.zeros(shape, dtype=torch.long, device=device)
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.device = device
        
    def update(self, values: torch.Tensor, counts: torch.Tensor):
        values = values.to(self.device, torch.float64)
        counts = counts.to(self.device, torch.long)
        
        mask = counts > 0
        if not mask.any():
            return
            
        new_count = self.count + counts
        delta = torch.zeros_like(self.mean)
        delta[mask] = values[mask] - self.mean[mask]
        
        safe_count = new_count.float().clamp(min=1)
        self.mean[mask] += delta[mask] * counts[mask].float() / safe_count[mask]
        self.count = new_count
        
    def get_mean(self) -> torch.Tensor:
        return self.mean.clone()
    
    def get_count(self) -> torch.Tensor:
        return self.count.clone()


# =============================================================================
# AWQ DEQUANTIZATION
# =============================================================================

def dequantize_awq(qweight, qzeros, scales, group_size=128):
    in_features = qweight.shape[0]
    out_features = scales.shape[1]
    num_groups = scales.shape[0]
    
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    weight_int = ((qweight.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF).reshape(in_features, out_features)
    zeros_int = ((qzeros.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF).reshape(num_groups, out_features)
    
    zeros_expanded = zeros_int.repeat_interleave(group_size, dim=0)[:in_features]
    scales_expanded = scales.repeat_interleave(group_size, dim=0)[:in_features]
    
    weight = (weight_int.float() - zeros_expanded.float()) * scales_expanded.float()
    return weight.T.half()


def get_weight(weights: dict, prefix: str) -> Optional[torch.Tensor]:
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
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(orig_dtype)


# =============================================================================
# ATTENTION
# =============================================================================

def forward_attention(hidden_states: torch.Tensor, weights: dict, prefix: str,
                     rope: RotaryEmbedding, num_heads: int, num_kv_heads: int,
                     head_dim: int) -> torch.Tensor:
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
    
    q_norm = weights.get(f"{prefix}self_attn.q_norm.weight")
    k_norm = weights.get(f"{prefix}self_attn.k_norm.weight")
    if q_norm is not None:
        q = rms_norm(q, q_norm.view(1, 1, 1, -1))
    if k_norm is not None:
        k = rms_norm(k, k_norm.view(1, 1, 1, -1))
        
    q, k = rope.apply(q, k)
    
    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, num_heads * head_dim)
    return F.linear(attn_output, o_proj, o_bias)


# =============================================================================
# MOE WITH REAP SCORING - FULL MODE (CORRECT!)
# =============================================================================

def forward_moe_full_reap(
    hidden_states: torch.Tensor, 
    weights: dict, 
    prefix: str,
    num_experts: int, 
    top_k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CORRECT REAP implementation - computes ALL expert outputs.
    
    This matches CerebrasResearch/reap methodology:
    REAP Score = mean(router_prob * ||expert_output||_L2)
    """
    B, S, H = hidden_states.shape
    total_tokens = B * S
    hidden_flat = hidden_states.view(total_tokens, H)
    
    gate_weight = get_weight(weights, f"{prefix}mlp.gate")
    if gate_weight is None:
        zeros = torch.zeros(num_experts, device=device)
        return hidden_states, zeros, zeros.long()
    
    # Router
    router_logits = F.linear(hidden_flat, gate_weight)
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    
    # Top-K for output
    topk_weights, topk_indices = torch.topk(router_probs, top_k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).half()
    
    moe_output = torch.zeros_like(hidden_flat)
    expert_scores = torch.zeros(num_experts, dtype=torch.float64, device=device)
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    
    # CRITICAL: Compute ALL expert outputs
    for exp_idx in range(num_experts):
        gate_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.gate_proj")
        up_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.up_proj")
        down_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.down_proj")
        
        if gate_proj is None:
            continue
        
        with torch.amp.autocast('cuda'):
            gate_out = F.silu(F.linear(hidden_flat, gate_proj))
            up_out = F.linear(hidden_flat, up_proj)
            expert_out = F.linear(gate_out * up_out, down_proj)
        
        # L2 norms
        output_norms = torch.norm(expert_out.float(), p=2, dim=-1)
        
        # Router prob for this expert
        expert_probs = router_probs[:, exp_idx]
        
        # REAP: prob-weighted norm
        weighted_norms = expert_probs.float() * output_norms
        expert_scores[exp_idx] = weighted_norms.sum().item()
        expert_counts[exp_idx] = total_tokens
        
        # Add to output if in top-k
        for token_idx in range(total_tokens):
            if exp_idx in topk_indices[token_idx]:
                slot = (topk_indices[token_idx] == exp_idx).nonzero(as_tuple=True)[0]
                moe_output[token_idx] += topk_weights[token_idx, slot] * expert_out[token_idx]
        
        if (exp_idx + 1) % 16 == 0:
            del gate_out, up_out, expert_out
            torch.cuda.empty_cache()
    
    return moe_output.view(B, S, H), expert_scores, expert_counts


def forward_moe_router_only(
    hidden_states: torch.Tensor, 
    weights: dict, 
    prefix: str,
    num_experts: int, 
    top_k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast approximation using router probabilities only.
    Less accurate but much faster.
    """
    B, S, H = hidden_states.shape
    total_tokens = B * S
    hidden_flat = hidden_states.view(total_tokens, H)
    
    gate_weight = get_weight(weights, f"{prefix}mlp.gate")
    if gate_weight is None:
        zeros = torch.zeros(num_experts, device=device)
        return hidden_states, zeros, zeros.long()
    
    router_logits = F.linear(hidden_flat, gate_weight)
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    
    # Approximate: prob * hidden_norm
    hidden_norms = torch.norm(hidden_flat.float(), p=2, dim=-1, keepdim=True)
    approx_scores = router_probs * hidden_norms
    
    expert_scores = approx_scores.sum(dim=0).double()
    expert_counts = torch.full((num_experts,), total_tokens, dtype=torch.long, device=device)
    
    # Compute output (top-k only)
    topk_weights, topk_indices = torch.topk(router_probs, top_k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).half()
    
    moe_output = torch.zeros_like(hidden_flat)
    
    for exp_idx in topk_indices.unique().tolist():
        gate_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.gate_proj")
        up_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.up_proj")
        down_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.down_proj")
        
        if gate_proj is None:
            continue
        
        mask = (topk_indices == exp_idx).any(dim=-1)
        tokens = hidden_flat[mask]
        
        if tokens.shape[0] == 0:
            continue
        
        with torch.amp.autocast('cuda'):
            g = F.silu(F.linear(tokens, gate_proj))
            u = F.linear(tokens, up_proj)
            out = F.linear(g * u, down_proj)
        
        positions = torch.where(mask)[0]
        for i, pos in enumerate(positions):
            slot_mask = (topk_indices[pos] == exp_idx)
            w = topk_weights[pos][slot_mask].sum()
            moe_output[pos] += w * out[i]
    
    return moe_output.view(B, S, H), expert_scores, expert_counts


# =============================================================================
# LAYER PROCESSOR
# =============================================================================

class LayerProcessor:
    def __init__(self, model_path: str, config: dict, num_layers: int):
        self.model_path = model_path
        self.config = config
        self.num_layers = num_layers
        
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_index = json.load(f)
        
        self.weight_map = weight_index['weight_map']
        self.shard_to_weights = defaultdict(list)
        for name, shard in self.weight_map.items():
            self.shard_to_weights[shard].append(name)
    
    def _load_layer_weights(self, layer_idx: int) -> dict:
        prefix = f"model.language_model.layers.{layer_idx}."
        weights = {}
        
        needed_shards = set()
        for name, shard in self.weight_map.items():
            if prefix in name:
                needed_shards.add(shard)
        
        for shard in needed_shards:
            shard_path = os.path.join(self.model_path, shard)
            with safe_open(shard_path, framework="pt", device="cuda:0") as f:
                for name in self.shard_to_weights[shard]:
                    if prefix in name:
                        weights[name] = f.get_tensor(name)
        
        return weights
    
    def process_all_layers(
        self, 
        hidden_states_list: List[torch.Tensor],
        rope: RotaryEmbedding,
        use_full_mode: bool = True,
        progress_callback=None
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        
        num_experts = self.config['num_experts']
        reap_scores = {}
        expert_counts = {}
        
        for layer_idx in range(self.num_layers):
            layer_start = time.time()
            
            layer_scores = torch.zeros(num_experts, dtype=torch.float64, device=device)
            layer_counts = torch.zeros(num_experts, dtype=torch.long, device=device)
            
            weights = self._load_layer_weights(layer_idx)
            
            for sample_idx, hidden in enumerate(hidden_states_list):
                hidden_batched = hidden.unsqueeze(0)
                
                with torch.no_grad():
                    output, scores, counts = self._forward_layer(
                        hidden_batched, weights, layer_idx, rope, use_full_mode
                    )
                
                layer_scores += scores
                layer_counts += counts
                hidden_states_list[sample_idx] = output.squeeze(0)
            
            mean_scores = layer_scores / layer_counts.float().clamp(min=1)
            reap_scores[layer_idx] = mean_scores.cpu()
            expert_counts[layer_idx] = layer_counts.cpu()
            
            del weights
            torch.cuda.empty_cache()
            
            if progress_callback:
                progress_callback(layer_idx, time.time() - layer_start, mean_scores)
        
        return reap_scores, expert_counts
    
    def _forward_layer(
        self, 
        hidden_states: torch.Tensor, 
        weights: dict,
        layer_idx: int, 
        rope: RotaryEmbedding,
        use_full_mode: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        prefix = f"model.language_model.layers.{layer_idx}."
        
        # Attention
        residual = hidden_states
        input_ln = weights.get(f"{prefix}input_layernorm.weight")
        if input_ln is not None:
            hidden_states = rms_norm(hidden_states, input_ln)
        
        hidden_states = forward_attention(
            hidden_states, weights, prefix, rope,
            self.config['num_attention_heads'],
            self.config['num_key_value_heads'],
            self.config['head_dim']
        )
        hidden_states = residual + hidden_states
        
        # MoE
        residual = hidden_states
        post_ln = weights.get(f"{prefix}post_attention_layernorm.weight")
        if post_ln is not None:
            hidden_normed = rms_norm(hidden_states, post_ln)
        else:
            hidden_normed = hidden_states
        
        if use_full_mode:
            moe_out, scores, counts = forward_moe_full_reap(
                hidden_normed, weights, prefix,
                self.config['num_experts'],
                self.config['num_experts_per_tok']
            )
        else:
            moe_out, scores, counts = forward_moe_router_only(
                hidden_normed, weights, prefix,
                self.config['num_experts'],
                self.config['num_experts_per_tok']
            )
        
        hidden_states = residual + moe_out
        return hidden_states, scores, counts


# =============================================================================
# VISION ENCODER
# =============================================================================

def load_vision_encoder(model_path: str):
    try:
        from transformers import AutoConfig
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLVisionEncoder
        
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        visual = Qwen3VLVisionEncoder(config.vision_config).to(device).half()
        
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_index = json.load(f)
        
        vision_shard = None
        for name, shard in weight_index['weight_map'].items():
            if 'visual' in name:
                vision_shard = os.path.join(model_path, shard)
                break
        
        if vision_shard:
            with safe_open(vision_shard, framework="pt", device="cuda:0") as f:
                state_dict = {k.replace("model.visual.", ""): f.get_tensor(k) 
                              for k in f.keys() if k.startswith("model.visual.")}
            visual.load_state_dict(state_dict, strict=False)
        
        visual.eval()
        return visual
    except Exception as e:
        print(f"  Vision encoder error: {e}")
        return None


# =============================================================================
# CALIBRATION DATASET
# =============================================================================

class CalibrationDataset:
    TEMPLATES = [
        'Gorselde "{region}" nesnesini bul.',
        'Locate "{region}" in the image.',
        '"{region}" konumunu tespit et.',
    ]
    
    def __init__(self, json_path: str):
        self.base_dir = os.path.dirname(json_path)
        self.samples = []
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            image_path = os.path.join(self.base_dir, item.get('image', ''))
            if os.path.exists(image_path):
                for region in item.get('regions', []):
                    if region:
                        self.samples.append({
                            'type': 'vision',
                            'image_path': image_path,
                            'region': region,
                        })
            elif 'prompt' in item:
                self.samples.append({
                    'type': 'text',
                    'prompt': item['prompt'],
                })
    
    def generate_samples(self, max_samples: Optional[int] = None) -> List[dict]:
        samples = self.samples
        if max_samples and len(samples) > max_samples:
            random.seed(CONFIG.random_seed)
            samples = random.sample(samples, max_samples)
        
        results = []
        for s in samples:
            if s['type'] == 'vision':
                template = random.choice(self.TEMPLATES)
                results.append({
                    'type': 'vision',
                    'image_path': s['image_path'],
                    'prompt': template.format(region=s['region']),
                })
            else:
                results.append(s)
        return results


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_distribution(reap_scores, expert_counts, num_experts, num_layers):
    print("\n" + "="*70)
    print("  EXPERT DISTRIBUTION ANALYSIS")
    print("="*70)
    
    all_counts = torch.stack([expert_counts[i] for i in range(num_layers)])
    total_counts = all_counts.sum(dim=0)
    
    zero_experts = (total_counts == 0).sum().item()
    
    print(f"\n  Zero-count experts: {zero_experts}/{num_experts}")
    
    if zero_experts > num_experts * 0.3:
        print(f"  ⚠️ WARNING: Many experts have zero counts!")
        print(f"     Ensure REAP_SCORING_MODE=full is set.")
    
    print("\n  Per-Layer Stats:")
    for layer_idx in [0, num_layers//2, num_layers-1]:
        counts = expert_counts[layer_idx]
        scores = reap_scores[layer_idx]
        active = (counts > 0).sum().item()
        print(f"    Layer {layer_idx}: {active}/128 active")
    
    print("="*70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  REAP V11 - Fixed VLM Expert Pruning")
    print("  CRITICAL FIX: compute_all_experts=True")
    print("="*70)
    
    print_config()
    
    use_full_mode = CONFIG.scoring_mode.lower() == "full"
    
    # 1. Load config
    print("\n[1/6] Loading configuration...")
    
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
    EXPERTS_TO_KEEP = NUM_EXPERTS - int(NUM_EXPERTS * CONFIG.prune_ratio)
    
    print(f"  Layers: {NUM_LAYERS}, Experts: {NUM_EXPERTS} -> {EXPERTS_TO_KEEP}")
    
    # 2. Load data
    print("\n[2/6] Loading calibration data...")
    dataset = CalibrationDataset(CONFIG.calibration_data_path)
    samples = dataset.generate_samples(CONFIG.max_samples)
    print(f"  Samples: {len(samples)}")
    
    # 3. Load processor
    print("\n[3/6] Loading processor...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(CONFIG.model_path, trust_remote_code=True)
    
    index_path = os.path.join(CONFIG.model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_index = json.load(f)
    
    embed_shard = weight_index['weight_map']["model.language_model.embed_tokens.weight"]
    with safe_open(os.path.join(CONFIG.model_path, embed_shard), framework="pt", device="cuda:0") as f:
        embed_weight = f.get_tensor("model.language_model.embed_tokens.weight")
    
    embed_tokens = nn.Embedding(embed_weight.shape[0], embed_weight.shape[1], device=device)
    embed_tokens.weight.data = embed_weight.half()
    
    # 4. Vision encoder
    print("\n[4/6] Loading vision encoder...")
    visual = load_vision_encoder(CONFIG.model_path)
    use_vision = visual is not None
    
    IMAGE_TOKEN_ID = config.get('image_token_id', 151655)
    
    # 5. Process samples
    print("\n[5/6] Processing samples...")
    all_hidden = []
    
    for sample in tqdm(samples, desc="  Processing"):
        try:
            if sample.get('type') == 'vision' and use_vision:
                image = Image.open(sample['image_path']).convert('RGB')
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': sample['prompt']}
                ]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors='pt', padding=True)
                
                input_ids = inputs['input_ids'].to(device)
                pixel_values = inputs['pixel_values'].to(device).half()
                image_grid_thw = inputs['image_grid_thw'].to(device)
                
                with torch.no_grad():
                    vision_embeds = visual(pixel_values, grid_thw=image_grid_thw)
                    text_embeds = embed_tokens(input_ids)
                
                image_mask = (input_ids[0] == IMAGE_TOKEN_ID)
                image_pos = torch.where(image_mask)[0]
                
                combined = text_embeds.clone()
                if len(image_pos) > 0:
                    for i in range(min(len(image_pos), vision_embeds.shape[1])):
                        combined[0, image_pos[i]] = vision_embeds[0, i]
                
                hidden = combined.squeeze(0)
            else:
                inputs = processor.tokenizer(sample['prompt'], return_tensors="pt",
                                            truncation=True, max_length=CONFIG.max_seq_len).to(device)
                with torch.no_grad():
                    hidden = embed_tokens(inputs['input_ids']).squeeze(0)
            
            if hidden.shape[0] > CONFIG.max_seq_len:
                hidden = hidden[:CONFIG.max_seq_len]
            
            all_hidden.append(hidden)
        except Exception as e:
            continue
    
    del embed_tokens
    if visual:
        del visual
    torch.cuda.empty_cache()
    
    print(f"  Processed: {len(all_hidden)} samples")
    
    # 6. REAP scoring
    print("\n[6/6] Running REAP scoring...")
    
    rope = RotaryEmbedding(model_config['head_dim'], CONFIG.max_seq_len, model_config['rope_theta'])
    
    layer_processor = LayerProcessor(CONFIG.model_path, model_config, NUM_LAYERS)
    
    def progress_cb(layer_idx, layer_time, scores):
        active = (scores > 0).sum().item()
        print(f"  Layer {layer_idx+1}/{NUM_LAYERS}: {active}/128 active, {layer_time:.1f}s")
    
    reap_scores, expert_counts = layer_processor.process_all_layers(
        all_hidden, rope, use_full_mode=use_full_mode, progress_callback=progress_cb
    )
    
    # Analysis
    if CONFIG.analyze_distribution:
        analyze_distribution(reap_scores, expert_counts, NUM_EXPERTS, NUM_LAYERS)
    
    # Save
    print("Saving results...")
    
    mean_scores_np = np.zeros((NUM_LAYERS, NUM_EXPERTS))
    counts_np = np.zeros((NUM_LAYERS, NUM_EXPERTS))
    
    for i in range(NUM_LAYERS):
        mean_scores_np[i] = reap_scores[i].numpy()
        counts_np[i] = expert_counts[i].numpy()
    
    experts_to_keep = {}
    for i in range(NUM_LAYERS):
        sorted_idx = np.argsort(mean_scores_np[i])[::-1]
        experts_to_keep[i] = sorted_idx[:EXPERTS_TO_KEEP].tolist()
    
    os.makedirs(CONFIG.output_path, exist_ok=True)
    
    metadata = {
        'reap_version': 'v11_fixed',
        'scoring_mode': CONFIG.scoring_mode,
        'critical_fix': 'Computes ALL expert outputs, not just top-k',
        'prune_ratio': CONFIG.prune_ratio,
        'experts_kept': EXPERTS_TO_KEEP,
        'num_experts_original': NUM_EXPERTS,
        'num_layers': NUM_LAYERS,
        'experts_to_keep': {str(k): v for k, v in experts_to_keep.items()},
        'mean_scores': mean_scores_np.tolist(),
        'expert_counts': counts_np.tolist(),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(CONFIG.output_path, 'reap_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    for fname in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        src = os.path.join(CONFIG.model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, CONFIG.output_path)
    
    print(f"\n{'='*70}")
    print(f"  COMPLETE! Output: {CONFIG.output_path}")
    print(f"  Next: python prune_model.py --verify")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
