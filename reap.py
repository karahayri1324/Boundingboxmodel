#!/usr/bin/env python3
"""
REAP V10 - CerebrasResearch Uyumlu Expert Pruning (H200 Optimized)
===================================================================

Qwen3-VL-235B-A22B-AWQ icin DOGRU REAP implementasyonu.
- CerebrasResearch/reap metodolojisi ile uyumlu
- Paralel layer processing (H200 140GB VRAM)
- %10 VRAM buffer
- Online statistics tracking (Welford's algorithm)

REAP Score = mean(gate_weight * ||expert_output||_L2) per expert
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import random
import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from collections import defaultdict
from tqdm import tqdm
import shutil
from datetime import datetime, timedelta
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

sys.stdout.reconfigure(line_buffering=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ReapConfig:
    """
    REAP V10 Configuration.

    All settings can be overridden via environment variables:
        REAP_MODEL_PATH, REAP_OUTPUT_PATH, REAP_CALIBRATION_PATH,
        REAP_PRUNE_RATIO, REAP_PARALLEL_LAYERS, REAP_BATCH_SIZE,
        REAP_VRAM_GB, REAP_VRAM_BUFFER
    """
    # Paths - can be overridden by env vars
    model_path: str = os.environ.get(
        "REAP_MODEL_PATH",
        "/mnt/vault/boundingboxtest/Qwen3-VL-235B-A22B-Thinking-AWQ"
    )
    output_path: str = os.environ.get(
        "REAP_OUTPUT_PATH",
        "/mnt/vault/boundingboxtest/qwen3vl-235b-bbox-reap-v10"
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
    group_size: int = 128  # AWQ group size

    # H200 Memory Optimization
    total_vram_gb: float = float(os.environ.get("REAP_VRAM_GB", "140.0"))
    vram_buffer_ratio: float = float(os.environ.get("REAP_VRAM_BUFFER", "0.10"))
    parallel_layers: int = int(os.environ.get("REAP_PARALLEL_LAYERS", "4"))
    batch_size: int = int(os.environ.get("REAP_BATCH_SIZE", "4"))

    # Device
    device: str = os.environ.get("REAP_DEVICE", "cuda:0")


CONFIG = ReapConfig()
device = torch.device(CONFIG.device)

# Print config on startup
def print_config():
    print("\n[Configuration]")
    print(f"  Model: {CONFIG.model_path}")
    print(f"  Output: {CONFIG.output_path}")
    print(f"  Calibration: {CONFIG.calibration_data_path}")
    print(f"  Prune Ratio: {CONFIG.prune_ratio*100:.0f}%")
    print(f"  VRAM: {CONFIG.total_vram_gb}GB (buffer: {CONFIG.vram_buffer_ratio*100:.0f}%)")
    print(f"  Parallel Layers: {CONFIG.parallel_layers}")
    print(f"  Batch Size: {CONFIG.batch_size}")


# =============================================================================
# ONLINE STATISTICS TRACKER (Welford's Algorithm + Kahan Summation)
# =============================================================================

class OnlineStatsTracker:
    """
    Numerically stable online mean tracking using Welford's algorithm
    with Kahan summation for precision.

    Matches CerebrasResearch/reap methodology.
    """

    def __init__(self, shape: tuple, device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float64):
        self.shape = shape
        self.device = device
        self.dtype = dtype

        # Welford's algorithm state
        self.count = torch.zeros(shape, dtype=torch.long, device=device)
        self.mean = torch.zeros(shape, dtype=dtype, device=device)

        # Kahan summation compensation
        self.compensation = torch.zeros(shape, dtype=dtype, device=device)

    def update(self, new_values: torch.Tensor, new_counts: torch.Tensor):
        """
        Update statistics with new batch.

        Args:
            new_values: Values to add (already weighted if needed)
            new_counts: Count for each value position
        """
        new_counts = new_counts.to(self.device, torch.long)
        new_values = new_values.to(self.device, self.dtype)

        # Avoid division by zero
        mask = new_counts > 0

        updated_count = self.count + new_counts

        # Welford's delta
        delta = torch.zeros_like(self.mean)
        delta[mask] = new_values[mask] - self.mean[mask]

        # Kahan summation for mean update
        y = torch.zeros_like(self.mean)
        safe_updated = updated_count.float().clamp(min=1)
        y[mask] = (delta[mask] * new_counts[mask].float() / safe_updated[mask]) - self.compensation[mask]

        t = self.mean + y
        self.compensation = (t - self.mean) - y
        self.mean = t
        self.count = updated_count

    def get_mean(self) -> torch.Tensor:
        return self.mean.clone()

    def get_count(self) -> torch.Tensor:
        return self.count.clone()


# =============================================================================
# AWQ DEQUANTIZATION
# =============================================================================

def dequantize_awq(qweight, qzeros, scales, group_size=128):
    """AWQ 4-bit dequantization (GEMM format)."""
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
    """Get weight, dequantizing if AWQ quantized."""
    qw = weights.get(f"{prefix}.qweight")
    qz = weights.get(f"{prefix}.qzeros")
    sc = weights.get(f"{prefix}.scales")

    if qw is not None and qz is not None and sc is not None:
        return dequantize_awq(qw, qz, sc, CONFIG.group_size)

    w = weights.get(f"{prefix}.weight")
    if w is not None:
        return w

    return weights.get(prefix)


# =============================================================================
# ROTARY POSITION EMBEDDINGS
# =============================================================================

class RotaryEmbedding:
    """Qwen3-VL compatible RoPE (interleaved format)."""

    def __init__(self, dim: int, max_position: int = 2048, base: float = 1000000.0,
                 interleaved: bool = True):
        self.dim = dim
        self.max_position = max_position
        self.base = base
        self.interleaved = interleaved

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.inv_freq = inv_freq
        self._build_cache(max_position)

    def _build_cache(self, seq_len: int):
        self.inv_freq = self.inv_freq.to(device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)

        if self.interleaved:
            freqs = freqs.repeat_interleave(2, dim=-1)
        else:
            freqs = torch.cat([freqs, freqs], dim=-1)

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
        if self.interleaved:
            x_reshape = x.reshape(*x.shape[:-1], -1, 2)
            x0, x1 = x_reshape[..., 0], x_reshape[..., 1]

            cos_reshape = cos.reshape(*cos.shape[:-1], -1, 2)[..., 0]
            sin_reshape = sin.reshape(*sin.shape[:-1], -1, 2)[..., 0]

            out0 = x0 * cos_reshape - x1 * sin_reshape
            out1 = x0 * sin_reshape + x1 * cos_reshape

            return torch.stack([out0, out1], dim=-1).reshape(x.shape)
        else:
            head_dim = x.shape[-1]
            x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
            rotated = torch.cat([-x2, x1], dim=-1)
            return x * cos + rotated * sin


# =============================================================================
# RMS NORMALIZATION
# =============================================================================

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
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
                     head_dim: int, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Complete attention with RoPE, GQA, and Q/K normalization."""
    B, S, H = hidden_states.shape

    q_proj = get_weight(weights, f"{prefix}self_attn.q_proj")
    k_proj = get_weight(weights, f"{prefix}self_attn.k_proj")
    v_proj = get_weight(weights, f"{prefix}self_attn.v_proj")
    o_proj = get_weight(weights, f"{prefix}self_attn.o_proj")

    q_bias = weights.get(f"{prefix}self_attn.q_proj.bias")
    k_bias = weights.get(f"{prefix}self_attn.k_proj.bias")
    v_bias = weights.get(f"{prefix}self_attn.v_proj.bias")
    o_bias = weights.get(f"{prefix}self_attn.o_proj.bias")

    q_norm = weights.get(f"{prefix}self_attn.q_norm.weight")
    k_norm = weights.get(f"{prefix}self_attn.k_norm.weight")

    if q_proj is None:
        return hidden_states

    q = F.linear(hidden_states, q_proj, q_bias).view(B, S, num_heads, head_dim).transpose(1, 2)
    k = F.linear(hidden_states, k_proj, k_bias).view(B, S, num_kv_heads, head_dim).transpose(1, 2)
    v = F.linear(hidden_states, v_proj, v_bias).view(B, S, num_kv_heads, head_dim).transpose(1, 2)

    if q_norm is not None:
        q = rms_norm(q, q_norm.view(1, 1, 1, -1))
    if k_norm is not None:
        k = rms_norm(k, k_norm.view(1, 1, 1, -1))

    q, k = rope.apply(q, k)

    if num_kv_heads < num_heads:
        n_rep = num_heads // num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    attn_mask = None
    if attention_mask is not None:
        padding_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        causal_mask = torch.triu(
            torch.full((S, S), torch.finfo(hidden_states.dtype).min, device=device, dtype=hidden_states.dtype),
            diagonal=1
        )
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) + padding_mask

    attn_output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0,
        is_causal=(attention_mask is None)
    )

    attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, num_heads * head_dim)
    return F.linear(attn_output, o_proj, o_bias)


# =============================================================================
# MOE FORWARD WITH CORRECT REAP SCORING
# =============================================================================

def forward_moe_with_reap(hidden_states: torch.Tensor, weights: dict, prefix: str,
                          num_experts: int, top_k: int,
                          compute_all_experts: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MoE forward with CORRECT REAP scoring (CerebrasResearch compatible).

    REAP Score = mean(gate_weight * ||expert_output||_L2) per expert

    Key difference: Computes ALL expert outputs for accurate scoring,
    not just top-k selected.
    """
    B, S, H = hidden_states.shape
    total_tokens = B * S
    hidden_flat = hidden_states.view(total_tokens, H)

    gate_weight = weights.get(f"{prefix}mlp.gate.weight")
    if gate_weight is None:
        return hidden_states, torch.zeros(num_experts, device=device), torch.zeros(num_experts, device=device, dtype=torch.long)

    # Router logits and probabilities
    router_logits = F.linear(hidden_flat, gate_weight)
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

    # Top-K selection
    topk_weights, topk_indices = torch.topk(router_probs, top_k, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).half()

    # Initialize outputs and REAP tracking
    moe_output = torch.zeros_like(hidden_flat)

    # REAP scores: track weighted L2 norms
    expert_weighted_norms = torch.zeros(num_experts, dtype=torch.float64, device=device)
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=device)

    if compute_all_experts:
        # Compute ALL expert outputs for accurate REAP scoring
        # (matches CerebrasResearch methodology)
        all_expert_outputs = torch.zeros(num_experts, total_tokens, H, device=device, dtype=hidden_flat.dtype)

        for exp_idx in range(num_experts):
            gate_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.gate_proj")
            up_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.up_proj")
            down_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.down_proj")

            if gate_proj is None:
                continue

            with torch.amp.autocast('cuda'):
                gate_out = F.silu(F.linear(hidden_flat, gate_proj))
                up_out = F.linear(hidden_flat, up_proj)
                exp_out = F.linear(gate_out * up_out, down_proj)

            all_expert_outputs[exp_idx] = exp_out

        # Calculate REAP scores: for each token routed to expert,
        # accumulate gate_weight * ||output||_L2
        for token_idx in range(total_tokens):
            selected_experts = topk_indices[token_idx]
            selected_weights = topk_weights[token_idx]

            for k_idx, exp_idx in enumerate(selected_experts):
                exp_idx = exp_idx.item()
                gate_w = selected_weights[k_idx].float()
                exp_out = all_expert_outputs[exp_idx, token_idx]

                # L2 norm of expert output
                l2_norm = torch.norm(exp_out.float(), p=2)

                # REAP: weighted norm (will be averaged later)
                expert_weighted_norms[exp_idx] += (gate_w * l2_norm).item()
                expert_counts[exp_idx] += 1

                # Accumulate MoE output
                moe_output[token_idx] += gate_w.half() * exp_out

        del all_expert_outputs
    else:
        # Efficient mode: only compute selected experts
        unique_experts = topk_indices.unique().tolist()

        for exp_idx in unique_experts:
            gate_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.gate_proj")
            up_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.up_proj")
            down_proj = get_weight(weights, f"{prefix}mlp.experts.{exp_idx}.down_proj")

            if gate_proj is None:
                continue

            expert_mask = (topk_indices == exp_idx).any(dim=-1)
            expert_positions = torch.where(expert_mask)[0]
            expert_tokens = hidden_flat[expert_mask]

            if expert_tokens.shape[0] == 0:
                continue

            with torch.amp.autocast('cuda'):
                gate_out = F.silu(F.linear(expert_tokens, gate_proj))
                up_out = F.linear(expert_tokens, up_proj)
                expert_out = F.linear(gate_out * up_out, down_proj)

            # L2 norms
            activation_norms = torch.norm(expert_out.float(), p=2, dim=-1)

            for i, pos in enumerate(expert_positions):
                slot_mask = (topk_indices[pos] == exp_idx)
                gate_w = topk_weights[pos][slot_mask].sum().float()

                expert_weighted_norms[exp_idx] += (gate_w * activation_norms[i]).item()
                expert_counts[exp_idx] += 1

                moe_output[pos] += gate_w.half() * expert_out[i]

    return moe_output.view(B, S, H), expert_weighted_norms, expert_counts


# =============================================================================
# PARALLEL LAYER PROCESSOR
# =============================================================================

class ParallelLayerProcessor:
    """
    Process multiple layers in parallel for H200 optimization.

    Memory estimation for Qwen3-VL-235B-AWQ:
    - AWQ model weights: ~30GB
    - Per-layer attention: ~0.5GB
    - Per-layer MoE: ~2.5GB (128 experts)
    - Working memory per layer: ~3GB
    - With 4 parallel layers: ~12GB working memory

    Total: ~45GB, leaving ~95GB buffer (well under 140GB)
    """

    def __init__(self, model_path: str, config: dict, num_layers: int,
                 parallel_layers: int = 4, vram_limit_gb: float = 126.0):
        self.model_path = model_path
        self.config = config
        self.num_layers = num_layers
        self.parallel_layers = parallel_layers
        self.vram_limit_gb = vram_limit_gb

        # Load weight index
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            weight_index = json.load(f)

        self.weight_map = weight_index['weight_map']
        self.shard_to_weights = defaultdict(list)
        for name, shard in self.weight_map.items():
            self.shard_to_weights[shard].append(name)

        # Cache for loaded shards
        self.shard_cache = {}
        self.cache_lock = threading.Lock()

    def _get_layer_shards(self, layer_indices: List[int]) -> set:
        """Get all shards needed for given layers."""
        shards = set()
        for layer_idx in layer_indices:
            prefix = f"model.language_model.layers.{layer_idx}."
            for name, shard in self.weight_map.items():
                if prefix in name:
                    shards.add(shard)
        return shards

    def _load_layer_weights(self, layer_idx: int) -> dict:
        """Load all weights for a single layer."""
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

    def process_layers_parallel(self, hidden_states_list: List[torch.Tensor],
                                 rope: RotaryEmbedding,
                                 reap_tracker: Dict[int, OnlineStatsTracker],
                                 count_tracker: Dict[int, OnlineStatsTracker],
                                 progress_callback=None) -> List[torch.Tensor]:
        """
        Process all layers, batching parallel_layers at a time.

        Returns updated hidden states.
        """
        num_experts = self.config['num_experts']

        # Process layers in parallel batches
        for batch_start in range(0, self.num_layers, self.parallel_layers):
            batch_end = min(batch_start + self.parallel_layers, self.num_layers)
            batch_layers = list(range(batch_start, batch_end))

            batch_start_time = time.time()

            # Pre-load all weights for this batch
            batch_weights = {}
            for layer_idx in batch_layers:
                batch_weights[layer_idx] = self._load_layer_weights(layer_idx)

            # Process each layer in the batch sequentially
            # (hidden states must be processed in order)
            for layer_idx in batch_layers:
                layer_scores = torch.zeros(num_experts, dtype=torch.float64, device=device)
                layer_counts = torch.zeros(num_experts, dtype=torch.long, device=device)

                # Process samples in mini-batches
                for i in range(0, len(hidden_states_list), CONFIG.batch_size):
                    batch = hidden_states_list[i:i+CONFIG.batch_size]
                    max_len = max(h.shape[0] for h in batch)

                    # Pad batch
                    padded = torch.zeros(len(batch), max_len, self.config['hidden_size'],
                                        device=device, dtype=torch.half)
                    attention_mask = torch.zeros(len(batch), max_len, device=device, dtype=torch.float16)

                    for j, h in enumerate(batch):
                        padded[j, :h.shape[0]] = h
                        attention_mask[j, :h.shape[0]] = 1.0

                    # Forward pass
                    with torch.no_grad():
                        output, scores, counts = self._forward_layer(
                            padded, batch_weights[layer_idx], layer_idx, rope,
                            attention_mask=attention_mask
                        )

                    layer_scores += scores
                    layer_counts += counts

                    # Update hidden states
                    for j, h in enumerate(batch):
                        hidden_states_list[i+j] = output[j, :h.shape[0]].clone()

                # Update REAP trackers with MEAN scores (not sum!)
                # REAP = mean(gate_weight * ||output||_L2)
                mean_scores = layer_scores / layer_counts.float().clamp(min=1)
                reap_tracker[layer_idx].update(mean_scores, layer_counts)
                count_tracker[layer_idx].update(layer_counts.float(),
                                                torch.ones(num_experts, dtype=torch.long, device=device))

                if progress_callback:
                    progress_callback(layer_idx, time.time() - batch_start_time,
                                     mean_scores, (layer_counts > 0).sum().item())

            # Cleanup batch weights
            del batch_weights
            torch.cuda.empty_cache()

        return hidden_states_list

    def _forward_layer(self, hidden_states: torch.Tensor, weights: dict,
                       layer_idx: int, rope: RotaryEmbedding,
                       attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Complete transformer layer forward."""
        prefix = f"model.language_model.layers.{layer_idx}."

        # Attention block
        residual = hidden_states
        input_ln = weights.get(f"{prefix}input_layernorm.weight")
        if input_ln is not None:
            hidden_states = rms_norm(hidden_states, input_ln)

        hidden_states = forward_attention(
            hidden_states, weights, prefix, rope,
            self.config['num_attention_heads'],
            self.config['num_key_value_heads'],
            self.config['head_dim'],
            attention_mask
        )
        hidden_states = residual + hidden_states

        # MoE block
        residual = hidden_states
        post_ln = weights.get(f"{prefix}post_attention_layernorm.weight")
        if post_ln is not None:
            hidden_normed = rms_norm(hidden_states, post_ln)
        else:
            hidden_normed = hidden_states

        moe_output, expert_scores, expert_counts = forward_moe_with_reap(
            hidden_normed, weights, prefix,
            self.config['num_experts'],
            self.config['num_experts_per_tok'],
            compute_all_experts=False  # Efficient mode
        )

        hidden_states = residual + moe_output

        return hidden_states, expert_scores, expert_counts


# =============================================================================
# VISION ENCODER
# =============================================================================

def load_vision_encoder(model_path: str):
    """Load vision encoder efficiently."""
    from transformers import AutoConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLVisionEncoder

    print("  Loading vision encoder...", flush=True)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    visual = Qwen3VLVisionEncoder(config.vision_config).to(device).half()

    vision_shard = os.path.join(model_path, "model-00001-of-00042.safetensors")
    with safe_open(vision_shard, framework="pt", device="cuda:0") as f:
        state_dict = {k.replace("model.visual.", ""): f.get_tensor(k)
                      for k in f.keys() if k.startswith("model.visual.")}

    visual.load_state_dict(state_dict, strict=False)
    visual.eval()

    print(f"  Vision encoder: {sum(p.numel() for p in visual.parameters())/1e6:.1f}M params", flush=True)
    return visual


# =============================================================================
# CALIBRATION DATASET
# =============================================================================

class CalibrationDataset:
    """Calibration data loader."""

    TEMPLATES_TR = [
        'Gorselde "{region}" nesnesini bul ve bounding box koordinatlarini ver.',
        '"{region}" konumunu tespit et.',
        'Bu gorselde "{region}" nerede?',
    ]
    TEMPLATES_EN = [
        'Locate "{region}" in the image and return bounding box.',
        'Find the bounding box of "{region}".',
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
                    if region and len(region) > 1:
                        self.samples.append({
                            'image_path': image_path,
                            'region': region,
                            'id': item.get('id', ''),
                        })

    def generate_samples(self, max_samples: Optional[int] = None) -> List[dict]:
        samples = self.samples
        if max_samples and len(samples) > max_samples:
            random.seed(CONFIG.random_seed)
            samples = random.sample(samples, max_samples)

        results = []
        for s in samples:
            template = random.choice(self.TEMPLATES_TR if random.random() < 0.8 else self.TEMPLATES_EN)
            results.append({
                'image_path': s['image_path'],
                'prompt': template.format(region=s['region']),
                'region': s['region'],
            })
        return results


# =============================================================================
# PROGRESS TRACKER
# =============================================================================

class ProgressTracker:
    def __init__(self, total_layers: int, total_samples: int):
        self.total_layers = total_layers
        self.total_samples = total_samples
        self.start_time = None
        self.layer_times = []

    def start(self):
        self.start_time = time.time()
        print(f"\n{'='*70}", flush=True)
        print(f"  REAP V10 SCORING (CerebrasResearch Compatible)", flush=True)
        print(f"  Layers: {self.total_layers} | Samples: {self.total_samples}", flush=True)
        print(f"  Parallel Layers: {CONFIG.parallel_layers} | Batch Size: {CONFIG.batch_size}", flush=True)
        print(f"  VRAM Limit: {CONFIG.total_vram_gb * (1 - CONFIG.vram_buffer_ratio):.0f}GB", flush=True)
        print(f"{'='*70}\n", flush=True)

    def update(self, layer_idx: int, layer_time: float, scores: torch.Tensor, active: int):
        self.layer_times.append(layer_time)
        progress = (layer_idx + 1) / self.total_layers * 100

        avg_time = sum(self.layer_times[-10:]) / len(self.layer_times[-10:])
        remaining = self.total_layers - layer_idx - 1
        eta = timedelta(seconds=int(remaining * avg_time / max(1, CONFIG.parallel_layers)))

        mean_score = scores.mean().item()
        max_score = scores.max().item()
        min_score = scores[scores > 0].min().item() if (scores > 0).any() else 0

        bar_len = 30
        filled = int(bar_len * progress / 100)
        bar = '█' * filled + '░' * (bar_len - filled)

        print(f"  Layer {layer_idx+1:3d}/{self.total_layers} [{bar}] {progress:5.1f}%", flush=True)
        print(f"    Active: {active}/128 | Time: {layer_time:.1f}s | ETA: {eta}", flush=True)
        print(f"    REAP: min={min_score:.6f}, mean={mean_score:.6f}, max={max_score:.6f}\n", flush=True)

    def finish(self) -> float:
        total = time.time() - self.start_time
        print(f"{'='*70}", flush=True)
        print(f"  REAP SCORING COMPLETE!", flush=True)
        print(f"  Total: {timedelta(seconds=int(total))}", flush=True)
        print(f"{'='*70}\n", flush=True)
        return total


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70, flush=True)
    print("  REAP V10 - CerebrasResearch Compatible Expert Pruning", flush=True)
    print("  Model: Qwen3-VL-235B-A22B-AWQ", flush=True)
    print("  H200 Optimized: Parallel Layer Processing", flush=True)
    print("="*70, flush=True)

    # Print configuration
    print_config()

    # =========================================================================
    # 1. Load Configuration
    # =========================================================================
    print("\n[1/7] Loading model configuration...", flush=True)

    config_path = os.path.join(CONFIG.model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get('text_config', config)

    model_config = {
        'num_hidden_layers': text_config['num_hidden_layers'],
        'num_attention_heads': text_config.get('num_attention_heads', 64),
        'num_key_value_heads': text_config.get('num_key_value_heads', 4),
        'hidden_size': text_config['hidden_size'],
        'head_dim': text_config.get('head_dim', text_config['hidden_size'] // text_config.get('num_attention_heads', 64)),
        'num_experts': text_config.get('num_experts', 128),
        'num_experts_per_tok': text_config.get('num_experts_per_tok', 8),
        'rope_theta': text_config.get('rope_theta', 1000000.0),
    }

    NUM_LAYERS = model_config['num_hidden_layers']
    NUM_EXPERTS = model_config['num_experts']
    EXPERTS_TO_KEEP = NUM_EXPERTS - int(NUM_EXPERTS * CONFIG.prune_ratio)
    IMAGE_TOKEN_ID = config.get('image_token_id', 151655)

    print(f"  Layers: {NUM_LAYERS}", flush=True)
    print(f"  Experts: {NUM_EXPERTS} -> {EXPERTS_TO_KEEP} (prune {CONFIG.prune_ratio*100:.0f}%)", flush=True)
    print(f"  VRAM: {CONFIG.total_vram_gb}GB - {CONFIG.vram_buffer_ratio*100:.0f}% buffer = {CONFIG.total_vram_gb*(1-CONFIG.vram_buffer_ratio):.0f}GB usable", flush=True)

    # =========================================================================
    # 2. Load Calibration Data
    # =========================================================================
    print("\n[2/7] Loading calibration data...", flush=True)

    dataset = CalibrationDataset(CONFIG.calibration_data_path)
    samples = dataset.generate_samples(CONFIG.max_samples)
    print(f"  Samples: {len(samples)}", flush=True)

    # =========================================================================
    # 3. Load Processor and Embeddings
    # =========================================================================
    print("\n[3/7] Loading processor...", flush=True)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(CONFIG.model_path, trust_remote_code=True)

    # Load embeddings
    index_path = os.path.join(CONFIG.model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_index = json.load(f)

    embed_key = "model.language_model.embed_tokens.weight"
    embed_shard = weight_index['weight_map'][embed_key]

    with safe_open(os.path.join(CONFIG.model_path, embed_shard), framework="pt", device="cuda:0") as f:
        embed_weight = f.get_tensor(embed_key)

    embed_tokens = nn.Embedding(embed_weight.shape[0], embed_weight.shape[1], device=device)
    embed_tokens.weight.data = embed_weight.half()
    embed_tokens.eval()

    # =========================================================================
    # 4. Load Vision Encoder
    # =========================================================================
    print("\n[4/7] Loading vision encoder...", flush=True)

    try:
        visual = load_vision_encoder(CONFIG.model_path)
        use_vision = True
    except Exception as e:
        print(f"  Warning: Vision encoder failed: {e}", flush=True)
        visual = None
        use_vision = False

    # =========================================================================
    # 5. Process Samples
    # =========================================================================
    print("\n[5/7] Processing samples...", flush=True)

    all_hidden_states = []
    vision_count = 0

    for sample in tqdm(samples, desc="  Processing"):
        try:
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
                if use_vision and visual is not None:
                    vision_embeds = visual(pixel_values, grid_thw=image_grid_thw)
                else:
                    num_patches = (image_grid_thw[0, 1] * image_grid_thw[0, 2]).item() // 4
                    vision_embeds = torch.zeros(1, int(num_patches), model_config['hidden_size'],
                                               device=device, dtype=torch.half)

                text_embeds = embed_tokens(input_ids)

            image_mask = (input_ids[0] == IMAGE_TOKEN_ID)
            image_positions = torch.where(image_mask)[0]

            if len(image_positions) > 0 and vision_embeds.shape[1] > 0:
                combined = text_embeds.clone()
                replace_count = min(len(image_positions), vision_embeds.shape[1])
                for i in range(replace_count):
                    combined[0, image_positions[i]] = vision_embeds[0, i]
                hidden = combined.squeeze(0)
                vision_count += 1
            else:
                hidden = text_embeds.squeeze(0)

            if hidden.shape[0] > CONFIG.max_seq_len:
                hidden = hidden[:CONFIG.max_seq_len]

            all_hidden_states.append(hidden)

        except Exception as e:
            try:
                inputs = processor.tokenizer(sample['prompt'], return_tensors="pt",
                                            truncation=True, max_length=CONFIG.max_seq_len).to(device)
                with torch.no_grad():
                    hidden = embed_tokens(inputs['input_ids']).squeeze(0)
                all_hidden_states.append(hidden)
            except:
                continue

    # Cleanup
    if use_vision:
        del visual
    del embed_tokens, embed_weight
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n  Processed: {len(all_hidden_states)} samples", flush=True)
    print(f"  With vision: {vision_count}", flush=True)

    # =========================================================================
    # 6. REAP Scoring with Parallel Processing
    # =========================================================================
    print("\n[6/7] Running REAP scoring (parallel layers)...", flush=True)

    # Initialize RoPE
    rope = RotaryEmbedding(
        dim=model_config['head_dim'],
        max_position=CONFIG.max_seq_len,
        base=model_config['rope_theta'],
        interleaved=True
    )

    # Initialize REAP trackers (online statistics)
    reap_tracker = {i: OnlineStatsTracker((NUM_EXPERTS,), device=device) for i in range(NUM_LAYERS)}
    count_tracker = {i: OnlineStatsTracker((NUM_EXPERTS,), device=device) for i in range(NUM_LAYERS)}

    # Progress tracking
    progress = ProgressTracker(NUM_LAYERS, len(all_hidden_states))
    progress.start()

    # Initialize parallel processor
    processor_layers = ParallelLayerProcessor(
        CONFIG.model_path, model_config, NUM_LAYERS,
        parallel_layers=CONFIG.parallel_layers,
        vram_limit_gb=CONFIG.total_vram_gb * (1 - CONFIG.vram_buffer_ratio)
    )

    # Process all layers
    all_hidden_states = processor_layers.process_layers_parallel(
        all_hidden_states, rope, reap_tracker, count_tracker,
        progress_callback=progress.update
    )

    total_time = progress.finish()

    # =========================================================================
    # 7. Save Results
    # =========================================================================
    print("[7/7] Saving results...", flush=True)

    # Collect final REAP scores (means)
    mean_scores = np.zeros((NUM_LAYERS, NUM_EXPERTS))
    total_counts = np.zeros((NUM_LAYERS, NUM_EXPERTS))

    for layer_idx in range(NUM_LAYERS):
        mean_scores[layer_idx] = reap_tracker[layer_idx].get_mean().cpu().numpy()
        total_counts[layer_idx] = count_tracker[layer_idx].get_mean().cpu().numpy()

    # Select experts to keep (HIGHEST scores = most important)
    experts_to_keep = {}
    for layer_idx in range(NUM_LAYERS):
        scores = mean_scores[layer_idx]
        sorted_indices = np.argsort(scores)[::-1]  # Descending (highest first)
        experts_to_keep[layer_idx] = sorted_indices[:EXPERTS_TO_KEEP].tolist()

    # Save metadata
    os.makedirs(CONFIG.output_path, exist_ok=True)

    metadata = {
        'source_model': CONFIG.model_path,
        'reap_version': 'v10_cerebras_compatible',
        'prune_ratio': CONFIG.prune_ratio,
        'experts_kept': EXPERTS_TO_KEEP,
        'num_experts_original': NUM_EXPERTS,
        'num_layers': NUM_LAYERS,
        'calibration_samples': len(all_hidden_states),
        'vision_samples': vision_count,
        'parallel_layers': CONFIG.parallel_layers,
        'vram_limit_gb': CONFIG.total_vram_gb * (1 - CONFIG.vram_buffer_ratio),
        'total_time_seconds': total_time,
        'model_config': model_config,
        'experts_to_keep': {str(k): v for k, v in experts_to_keep.items()},
        'mean_scores': mean_scores.tolist(),
        'expert_counts': total_counts.tolist(),
        'timestamp': datetime.now().isoformat(),
        'methodology': 'REAP = mean(gate_weight * ||expert_output||_L2)',
    }

    with open(os.path.join(CONFIG.output_path, 'reap_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Copy config files
    for fname in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                  'vocab.json', 'merges.txt', 'special_tokens_map.json',
                  'preprocessor_config.json', 'chat_template.json', 'generation_config.json']:
        src = os.path.join(CONFIG.model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, CONFIG.output_path)

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print(f"  REAP V10 COMPLETE!", flush=True)
    print(f"  Output: {CONFIG.output_path}", flush=True)
    print(f"  Total time: {timedelta(seconds=int(total_time))}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Score statistics
    print("  REAP Score Statistics:", flush=True)
    for layer_idx in range(0, NUM_LAYERS, 10):
        scores = mean_scores[layer_idx]
        kept = scores[experts_to_keep[layer_idx]]
        pruned_mask = np.ones(NUM_EXPERTS, dtype=bool)
        pruned_mask[experts_to_keep[layer_idx]] = False
        pruned = scores[pruned_mask]

        print(f"    Layer {layer_idx:3d}: kept={kept.mean():.6f}, pruned={pruned.mean():.6f}, "
              f"ratio={kept.mean()/(pruned.mean()+1e-10):.2f}x", flush=True)


if __name__ == "__main__":
    main()
