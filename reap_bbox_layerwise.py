#!/usr/bin/env python3
"""
REAP Layer-by-Layer for Bounding Box Detection Task
===================================================

Bu script, Qwen3-VL-235B-A22B modelini bounding box detection görevi için
optimize etmek üzere REAP (Router-based Expert Allocation Pruning) uygular.

Calibration verisi olarak region description'ları (görsel üzerinde tespit
edilecek nesneler/metinler) kullanılır. Bu sayede model, bbox detection
için en önemli expert'ları korur.

Entegre çalışma:
1. create_reap_calibration_dataset.py çalıştırılır -> calibration_data.json oluşur
2. Bu script calibration_data.json'ı okur
3. Her region için bbox detection promptu oluşturur
4. Bu promptlar üzerinden REAP skorları hesaplanır
5. En önemli expert'lar korunarak model prune edilir
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import time
import random
import gc
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from collections import defaultdict
from tqdm import tqdm
import shutil
import re
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model paths
MODEL_PATH = "/workspace/Qwen3-VL-235B-A22B-Thinking-AWQ"
OUTPUT_PATH = "/workspace/qwen3vl-235b-bbox-reap"

# Calibration data (create_reap_calibration_dataset.py tarafından oluşturulur)
CALIBRATION_DATA_PATH = "/mnt/data/maarifxtrainingmodelqwen/Boundingboxmodel/reap_calibration_data/calibration_data.json"

# Model architecture
NUM_LAYERS = 94
NUM_EXPERTS = 128
TOP_K = 8
PRUNE_RATIO = 0.40
EXPERTS_TO_KEEP = NUM_EXPERTS - int(NUM_EXPERTS * PRUNE_RATIO)  # 77

# Calibration settings
MAX_SAMPLES = 3000  # Max calibration samples
MAX_SEQ_LEN = 256
RANDOM_SEED = 42
GROUP_SIZE = 128  # AWQ

device = torch.device("cuda:0")


# =============================================================================
# BBOX CALIBRATION PROMPT GENERATOR
# =============================================================================

class BBoxCalibrationGenerator:
    """
    Calibration data'dan REAP için prompt üretir.
    Her region için farklı bbox detection promptları oluşturur.
    """

    # Prompt templates - model bu tarz promptları görecek
    TEMPLATES_TR = [
        'Görselde "{region}" nesnesini bul ve bounding box koordinatlarını ver. Cevap: {{"bbox_2d": [',
        '"{region}" konumunu tespit et. Koordinatlar: {{"bbox_2d": [',
        'Bu görselde "{region}" nerede? {{"bbox_2d": [',
        '"{region}" için bounding box bul: {{"bbox_2d": [',
        'Tespit: "{region}" -> {{"bbox_2d": [',
    ]

    TEMPLATES_EN = [
        'Locate "{region}" in the image and return bounding box. Answer: {{"bbox_2d": [',
        'Find the bounding box of "{region}": {{"bbox_2d": [',
        'Detect "{region}" coordinates: {{"bbox_2d": [',
        'BBox for "{region}": {{"bbox_2d": [',
    ]

    def __init__(self, calibration_data_path: str):
        self.data_path = calibration_data_path
        self.samples = []
        self.load_data()

    def load_data(self):
        """Calibration data'yı yükle."""
        print(f"\n  Loading calibration data: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Her image'deki her region için ayrı sample oluştur
        for item in data:
            image_id = item.get('id', '')
            regions = item.get('regions', [])

            for region in regions:
                if region and len(region) > 1:
                    self.samples.append({
                        'image_id': image_id,
                        'region': region,
                    })

        print(f"  Total calibration samples: {len(self.samples)}")

    def generate_prompts(self, max_samples: int = None) -> list:
        """REAP için kalibrasyon promptları üret."""
        samples = self.samples
        if max_samples and len(samples) > max_samples:
            random.seed(RANDOM_SEED)
            samples = random.sample(samples, max_samples)

        prompts = []
        for sample in samples:
            region = sample['region']

            # %80 Türkçe, %20 İngilizce
            if random.random() < 0.8:
                template = random.choice(self.TEMPLATES_TR)
            else:
                template = random.choice(self.TEMPLATES_EN)

            prompt = template.format(region=region)
            prompts.append(prompt)

        return prompts

    def get_stats(self) -> dict:
        """İstatistikler."""
        unique_images = len(set(s['image_id'] for s in self.samples))
        return {
            'total_samples': len(self.samples),
            'unique_images': unique_images,
            'avg_regions_per_image': len(self.samples) / unique_images if unique_images > 0 else 0,
        }


# =============================================================================
# AWQ DEQUANTIZATION
# =============================================================================

def dequantize_awq_fast(qweight, qzeros, scales, group_size=128):
    """AWQ 4-bit -> fp16 dequantization."""
    in_features = qweight.shape[0]
    out_features = scales.shape[1]
    num_groups = scales.shape[0]

    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    qw = qweight.unsqueeze(-1)
    weight_int = ((qw >> shifts.view(1, 1, 8)) & 0xF).reshape(in_features, out_features)

    qz = qzeros.unsqueeze(-1)
    zeros_int = ((qz >> shifts.view(1, 1, 8)) & 0xF).reshape(num_groups, out_features)

    zeros_expanded = zeros_int.repeat_interleave(group_size, dim=0)[:in_features]
    scales_expanded = scales.repeat_interleave(group_size, dim=0)[:in_features]

    weight_fp = (weight_int.float() - zeros_expanded.float()) * scales_expanded.float()
    return weight_fp.half()


# =============================================================================
# MAIN REAP
# =============================================================================

def main():
    print("=" * 70)
    print("REAP Layer-by-Layer - Bounding Box Detection Task")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Calibration: {CALIBRATION_DATA_PATH}")
    print(f"  Prune: {PRUNE_RATIO*100:.0f}% ({NUM_EXPERTS} -> {EXPERTS_TO_KEEP} experts)")

    # =========================================================================
    # Step 1: Load calibration data
    # =========================================================================
    print("\n[1/6] Loading BBox calibration data...")

    calibrator = BBoxCalibrationGenerator(CALIBRATION_DATA_PATH)
    stats = calibrator.get_stats()
    print(f"  Images: {stats['unique_images']}")
    print(f"  Regions: {stats['total_samples']}")
    print(f"  Avg regions/image: {stats['avg_regions_per_image']:.1f}")

    # Generate prompts
    prompts = calibrator.generate_prompts(max_samples=MAX_SAMPLES)
    print(f"  Calibration prompts: {len(prompts)}")

    # Show samples
    print("\n  Sample prompts:")
    for i, p in enumerate(prompts[:3]):
        print(f"    {i+1}. {p[:70]}...")

    # =========================================================================
    # Step 2: Load weight index
    # =========================================================================
    print("\n[2/6] Loading weight index...")
    with open(os.path.join(MODEL_PATH, "model.safetensors.index.json")) as f:
        weight_index = json.load(f)
    weight_map = weight_index['weight_map']

    shard_to_weights = defaultdict(list)
    for name, shard in weight_map.items():
        shard_to_weights[shard].append(name)

    print(f"  Weights: {len(weight_map)}")
    print(f"  Shards: {len(shard_to_weights)}")

    # =========================================================================
    # Step 3: Load embedding & tokenizer
    # =========================================================================
    print("\n[3/6] Loading embedding and tokenizer...")

    embed_name = 'model.language_model.embed_tokens.weight'
    embed_shard = weight_map[embed_name]
    with safe_open(os.path.join(MODEL_PATH, embed_shard), framework="pt", device="cuda:0") as f:
        embed_weight = f.get_tensor(embed_name)

    HIDDEN_SIZE = embed_weight.shape[1]
    print(f"  Embedding: {embed_weight.shape}")
    print(f"  Hidden: {HIDDEN_SIZE}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Tokenize prompts
    all_tokens = []
    for prompt in tqdm(prompts, desc="  Tokenizing"):
        try:
            tokens = tokenizer.encode(prompt, max_length=MAX_SEQ_LEN, truncation=True)
            if len(tokens) > 0:
                all_tokens.append(torch.tensor(tokens, device=device))
        except:
            continue

    total_tokens = sum(len(t) for t in all_tokens)
    print(f"  Valid samples: {len(all_tokens)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # =========================================================================
    # Step 4: Layer-by-layer REAP
    # =========================================================================
    print("\n[4/6] Running REAP scoring...")
    print("=" * 70)

    reap_scores = torch.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=torch.float64, device=device)
    expert_counts = torch.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=torch.int64, device=device)

    start_time = time.time()

    # Embed all tokens
    print("  Computing embeddings...")
    hidden_states_list = []
    with torch.no_grad():
        for tokens in tqdm(all_tokens, desc="  Embedding"):
            h = F.embedding(tokens, embed_weight)
            hidden_states_list.append(h)

    print(f"  GPU after embedding: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Process each layer
    for layer_idx in range(NUM_LAYERS):
        layer_start = time.time()

        layer_prefix = f"model.language_model.layers.{layer_idx}."
        layer_weights = {}
        needed_shards = set()

        for name, shard in weight_map.items():
            if name.startswith(layer_prefix):
                needed_shards.add(shard)

        for shard in needed_shards:
            shard_path = os.path.join(MODEL_PATH, shard)
            with safe_open(shard_path, framework="pt", device="cuda:0") as f:
                for name in shard_to_weights[shard]:
                    if name.startswith(layer_prefix):
                        layer_weights[name] = f.get_tensor(name)

        gate_name = f"{layer_prefix}mlp.gate.weight"
        norm_name = f"{layer_prefix}input_layernorm.weight"

        gate_weight = layer_weights.get(gate_name)
        norm_weight = layer_weights.get(norm_name)

        if gate_weight is None:
            print(f"  Layer {layer_idx}: No gate, skipping")
            continue

        with torch.no_grad(), torch.amp.autocast('cuda'):
            for sample_idx, hidden in enumerate(hidden_states_list):
                seq_len = hidden.shape[0]

                # RMSNorm
                if norm_weight is not None:
                    variance = hidden.pow(2).mean(-1, keepdim=True)
                    hidden_normed = hidden * torch.rsqrt(variance + 1e-6) * norm_weight
                else:
                    hidden_normed = hidden

                # Router
                router_logits = F.linear(hidden_normed, gate_weight)
                router_probs = F.softmax(router_logits, dim=-1)
                topk_weights, topk_indices = torch.topk(router_probs, TOP_K, dim=-1)
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

                unique_experts = topk_indices.unique()
                expert_outputs = {}

                for expert_idx in unique_experts.tolist():
                    gate_proj_qw = layer_weights.get(f"{layer_prefix}mlp.experts.{expert_idx}.gate_proj.qweight")
                    gate_proj_qz = layer_weights.get(f"{layer_prefix}mlp.experts.{expert_idx}.gate_proj.qzeros")
                    gate_proj_sc = layer_weights.get(f"{layer_prefix}mlp.experts.{expert_idx}.gate_proj.scales")

                    up_proj_qw = layer_weights.get(f"{layer_prefix}mlp.experts.{expert_idx}.up_proj.qweight")
                    up_proj_qz = layer_weights.get(f"{layer_prefix}mlp.experts.{expert_idx}.up_proj.qzeros")
                    up_proj_sc = layer_weights.get(f"{layer_prefix}mlp.experts.{expert_idx}.up_proj.scales")

                    if gate_proj_qw is None or up_proj_qw is None:
                        expert_outputs[expert_idx] = torch.norm(hidden_normed, p=2, dim=-1)
                        continue

                    gate_proj = dequantize_awq_fast(gate_proj_qw, gate_proj_qz, gate_proj_sc, GROUP_SIZE)
                    up_proj = dequantize_awq_fast(up_proj_qw, up_proj_qz, up_proj_sc, GROUP_SIZE)

                    gate_out = F.linear(hidden_normed, gate_proj.T)
                    up_out = F.linear(hidden_normed, up_proj.T)
                    expert_out = F.silu(gate_out) * up_out

                    expert_outputs[expert_idx] = torch.norm(expert_out.float(), p=2, dim=-1)
                    del gate_proj, up_proj, gate_out, up_out, expert_out

                # Accumulate REAP scores
                for pos in range(seq_len):
                    for k in range(TOP_K):
                        expert_idx = topk_indices[pos, k].item()
                        gate_w = topk_weights[pos, k].item()

                        if expert_idx in expert_outputs:
                            act_norm = expert_outputs[expert_idx][pos].item()
                        else:
                            act_norm = torch.norm(hidden_normed[pos].float(), p=2).item()

                        score = gate_w * act_norm
                        reap_scores[layer_idx, expert_idx] += score
                        expert_counts[layer_idx, expert_idx] += 1

                hidden_states_list[sample_idx] = hidden_normed.clone()

        del layer_weights
        torch.cuda.empty_cache()
        gc.collect()

        layer_time = time.time() - layer_start
        elapsed = time.time() - start_time
        eta = (NUM_LAYERS - layer_idx - 1) * (elapsed / (layer_idx + 1))
        gpu_mem = torch.cuda.memory_allocated() / 1024**3

        if (layer_idx + 1) % 5 == 0 or layer_idx == 0:
            print(f"  Layer {layer_idx+1}/{NUM_LAYERS} | {layer_time:.1f}s | GPU: {gpu_mem:.1f}GB | ETA: {eta/60:.1f}min")

    elapsed = time.time() - start_time
    print(f"\n  REAP scoring complete: {elapsed/60:.1f} minutes")

    # =========================================================================
    # Step 5: Compute pruning decisions
    # =========================================================================
    print("\n[5/6] Computing expert importance...")

    mean_scores = (reap_scores / expert_counts.float().clamp(min=1)).cpu().numpy()

    experts_to_keep = {}
    experts_to_prune = {}
    for layer in range(NUM_LAYERS):
        scores = mean_scores[layer]
        sorted_idx = np.argsort(scores)[::-1]
        experts_to_keep[layer] = sorted(sorted_idx[:EXPERTS_TO_KEEP].tolist())
        experts_to_prune[layer] = sorted_idx[EXPERTS_TO_KEEP:].tolist()

    print(f"  Keeping: {EXPERTS_TO_KEEP} experts/layer")
    print(f"  Pruning: {NUM_EXPERTS - EXPERTS_TO_KEEP} experts/layer")

    for l in [0, 46, 93]:
        scores = mean_scores[l]
        print(f"  Layer {l}: score [{scores.min():.2f}, {scores.max():.2f}]")

    del hidden_states_list, embed_weight, reap_scores, expert_counts
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # Step 6: Create pruned model
    # =========================================================================
    print("\n[6/6] Creating pruned model...")
    print("=" * 70)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    expert_mapping = {l: {old: new for new, old in enumerate(sorted(kept))}
                      for l, kept in experts_to_keep.items()}

    expert_pattern = re.compile(
        r'model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(down_proj|gate_proj|up_proj)\.(qweight|qzeros|scales)'
    )
    gate_pattern = re.compile(r'model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight')

    new_weight_map = {}
    current_shard = {}
    current_size = 0
    shard_idx = 1
    total_original = 0
    total_pruned = 0
    MAX_SHARD = 5 * 1024**3

    for shard_file in tqdm(sorted(shard_to_weights.keys()), desc="  Pruning shards"):
        with safe_open(os.path.join(MODEL_PATH, shard_file), framework="pt", device="cpu") as f:
            for name in shard_to_weights[shard_file]:
                tensor = f.get_tensor(name)
                size = tensor.numel() * tensor.element_size()
                total_original += size

                em = expert_pattern.match(name)
                gm = gate_pattern.match(name)

                if em:
                    layer, expert = int(em.group(1)), int(em.group(2))
                    if expert not in experts_to_keep[layer]:
                        continue
                    new_idx = expert_mapping[layer][expert]
                    new_name = f"model.language_model.layers.{layer}.mlp.experts.{new_idx}.{em.group(3)}.{em.group(4)}"
                    current_shard[new_name] = tensor
                    total_pruned += size
                elif gm:
                    layer = int(gm.group(1))
                    pruned = tensor[experts_to_keep[layer], :].clone()
                    current_shard[name] = pruned
                    total_pruned += pruned.numel() * pruned.element_size()
                else:
                    current_shard[name] = tensor
                    total_pruned += size

                current_size += size
                if current_size >= MAX_SHARD:
                    sn = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
                    save_file(current_shard, os.path.join(OUTPUT_PATH, sn))
                    for n in current_shard:
                        new_weight_map[n] = sn
                    current_shard, current_size, shard_idx = {}, 0, shard_idx + 1

    if current_shard:
        sn = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        save_file(current_shard, os.path.join(OUTPUT_PATH, sn))
        for n in current_shard:
            new_weight_map[n] = sn

    # Rename shards
    total_shards = shard_idx
    for i in range(1, total_shards + 1):
        old = os.path.join(OUTPUT_PATH, f"model-{i:05d}-of-XXXXX.safetensors")
        new = os.path.join(OUTPUT_PATH, f"model-{i:05d}-of-{total_shards:05d}.safetensors")
        if os.path.exists(old):
            os.rename(old, new)
            for n, s in list(new_weight_map.items()):
                if s == f"model-{i:05d}-of-XXXXX.safetensors":
                    new_weight_map[n] = f"model-{i:05d}-of-{total_shards:05d}.safetensors"

    # Save index
    with open(os.path.join(OUTPUT_PATH, "model.safetensors.index.json"), 'w') as f:
        json.dump({"metadata": {"total_size": total_pruned}, "weight_map": new_weight_map}, f, indent=2)

    # Update config
    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        config = json.load(f)
    config['text_config']['num_experts'] = EXPERTS_TO_KEEP
    with open(os.path.join(OUTPUT_PATH, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Copy files
    for fn in ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
               "generation_config.json", "chat_template.json", "merges.txt", "vocab.json"]:
        src = os.path.join(MODEL_PATH, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUTPUT_PATH, fn))

    # Save report
    report = {
        "method": "REAP Layer-by-Layer",
        "task": "Bounding Box Detection",
        "calibration_data": CALIBRATION_DATA_PATH,
        "calibration_stats": stats,
        "formula": "score = gate_weight × ||expert_activation||_2",
        "experts": f"{NUM_EXPERTS} -> {EXPERTS_TO_KEEP}",
        "prune_ratio": f"{PRUNE_RATIO*100:.0f}%",
        "samples": len(all_tokens),
        "tokens": total_tokens,
        "time_minutes": elapsed / 60,
        "original_gb": total_original / 1024**3,
        "pruned_gb": total_pruned / 1024**3,
        "reduction": f"{(1 - total_pruned / total_original) * 100:.1f}%",
    }

    with open(os.path.join(OUTPUT_PATH, "reap_bbox_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 70)
    print("REAP BBOX COMPLETE!")
    print("=" * 70)
    print(f"""
  Output: {OUTPUT_PATH}
  Size: {total_original/1024**3:.1f} GB -> {total_pruned/1024**3:.1f} GB ({(1-total_pruned/total_original)*100:.1f}% reduction)
  Experts: {NUM_EXPERTS} -> {EXPERTS_TO_KEEP}
  Calibration: {len(all_tokens)} samples, {total_tokens:,} tokens
  Time: {elapsed/60:.1f} minutes
""")


if __name__ == "__main__":
    main()
