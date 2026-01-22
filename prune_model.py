#!/usr/bin/env python3
"""
REAP Model Pruning Script (V10 Compatible)
==========================================

Prunes experts from Qwen3-VL MoE model based on REAP metadata.
Compatible with CerebrasResearch/reap methodology.

Usage:
    python prune_model.py [--metadata PATH] [--output PATH]
"""

import os
import sys
import json
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from collections import defaultdict
import shutil
import re
from datetime import datetime


def parse_args():
    """
    Parse arguments with Docker environment variable support.

    Environment variables:
        REAP_MODEL_PATH, REAP_OUTPUT_PATH, REAP_PRUNED_OUTPUT_PATH
    """
    parser = argparse.ArgumentParser(description="REAP Model Pruning (V10)")

    # Default paths from environment or fallback
    default_model = os.environ.get(
        "REAP_MODEL_PATH",
        "/mnt/vault/boundingboxtest/Qwen3-VL-235B-A22B-Thinking-AWQ"
    )
    default_metadata = os.environ.get(
        "REAP_OUTPUT_PATH",
        "/mnt/vault/boundingboxtest/qwen3vl-235b-bbox-reap-v10"
    ) + "/reap_metadata.json"
    default_output = os.environ.get(
        "REAP_PRUNED_OUTPUT_PATH",
        "/mnt/vault/boundingboxtest/Qwen3-VL-235B-A22B-REAP-Pruned-V10"
    )

    parser.add_argument("--model", type=str, default=default_model,
                       help="Path to source model")
    parser.add_argument("--metadata", type=str, default=default_metadata,
                       help="Path to REAP metadata JSON")
    parser.add_argument("--output", type=str, default=default_output,
                       help="Output path for pruned model")
    parser.add_argument("--verify", action="store_true",
                       help="Verify output after pruning")
    return parser.parse_args()


def main():
    args = parse_args()

    MODEL_PATH = args.model
    METADATA_PATH = args.metadata
    OUTPUT_PATH = args.output

    print("\n" + "="*70)
    print("  REAP Model Pruning (V10 - CerebrasResearch Compatible)")
    print("="*70)

    # =========================================================================
    # 1. Load REAP Metadata
    # =========================================================================
    print("\n[1/6] Loading REAP metadata...")

    if not os.path.exists(METADATA_PATH):
        print(f"  ERROR: Metadata not found: {METADATA_PATH}")
        sys.exit(1)

    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    experts_to_keep = metadata['experts_to_keep']
    num_layers = metadata['num_layers']
    experts_kept = metadata['experts_kept']
    num_experts_original = metadata['num_experts_original']
    prune_ratio = metadata['prune_ratio']
    reap_version = metadata.get('reap_version', 'unknown')
    methodology = metadata.get('methodology', 'REAP scoring')

    print(f"  REAP Version: {reap_version}")
    print(f"  Methodology: {methodology}")
    print(f"  Layers: {num_layers}")
    print(f"  Experts: {num_experts_original} -> {experts_kept} ({prune_ratio*100:.0f}% pruned)")

    # =========================================================================
    # 2. Validate Source Model
    # =========================================================================
    print("\n[2/6] Validating source model...")

    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)

    # Load source config
    source_config_path = os.path.join(MODEL_PATH, 'config.json')
    with open(source_config_path, 'r') as f:
        source_config = json.load(f)

    text_config = source_config.get('text_config', source_config)
    source_experts = text_config.get('num_experts', 128)
    source_layers = text_config.get('num_hidden_layers', 80)

    print(f"  Source model: {os.path.basename(MODEL_PATH)}")
    print(f"  Source experts: {source_experts}")
    print(f"  Source layers: {source_layers}")

    if source_experts != num_experts_original:
        print(f"  WARNING: Expert count mismatch! Source: {source_experts}, Metadata: {num_experts_original}")

    if source_layers != num_layers:
        print(f"  WARNING: Layer count mismatch! Source: {source_layers}, Metadata: {num_layers}")

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # =========================================================================
    # 3. Build Expert Mapping
    # =========================================================================
    print("\n[3/6] Building expert index mapping...")

    # Map old expert indices to new consecutive indices
    expert_mapping = {}  # {layer_idx: {old_idx: new_idx}}

    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)

        if layer_key not in experts_to_keep:
            print(f"  WARNING: No expert data for layer {layer_idx}")
            expert_mapping[layer_idx] = {i: i for i in range(experts_kept)}
            continue

        # Sort kept experts for consistent new indices
        kept = sorted(experts_to_keep[layer_key])

        if len(kept) != experts_kept:
            print(f"  WARNING: Layer {layer_idx} has {len(kept)} experts, expected {experts_kept}")

        expert_mapping[layer_idx] = {old_idx: new_idx for new_idx, old_idx in enumerate(kept)}

    print(f"  Created mapping for {len(expert_mapping)} layers")

    # Show sample mapping
    if 0 in expert_mapping and len(expert_mapping[0]) > 0:
        sample_old = list(expert_mapping[0].keys())[:5]
        sample_new = [expert_mapping[0][k] for k in sample_old]
        print(f"  Sample (layer 0): {dict(zip(sample_old[:3], sample_new[:3]))}")

    # =========================================================================
    # 4. Scan Model Shards
    # =========================================================================
    print("\n[4/6] Scanning model shards...")

    shard_files = sorted([f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')])
    print(f"  Found {len(shard_files)} shards")

    # Build weight index
    weight_to_shard = {}
    shard_weights = defaultdict(list)

    for shard_file in tqdm(shard_files, desc="  Indexing"):
        shard_path = os.path.join(MODEL_PATH, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_to_shard[key] = shard_file
                shard_weights[shard_file].append(key)

    print(f"  Total weights: {len(weight_to_shard)}")

    # Patterns
    expert_pattern = re.compile(
        r'model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)'
    )
    gate_pattern = re.compile(
        r'model\.language_model\.layers\.(\d+)\.mlp\.gate\.(weight|qweight|scales|qzeros)'
    )

    # Count weights
    expert_weights = [k for k in weight_to_shard if expert_pattern.match(k)]
    gate_weights = [k for k in weight_to_shard if gate_pattern.match(k)]
    other_weights = len(weight_to_shard) - len(expert_weights) - len(gate_weights)

    print(f"  Expert weights: {len(expert_weights)}")
    print(f"  Gate weights: {len(gate_weights)}")
    print(f"  Other weights: {other_weights}")

    # =========================================================================
    # 5. Prune Model
    # =========================================================================
    print("\n[5/6] Pruning model...")

    weight_map = {}
    total_pruned = 0
    total_kept = 0
    total_unchanged = 0

    for shard_idx, shard_file in enumerate(tqdm(shard_files, desc="  Processing")):
        shard_path = os.path.join(MODEL_PATH, shard_file)
        output_shard_path = os.path.join(OUTPUT_PATH, shard_file)

        new_tensors = {}

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # Check expert weight
                expert_match = expert_pattern.match(key)
                if expert_match:
                    layer_idx = int(expert_match.group(1))
                    expert_idx = int(expert_match.group(2))
                    weight_name = expert_match.group(3)

                    if expert_idx in expert_mapping.get(layer_idx, {}):
                        new_expert_idx = expert_mapping[layer_idx][expert_idx]
                        new_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{new_expert_idx}.{weight_name}"
                        new_tensors[new_key] = tensor
                        weight_map[new_key] = shard_file
                        total_kept += 1
                    else:
                        total_pruned += 1
                    continue

                # Check gate weight
                gate_match = gate_pattern.match(key)
                if gate_match:
                    layer_idx = int(gate_match.group(1))
                    weight_type = gate_match.group(2)

                    # Get kept expert indices for this layer
                    kept_indices = sorted(experts_to_keep.get(str(layer_idx), []))

                    if len(kept_indices) == 0:
                        new_tensors[key] = tensor
                        weight_map[key] = shard_file
                        total_unchanged += 1
                        continue

                    # For AWQ quantized weights
                    if weight_type == 'qweight':
                        # qweight shape: [hidden_size//8, num_experts] (packed)
                        # Actually for gate: [hidden_size, num_experts//8]
                        # This depends on AWQ packing format
                        # Usually gate weight is [num_experts, hidden_size]
                        # Safest approach: check shape and handle accordingly

                        if tensor.dim() == 2:
                            if tensor.shape[0] == num_experts_original:
                                # [num_experts, hidden//8] - select rows
                                new_tensor = tensor[kept_indices, :]
                            elif tensor.shape[1] == num_experts_original:
                                # [hidden, num_experts] - select columns
                                new_tensor = tensor[:, kept_indices]
                            else:
                                # Keep as-is if shape doesn't match
                                new_tensor = tensor
                        else:
                            new_tensor = tensor

                    elif weight_type == 'weight':
                        # Regular weight: [num_experts, hidden_size]
                        if tensor.shape[0] == num_experts_original:
                            new_tensor = tensor[kept_indices, :]
                        else:
                            new_tensor = tensor

                    elif weight_type in ['scales', 'qzeros']:
                        # AWQ scales/zeros: [groups, num_experts] or [num_experts//8, hidden//group]
                        if tensor.dim() == 2:
                            if tensor.shape[1] == num_experts_original:
                                new_tensor = tensor[:, kept_indices]
                            elif tensor.shape[0] == num_experts_original:
                                new_tensor = tensor[kept_indices, :]
                            else:
                                new_tensor = tensor
                        else:
                            new_tensor = tensor
                    else:
                        new_tensor = tensor

                    new_tensors[key] = new_tensor
                    weight_map[key] = shard_file
                    total_unchanged += 1
                    continue

                # Non-expert, non-gate weight
                new_tensors[key] = tensor
                weight_map[key] = shard_file
                total_unchanged += 1

        # Save pruned shard
        if new_tensors:
            save_file(new_tensors, output_shard_path)

    print(f"\n  Expert weights pruned: {total_pruned}")
    print(f"  Expert weights kept: {total_kept}")
    print(f"  Other weights: {total_unchanged}")

    # =========================================================================
    # 6. Create Config and Index
    # =========================================================================
    print("\n[6/6] Creating config and index files...")

    # Modify config
    config = source_config.copy()

    if 'text_config' in config:
        config['text_config']['num_experts'] = experts_kept

    # Add REAP pruning info
    config['reap_pruning'] = {
        'pruned': True,
        'reap_version': reap_version,
        'methodology': methodology,
        'original_experts': num_experts_original,
        'kept_experts': experts_kept,
        'prune_ratio': prune_ratio,
        'pruned_at': datetime.now().isoformat(),
        'source_metadata': METADATA_PATH,
    }

    with open(os.path.join(OUTPUT_PATH, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Create model index
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_PATH, f))
        for f in os.listdir(OUTPUT_PATH) if f.endswith('.safetensors')
    )

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }

    with open(os.path.join(OUTPUT_PATH, 'model.safetensors.index.json'), 'w') as f:
        json.dump(index, f, indent=2)

    # Copy support files
    support_files = [
        'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
        'merges.txt', 'preprocessor_config.json', 'chat_template.json',
        'generation_config.json', 'special_tokens_map.json'
    ]

    for fname in support_files:
        src = os.path.join(MODEL_PATH, fname)
        dst = os.path.join(OUTPUT_PATH, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Copy REAP metadata
    shutil.copy2(METADATA_PATH, os.path.join(OUTPUT_PATH, 'reap_metadata.json'))

    # Calculate size
    original_size = sum(
        os.path.getsize(os.path.join(MODEL_PATH, f))
        for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')
    )
    pruned_size = total_size

    # Summary
    print("\n" + "="*70)
    print("  PRUNING COMPLETE!")
    print("="*70)
    print(f"  Original: {original_size / 1e9:.2f} GB")
    print(f"  Pruned:   {pruned_size / 1e9:.2f} GB")
    print(f"  Saved:    {(original_size - pruned_size) / 1e9:.2f} GB ({(1 - pruned_size/original_size)*100:.1f}%)")
    print(f"  Output:   {OUTPUT_PATH}")
    print("="*70 + "\n")

    # Verification
    if args.verify:
        print("Verifying output...")
        verify_pruned_model(OUTPUT_PATH, experts_kept, num_layers)


def verify_pruned_model(output_path: str, expected_experts: int, expected_layers: int):
    """Verify the pruned model has correct structure."""
    print("\n[Verification]")

    # Check config
    with open(os.path.join(output_path, 'config.json'), 'r') as f:
        config = json.load(f)

    text_config = config.get('text_config', config)
    actual_experts = text_config.get('num_experts')

    if actual_experts == expected_experts:
        print(f"  Config experts: OK ({actual_experts})")
    else:
        print(f"  Config experts: MISMATCH (got {actual_experts}, expected {expected_experts})")

    # Check index
    with open(os.path.join(output_path, 'model.safetensors.index.json'), 'r') as f:
        index = json.load(f)

    weight_map = index['weight_map']

    # Count expert weights per layer
    expert_pattern = re.compile(r'model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.')

    layer_expert_counts = defaultdict(set)
    for key in weight_map:
        match = expert_pattern.match(key)
        if match:
            layer_idx = int(match.group(1))
            expert_idx = int(match.group(2))
            layer_expert_counts[layer_idx].add(expert_idx)

    # Check each layer
    issues = []
    for layer_idx in range(expected_layers):
        experts = layer_expert_counts.get(layer_idx, set())
        if len(experts) != expected_experts:
            issues.append(f"Layer {layer_idx}: {len(experts)} experts (expected {expected_experts})")

        # Check indices are consecutive 0 to N-1
        if experts and (min(experts) != 0 or max(experts) != expected_experts - 1):
            issues.append(f"Layer {layer_idx}: non-consecutive indices {min(experts)}-{max(experts)}")

    if issues:
        print("  Issues found:")
        for issue in issues[:10]:
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    else:
        print(f"  All {expected_layers} layers have {expected_experts} consecutive experts: OK")


if __name__ == "__main__":
    main()
