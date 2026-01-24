#!/usr/bin/env python3
"""
REAP V11 Model Pruning Script
=============================

Prunes experts from Qwen3-VL MoE model based on REAP metadata.
Fixed version with better AWQ handling.

Usage:
    python prune_model.py [--metadata PATH] [--output PATH] [--verify]
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
    parser = argparse.ArgumentParser(description="REAP V11 Model Pruning")
    
    default_model = os.environ.get(
        "REAP_MODEL_PATH",
        "/mnt/vault/boundingboxtest/Qwen3-VL-235B-A22B-Thinking-AWQ"
    )
    default_metadata = os.environ.get(
        "REAP_OUTPUT_PATH",
        "/mnt/vault/boundingboxtest/qwen3vl-235b-reap-v11"
    ) + "/reap_metadata.json"
    default_output = os.environ.get(
        "REAP_PRUNED_OUTPUT_PATH",
        "/mnt/vault/boundingboxtest/Qwen3-VL-235B-REAP-Pruned-V11"
    )
    
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--metadata", type=str, default=default_metadata)
    parser.add_argument("--output", type=str, default=default_output)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    MODEL_PATH = args.model
    METADATA_PATH = args.metadata
    OUTPUT_PATH = args.output
    
    print("\n" + "="*70)
    print("  REAP V11 Model Pruning")
    print("="*70)
    
    # 1. Load metadata
    print("\n[1/6] Loading REAP metadata...")
    
    if not os.path.exists(METADATA_PATH):
        print(f"  ERROR: {METADATA_PATH} not found")
        sys.exit(1)
    
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    experts_to_keep = metadata['experts_to_keep']
    num_layers = metadata['num_layers']
    experts_kept = metadata['experts_kept']
    num_experts_original = metadata['num_experts_original']
    prune_ratio = metadata['prune_ratio']
    
    print(f"  Version: {metadata.get('reap_version', 'unknown')}")
    print(f"  Experts: {num_experts_original} -> {experts_kept}")
    
    # 2. Validate model
    print("\n[2/6] Validating source model...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: {MODEL_PATH} not found")
        sys.exit(1)
    
    with open(os.path.join(MODEL_PATH, 'config.json'), 'r') as f:
        source_config = json.load(f)
    
    text_config = source_config.get('text_config', source_config)
    source_experts = text_config.get('num_experts', 128)
    
    print(f"  Source experts: {source_experts}")
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 3. Build mapping
    print("\n[3/6] Building expert mapping...")
    
    expert_mapping = {}
    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        if layer_key not in experts_to_keep:
            expert_mapping[layer_idx] = {i: i for i in range(experts_kept)}
            continue
        
        kept = sorted(experts_to_keep[layer_key])
        expert_mapping[layer_idx] = {old: new for new, old in enumerate(kept)}
    
    print(f"  Mapped {len(expert_mapping)} layers")
    
    # 4. Scan shards
    print("\n[4/6] Scanning shards...")
    
    shard_files = sorted([f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')])
    print(f"  Found {len(shard_files)} shards")
    
    weight_to_shard = {}
    shard_weights = defaultdict(list)
    
    for shard_file in tqdm(shard_files, desc="  Indexing"):
        shard_path = os.path.join(MODEL_PATH, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_to_shard[key] = shard_file
                shard_weights[shard_file].append(key)
    
    expert_pattern = re.compile(
        r'model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)'
    )
    gate_pattern = re.compile(
        r'model\.language_model\.layers\.(\d+)\.mlp\.gate\.(weight|qweight|scales|qzeros)'
    )
    
    # 5. Prune
    print("\n[5/6] Pruning model...")
    
    weight_map = {}
    total_pruned = 0
    total_kept = 0
    total_unchanged = 0
    
    for shard_file in tqdm(shard_files, desc="  Processing"):
        shard_path = os.path.join(MODEL_PATH, shard_file)
        output_path = os.path.join(OUTPUT_PATH, shard_file)
        
        new_tensors = {}
        
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # Expert weight
                expert_match = expert_pattern.match(key)
                if expert_match:
                    layer_idx = int(expert_match.group(1))
                    expert_idx = int(expert_match.group(2))
                    weight_name = expert_match.group(3)
                    
                    if expert_idx in expert_mapping.get(layer_idx, {}):
                        new_idx = expert_mapping[layer_idx][expert_idx]
                        new_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{new_idx}.{weight_name}"
                        new_tensors[new_key] = tensor
                        weight_map[new_key] = shard_file
                        total_kept += 1
                    else:
                        total_pruned += 1
                    continue
                
                # Gate weight - prune rows/cols for kept experts
                gate_match = gate_pattern.match(key)
                if gate_match:
                    layer_idx = int(gate_match.group(1))
                    weight_type = gate_match.group(2)
                    
                    kept_indices = sorted(experts_to_keep.get(str(layer_idx), []))
                    
                    if len(kept_indices) == 0 or tensor.dim() != 2:
                        new_tensors[key] = tensor
                        weight_map[key] = shard_file
                        total_unchanged += 1
                        continue
                    
                    # Handle different weight types
                    if weight_type == 'weight':
                        # [num_experts, hidden_size] -> select rows
                        if tensor.shape[0] == num_experts_original:
                            new_tensor = tensor[kept_indices, :]
                        else:
                            new_tensor = tensor
                    elif weight_type == 'qweight':
                        # AWQ packed format - be careful
                        if tensor.shape[0] == num_experts_original:
                            new_tensor = tensor[kept_indices, :]
                        elif tensor.shape[1] == num_experts_original:
                            new_tensor = tensor[:, kept_indices]
                        else:
                            new_tensor = tensor
                    elif weight_type in ['scales', 'qzeros']:
                        if tensor.shape[0] == num_experts_original:
                            new_tensor = tensor[kept_indices, :]
                        elif tensor.shape[1] == num_experts_original:
                            new_tensor = tensor[:, kept_indices]
                        else:
                            new_tensor = tensor
                    else:
                        new_tensor = tensor
                    
                    new_tensors[key] = new_tensor
                    weight_map[key] = shard_file
                    total_unchanged += 1
                    continue
                
                # Other weights
                new_tensors[key] = tensor
                weight_map[key] = shard_file
                total_unchanged += 1
        
        if new_tensors:
            save_file(new_tensors, output_path)
    
    print(f"\n  Pruned: {total_pruned}")
    print(f"  Kept: {total_kept}")
    print(f"  Unchanged: {total_unchanged}")
    
    # 6. Create config
    print("\n[6/6] Creating config files...")
    
    config = source_config.copy()
    if 'text_config' in config:
        config['text_config']['num_experts'] = experts_kept
    
    config['reap_pruning'] = {
        'pruned': True,
        'version': metadata.get('reap_version', 'v11'),
        'original_experts': num_experts_original,
        'kept_experts': experts_kept,
        'prune_ratio': prune_ratio,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(OUTPUT_PATH, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Index
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
    for fname in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json',
                  'merges.txt', 'preprocessor_config.json', 'chat_template.json',
                  'generation_config.json', 'special_tokens_map.json']:
        src = os.path.join(MODEL_PATH, fname)
        if os.path.exists(src):
            shutil.copy2(src, OUTPUT_PATH)
    
    shutil.copy2(METADATA_PATH, os.path.join(OUTPUT_PATH, 'reap_metadata.json'))
    
    # Size
    original_size = sum(
        os.path.getsize(os.path.join(MODEL_PATH, f))
        for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')
    )
    
    print(f"\n{'='*70}")
    print(f"  PRUNING COMPLETE!")
    print(f"  Original: {original_size/1e9:.2f} GB")
    print(f"  Pruned: {total_size/1e9:.2f} GB")
    print(f"  Saved: {(original_size-total_size)/1e9:.2f} GB ({(1-total_size/original_size)*100:.1f}%)")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"{'='*70}\n")
    
    if args.verify:
        verify(OUTPUT_PATH, experts_kept, num_layers)


def verify(output_path: str, expected_experts: int, expected_layers: int):
    print("\n[Verification]")
    
    with open(os.path.join(output_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    text_config = config.get('text_config', config)
    actual = text_config.get('num_experts')
    
    if actual == expected_experts:
        print(f"  Config: OK ({actual} experts)")
    else:
        print(f"  Config: MISMATCH ({actual} vs {expected_experts})")
    
    with open(os.path.join(output_path, 'model.safetensors.index.json'), 'r') as f:
        index = json.load(f)
    
    pattern = re.compile(r'model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.')
    
    layer_experts = defaultdict(set)
    for key in index['weight_map']:
        match = pattern.match(key)
        if match:
            layer_experts[int(match.group(1))].add(int(match.group(2)))
    
    issues = []
    for layer_idx in range(expected_layers):
        experts = layer_experts.get(layer_idx, set())
        if len(experts) != expected_experts:
            issues.append(f"Layer {layer_idx}: {len(experts)} experts")
        elif experts and (min(experts) != 0 or max(experts) != expected_experts - 1):
            issues.append(f"Layer {layer_idx}: non-consecutive")
    
    if issues:
        print(f"  Issues: {len(issues)}")
        for issue in issues[:5]:
            print(f"    - {issue}")
    else:
        print(f"  All {expected_layers} layers verified: OK")


if __name__ == "__main__":
    main()
