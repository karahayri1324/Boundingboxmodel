#!/usr/bin/env python3
"""
REAP V12 Model Pruning Script
==============================

Prunes experts from Qwen3-VL MoE model based on REAP V12 metadata.
Corrected version with proper weight handling.

Key fixes:
1. Proper gate weight slicing for kept experts
2. Correct expert index remapping
3. AWQ weight handling verification
4. Comprehensive validation

Usage:
    python prune_model_v12.py [--metadata PATH] [--output PATH] [--verify]
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
from typing import Dict, List, Set, Tuple, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="REAP V12 Model Pruning")
    
    default_model = os.environ.get(
        "REAP_MODEL_PATH",
        "/app/data/models/Qwen3-VL-235B-A22B-Thinking-AWQ"
    )
    default_metadata = os.environ.get(
        "REAP_OUTPUT_PATH",
        "/app/output/reap-v12"
    ) + "/reap_metadata.json"
    default_output = os.environ.get(
        "REAP_PRUNED_OUTPUT_PATH",
        "/app/output/Qwen3-VL-235B-REAP-Pruned-V12"
    )
    
    parser.add_argument("--model", type=str, default=default_model,
                        help="Source model path")
    parser.add_argument("--metadata", type=str, default=default_metadata,
                        help="REAP metadata JSON path")
    parser.add_argument("--output", type=str, default=default_output,
                        help="Output path for pruned model")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification after pruning")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be pruned without doing it")
    
    return parser.parse_args()


class ExpertMapper:
    """
    Maps original expert indices to new consecutive indices.
    
    Example: If keeping experts [3, 7, 12, 45] from 128:
        - Original expert 3 -> New expert 0
        - Original expert 7 -> New expert 1
        - Original expert 12 -> New expert 2
        - Original expert 45 -> New expert 3
    """
    
    def __init__(self, experts_to_keep: Dict[str, List[int]], num_layers: int):
        self.mapping: Dict[int, Dict[int, int]] = {}
        self.reverse_mapping: Dict[int, Dict[int, int]] = {}
        
        for layer_idx in range(num_layers):
            layer_key = str(layer_idx)
            kept = sorted(experts_to_keep.get(layer_key, []))
            
            # old_idx -> new_idx
            self.mapping[layer_idx] = {old: new for new, old in enumerate(kept)}
            # new_idx -> old_idx
            self.reverse_mapping[layer_idx] = {new: old for new, old in enumerate(kept)}
    
    def get_new_index(self, layer_idx: int, old_expert_idx: int) -> Optional[int]:
        """Get new index for an expert, or None if pruned."""
        return self.mapping.get(layer_idx, {}).get(old_expert_idx)
    
    def is_kept(self, layer_idx: int, expert_idx: int) -> bool:
        """Check if expert is kept."""
        return expert_idx in self.mapping.get(layer_idx, {})
    
    def get_kept_indices(self, layer_idx: int) -> List[int]:
        """Get sorted list of kept expert indices for a layer."""
        return sorted(self.mapping.get(layer_idx, {}).keys())


def analyze_weights(model_path: str) -> Dict:
    """Analyze model weight structure."""
    
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_index = json.load(f)
    
    # Patterns
    expert_pattern = re.compile(
        r'model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)'
    )
    gate_pattern = re.compile(
        r'model\.language_model\.layers\.(\d+)\.mlp\.gate\.(weight|qweight|scales|qzeros)'
    )
    
    analysis = {
        'expert_weights': defaultdict(lambda: defaultdict(list)),
        'gate_weights': defaultdict(list),
        'other_weights': [],
        'is_awq': False,
    }
    
    for name in weight_index['weight_map'].keys():
        expert_match = expert_pattern.match(name)
        if expert_match:
            layer = int(expert_match.group(1))
            expert = int(expert_match.group(2))
            weight_type = expert_match.group(3)
            analysis['expert_weights'][layer][expert].append(weight_type)
            continue
        
        gate_match = gate_pattern.match(name)
        if gate_match:
            layer = int(gate_match.group(1))
            weight_type = gate_match.group(2)
            analysis['gate_weights'][layer].append(weight_type)
            if weight_type == 'qweight':
                analysis['is_awq'] = True
            continue
        
        analysis['other_weights'].append(name)
    
    return analysis


def prune_gate_weight(
    tensor: torch.Tensor,
    kept_indices: List[int],
    weight_type: str,
    num_experts_original: int
) -> torch.Tensor:
    """
    Prune gate weight tensor to only include kept experts.
    
    Gate weight shapes:
    - FP16 weight: [num_experts, hidden_size]
    - AWQ qweight: varies, typically [hidden_size/8, num_experts] or similar
    - AWQ scales/qzeros: [num_groups, num_experts]
    """
    kept_tensor = torch.tensor(kept_indices, dtype=torch.long, device=tensor.device)
    
    if weight_type == 'weight':
        # Standard: [num_experts, hidden_size] -> select rows
        if tensor.shape[0] == num_experts_original:
            return tensor.index_select(0, kept_tensor)
        else:
            print(f"    Warning: Unexpected weight shape {tensor.shape}")
            return tensor
    
    elif weight_type == 'qweight':
        # AWQ packed: need to check which dimension has num_experts
        if tensor.shape[0] == num_experts_original:
            return tensor.index_select(0, kept_tensor)
        elif tensor.shape[1] == num_experts_original:
            return tensor.index_select(1, kept_tensor)
        else:
            # May be packed differently
            print(f"    Warning: qweight shape {tensor.shape}, keeping as-is")
            return tensor
    
    elif weight_type in ['scales', 'qzeros']:
        # Usually [num_groups, num_experts] or [num_experts, ...]
        if tensor.shape[0] == num_experts_original:
            return tensor.index_select(0, kept_tensor)
        elif tensor.shape[1] == num_experts_original:
            return tensor.index_select(1, kept_tensor)
        else:
            print(f"    Warning: {weight_type} shape {tensor.shape}, keeping as-is")
            return tensor
    
    return tensor


def main():
    args = parse_args()
    
    MODEL_PATH = args.model
    METADATA_PATH = args.metadata
    OUTPUT_PATH = args.output
    
    print("\n" + "="*70)
    print("  REAP V12 Model Pruning")
    print("="*70)
    
    # 1. Load metadata
    print("\n[1/7] Loading REAP metadata...")
    
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
    
    print(f"  REAP Version: {metadata.get('reap_version', 'unknown')}")
    print(f"  Experts: {num_experts_original} -> {experts_kept}")
    print(f"  Prune ratio: {prune_ratio*100:.0f}%")
    
    # 2. Validate source model
    print("\n[2/7] Validating source model...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    with open(os.path.join(MODEL_PATH, 'config.json'), 'r') as f:
        source_config = json.load(f)
    
    text_config = source_config.get('text_config', source_config)
    source_experts = text_config.get('num_experts', 128)
    
    if source_experts != num_experts_original:
        print(f"  WARNING: Config mismatch - model has {source_experts}, metadata expects {num_experts_original}")
    
    print(f"  Source experts: {source_experts}")
    
    # 3. Analyze weight structure
    print("\n[3/7] Analyzing weight structure...")
    
    analysis = analyze_weights(MODEL_PATH)
    is_awq = analysis['is_awq']
    
    print(f"  Is AWQ quantized: {is_awq}")
    print(f"  Layers with experts: {len(analysis['expert_weights'])}")
    print(f"  Other weights: {len(analysis['other_weights'])}")
    
    # 4. Build mapping
    print("\n[4/7] Building expert mapping...")
    
    mapper = ExpertMapper(experts_to_keep, num_layers)
    
    # Verify mapping
    sample_layer = 0
    kept = mapper.get_kept_indices(sample_layer)
    print(f"  Sample (Layer 0): Keeping experts {kept[:5]}... ({len(kept)} total)")
    
    if args.dry_run:
        print("\n  DRY RUN - Exiting without pruning")
        return
    
    # 5. Setup output
    print("\n[5/7] Setting up output directory...")
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 6. Prune weights
    print("\n[6/7] Pruning model weights...")
    
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_index = json.load(f)
    
    shard_files = sorted(set(weight_index['weight_map'].values()))
    print(f"  Processing {len(shard_files)} shards...")
    
    # Patterns
    expert_pattern = re.compile(
        r'model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)'
    )
    gate_pattern = re.compile(
        r'model\.language_model\.layers\.(\d+)\.mlp\.gate\.(weight|qweight|scales|qzeros)'
    )
    
    new_weight_map = {}
    stats = {'pruned': 0, 'kept': 0, 'gate_modified': 0, 'unchanged': 0}
    
    for shard_file in tqdm(shard_files, desc="  Processing shards"):
        shard_path = os.path.join(MODEL_PATH, shard_file)
        output_path = os.path.join(OUTPUT_PATH, shard_file)
        
        new_tensors = {}
        
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # Check if expert weight
                expert_match = expert_pattern.match(key)
                if expert_match:
                    layer_idx = int(expert_match.group(1))
                    expert_idx = int(expert_match.group(2))
                    weight_name = expert_match.group(3)
                    
                    new_idx = mapper.get_new_index(layer_idx, expert_idx)
                    
                    if new_idx is not None:
                        # Keep with new index
                        new_key = f"model.language_model.layers.{layer_idx}.mlp.experts.{new_idx}.{weight_name}"
                        new_tensors[new_key] = tensor
                        new_weight_map[new_key] = shard_file
                        stats['kept'] += 1
                    else:
                        # Prune
                        stats['pruned'] += 1
                    continue
                
                # Check if gate weight
                gate_match = gate_pattern.match(key)
                if gate_match:
                    layer_idx = int(gate_match.group(1))
                    weight_type = gate_match.group(2)
                    
                    kept_indices = mapper.get_kept_indices(layer_idx)
                    
                    if len(kept_indices) > 0:
                        new_tensor = prune_gate_weight(
                            tensor, kept_indices, weight_type, num_experts_original
                        )
                        new_tensors[key] = new_tensor
                        new_weight_map[key] = shard_file
                        stats['gate_modified'] += 1
                    else:
                        new_tensors[key] = tensor
                        new_weight_map[key] = shard_file
                        stats['unchanged'] += 1
                    continue
                
                # Other weights - keep as-is
                new_tensors[key] = tensor
                new_weight_map[key] = shard_file
                stats['unchanged'] += 1
        
        # Save shard
        if new_tensors:
            save_file(new_tensors, output_path)
    
    print(f"\n  Statistics:")
    print(f"    Expert weights pruned: {stats['pruned']}")
    print(f"    Expert weights kept: {stats['kept']}")
    print(f"    Gate weights modified: {stats['gate_modified']}")
    print(f"    Other weights: {stats['unchanged']}")
    
    # 7. Create config and index
    print("\n[7/7] Creating config and index files...")
    
    # Update config
    new_config = source_config.copy()
    if 'text_config' in new_config:
        new_config['text_config']['num_experts'] = experts_kept
    else:
        new_config['num_experts'] = experts_kept
    
    new_config['reap_pruning'] = {
        'pruned': True,
        'version': metadata.get('reap_version', 'v12'),
        'original_experts': num_experts_original,
        'kept_experts': experts_kept,
        'prune_ratio': prune_ratio,
        'formula': metadata.get('formula', 'REAP'),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(OUTPUT_PATH, 'config.json'), 'w') as f:
        json.dump(new_config, f, indent=2)
    
    # Create index
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_PATH, f))
        for f in os.listdir(OUTPUT_PATH) if f.endswith('.safetensors')
    )
    
    new_index = {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map
    }
    
    with open(os.path.join(OUTPUT_PATH, 'model.safetensors.index.json'), 'w') as f:
        json.dump(new_index, f, indent=2)
    
    # Copy support files
    support_files = [
        'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
        'merges.txt', 'preprocessor_config.json', 'chat_template.json',
        'generation_config.json', 'special_tokens_map.json'
    ]
    
    for fname in support_files:
        src = os.path.join(MODEL_PATH, fname)
        if os.path.exists(src):
            shutil.copy2(src, OUTPUT_PATH)
    
    # Copy REAP metadata
    shutil.copy2(METADATA_PATH, os.path.join(OUTPUT_PATH, 'reap_metadata.json'))
    
    # Size comparison
    original_size = sum(
        os.path.getsize(os.path.join(MODEL_PATH, f))
        for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')
    )
    
    print(f"\n{'='*70}")
    print(f"  PRUNING COMPLETE!")
    print(f"{'='*70}")
    print(f"  Original size: {original_size/1e9:.2f} GB")
    print(f"  Pruned size: {total_size/1e9:.2f} GB")
    print(f"  Size saved: {(original_size-total_size)/1e9:.2f} GB ({(1-total_size/original_size)*100:.1f}%)")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"{'='*70}\n")
    
    if args.verify:
        verify_pruned_model(OUTPUT_PATH, experts_kept, num_layers)


def verify_pruned_model(output_path: str, expected_experts: int, expected_layers: int):
    """Verify the pruned model structure."""
    
    print("\n[Verification]")
    
    # Check config
    with open(os.path.join(output_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    text_config = config.get('text_config', config)
    actual_experts = text_config.get('num_experts')
    
    if actual_experts == expected_experts:
        print(f"  ✓ Config: {actual_experts} experts (correct)")
    else:
        print(f"  ✗ Config: {actual_experts} experts (expected {expected_experts})")
    
    # Check weight index
    with open(os.path.join(output_path, 'model.safetensors.index.json'), 'r') as f:
        index = json.load(f)
    
    expert_pattern = re.compile(
        r'model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.'
    )
    
    layer_experts: Dict[int, Set[int]] = defaultdict(set)
    
    for key in index['weight_map'].keys():
        match = expert_pattern.match(key)
        if match:
            layer = int(match.group(1))
            expert = int(match.group(2))
            layer_experts[layer].add(expert)
    
    # Verify each layer
    issues = []
    
    for layer_idx in range(expected_layers):
        experts = layer_experts.get(layer_idx, set())
        
        if len(experts) != expected_experts:
            issues.append(f"Layer {layer_idx}: {len(experts)} experts (expected {expected_experts})")
        elif experts != set(range(expected_experts)):
            issues.append(f"Layer {layer_idx}: non-consecutive indices {sorted(experts)[:5]}...")
    
    if issues:
        print(f"  ✗ Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"      - {issue}")
        if len(issues) > 10:
            print(f"      ... and {len(issues)-10} more")
    else:
        print(f"  ✓ All {expected_layers} layers verified: {expected_experts} consecutive experts each")
    
    # Check REAP metadata
    reap_meta_path = os.path.join(output_path, 'reap_metadata.json')
    if os.path.exists(reap_meta_path):
        print(f"  ✓ REAP metadata present")
    else:
        print(f"  ✗ REAP metadata missing")
    
    print()


if __name__ == "__main__":
    main()