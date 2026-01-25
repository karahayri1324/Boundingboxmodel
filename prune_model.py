#!/usr/bin/env python3
"""
REAP Pruning Script V12.4 - Qwen3-VL-235B-A22B-Thinking-AWQ
============================================================

FIXES in V12.4:
1. Gate weight slicing - auto-detect expert dimension
2. Proper handling of FP16 gate vs AWQ expert weights

Usage:
    python prune_model_v12.4.py --metadata /path/to/reap_metadata.json --output /path/to/output
"""

import argparse
import json
import os
import sys
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# =============================================================================
# EXPERT MAPPING
# =============================================================================

class ExpertMapper:
    """Maps original expert indices to new consecutive indices."""
    
    def __init__(self, experts_to_keep: Dict[str, List[int]], num_layers: int):
        self.num_layers = num_layers
        self.layer_maps: Dict[int, Dict[int, int]] = {}
        
        for layer_idx in range(num_layers):
            kept = sorted(experts_to_keep.get(str(layer_idx), []))
            self.layer_maps[layer_idx] = {
                old_idx: new_idx 
                for new_idx, old_idx in enumerate(kept)
            }
    
    def get_new_index(self, layer_idx: int, old_expert_idx: int) -> Optional[int]:
        """Get new expert index, or None if pruned."""
        return self.layer_maps.get(layer_idx, {}).get(old_expert_idx)
    
    def get_kept_experts(self, layer_idx: int) -> List[int]:
        """Get list of kept expert indices (original)."""
        return sorted(self.layer_maps.get(layer_idx, {}).keys())
    
    def get_num_kept(self, layer_idx: int) -> int:
        """Get number of kept experts for a layer."""
        return len(self.layer_maps.get(layer_idx, {}))


# =============================================================================
# GATE WEIGHT PRUNING (FIXED)
# =============================================================================

def prune_gate_weight(
    tensor: torch.Tensor,
    kept_indices: List[int],
    weight_type: str,
    num_experts_original: int
) -> torch.Tensor:
    """
    Prune gate weight tensor to only include kept experts.
    
    FIX in V12.4: Auto-detect expert dimension instead of assuming dim=0
    
    Qwen3-VL gate shapes (FP16, not AWQ):
    - weight: [hidden_size, num_experts] = [4096, 128] -> slice dim=1
    
    But nn.Linear stores as [out_features, in_features], so:
    - If gate = nn.Linear(hidden_size, num_experts)
    - Then weight.shape = [num_experts, hidden_size] = [128, 4096] -> slice dim=0
    
    We auto-detect by finding which dimension equals num_experts.
    """
    kept_tensor = torch.tensor(kept_indices, dtype=torch.long, device=tensor.device)
    
    # Find dimension with num_experts
    expert_dim = None
    for dim, size in enumerate(tensor.shape):
        if size == num_experts_original:
            expert_dim = dim
            break
    
    if expert_dim is None:
        print(f"    ⚠️ Cannot find expert dim ({num_experts_original}) in shape {tensor.shape}")
        print(f"    Skipping gate pruning for this tensor")
        return tensor
    
    print(f"    Gate {weight_type}: {list(tensor.shape)} -> slicing dim {expert_dim} "
          f"({num_experts_original} -> {len(kept_indices)})")
    
    return tensor.index_select(expert_dim, kept_tensor)


def prune_gate_awq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    kept_indices: List[int],
    num_experts_original: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prune AWQ quantized gate weight.
    
    Note: For Qwen3-VL-AWQ, gate is NOT quantized (modules_to_not_convert).
    This function exists for other models that might have AWQ gate.
    
    AWQ shapes:
    - qweight: [in_features/8, out_features] packed int32
    - scales: [num_groups, out_features]
    - qzeros: [num_groups, out_features]
    
    out_features = num_experts for gate
    """
    kept_tensor = torch.tensor(kept_indices, dtype=torch.long, device=qweight.device)
    
    # For AWQ, num_experts is typically in dim=1 (out_features)
    expert_dim = None
    
    # Check scales shape first (most reliable)
    for dim, size in enumerate(scales.shape):
        if size == num_experts_original:
            expert_dim = dim
            break
    
    if expert_dim is None:
        print(f"    ⚠️ Cannot find expert dim in AWQ gate scales {scales.shape}")
        return qweight, qzeros, scales
    
    print(f"    Gate AWQ: slicing dim {expert_dim} ({num_experts_original} -> {len(kept_indices)})")
    
    new_qweight = qweight.index_select(expert_dim, kept_tensor)
    new_qzeros = qzeros.index_select(expert_dim, kept_tensor)
    new_scales = scales.index_select(expert_dim, kept_tensor)
    
    return new_qweight, new_qzeros, new_scales


# =============================================================================
# WEIGHT PROCESSING
# =============================================================================

def process_shard(
    shard_path: str,
    shard_weights: List[str],
    expert_mapper: ExpertMapper,
    config: dict,
    output_shard_path: str
) -> Dict[str, torch.Tensor]:
    """Process a single shard, pruning expert weights."""
    
    layer_prefix = config.get('layer_prefix', 'model.language_model.layers')
    num_experts_original = config['num_experts_original']
    
    new_weights = {}
    
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for weight_name in tqdm(shard_weights, desc=f"  {os.path.basename(shard_path)}", leave=False):
            tensor = f.get_tensor(weight_name)
            
            # Check if this is an expert weight
            if '.mlp.experts.' in weight_name:
                # Parse layer and expert index
                # Format: model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{type}
                parts = weight_name.split('.')
                
                layer_idx = None
                expert_idx = None
                
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                        except ValueError:
                            pass
                    if part == 'experts' and i + 1 < len(parts):
                        try:
                            expert_idx = int(parts[i + 1])
                        except ValueError:
                            pass
                
                if layer_idx is None or expert_idx is None:
                    print(f"    ⚠️ Cannot parse: {weight_name}")
                    new_weights[weight_name] = tensor
                    continue
                
                # Check if expert is kept
                new_expert_idx = expert_mapper.get_new_index(layer_idx, expert_idx)
                
                if new_expert_idx is None:
                    # Expert is pruned, skip
                    continue
                
                # Rename with new expert index
                new_name = weight_name.replace(
                    f'.experts.{expert_idx}.',
                    f'.experts.{new_expert_idx}.'
                )
                new_weights[new_name] = tensor
                
            elif '.mlp.gate.' in weight_name:
                # Gate weight - needs special handling
                # Parse layer index
                parts = weight_name.split('.')
                layer_idx = None
                
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                        except ValueError:
                            pass
                
                if layer_idx is None:
                    print(f"    ⚠️ Cannot parse gate layer: {weight_name}")
                    new_weights[weight_name] = tensor
                    continue
                
                kept_experts = expert_mapper.get_kept_experts(layer_idx)
                
                # Determine weight type
                if weight_name.endswith('.weight'):
                    # FP16 weight (Qwen3-VL gate is not AWQ quantized)
                    new_tensor = prune_gate_weight(
                        tensor, kept_experts, 'weight', num_experts_original
                    )
                    new_weights[weight_name] = new_tensor
                    
                elif weight_name.endswith('.bias'):
                    # Bias (if exists)
                    new_tensor = prune_gate_weight(
                        tensor, kept_experts, 'bias', num_experts_original
                    )
                    new_weights[weight_name] = new_tensor
                    
                elif weight_name.endswith('.qweight'):
                    # AWQ qweight (shouldn't happen for Qwen3-VL gate, but handle it)
                    print(f"    ⚠️ Gate has AWQ qweight (unexpected): {weight_name}")
                    new_weights[weight_name] = tensor
                    
                elif weight_name.endswith('.scales'):
                    # AWQ scales
                    new_tensor = prune_gate_weight(
                        tensor, kept_experts, 'scales', num_experts_original
                    )
                    new_weights[weight_name] = new_tensor
                    
                elif weight_name.endswith('.qzeros'):
                    # AWQ qzeros
                    new_tensor = prune_gate_weight(
                        tensor, kept_experts, 'qzeros', num_experts_original
                    )
                    new_weights[weight_name] = new_tensor
                    
                else:
                    new_weights[weight_name] = tensor
            else:
                # Non-expert, non-gate weight - keep as is
                new_weights[weight_name] = tensor
    
    return new_weights


# =============================================================================
# MAIN PRUNING LOGIC
# =============================================================================

def prune_model(
    metadata_path: str,
    output_path: str,
    verify_only: bool = False
):
    """Main pruning function."""
    
    print("\n" + "="*70)
    print("  REAP Pruning V12.4 - Qwen3-VL-235B-A22B-Thinking-AWQ")
    print("="*70)
    
    # Load metadata
    print("\n[1/5] Loading metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_path = metadata.get('model_path', os.path.dirname(metadata_path))
    experts_to_keep = metadata['experts_to_keep']
    num_layers = metadata['num_layers']
    num_experts_original = metadata['num_experts_original']
    experts_kept_per_layer = metadata['experts_kept']
    
    print(f"  Model: {model_path}")
    print(f"  Layers: {num_layers}")
    print(f"  Experts: {num_experts_original} -> {experts_kept_per_layer} per layer")
    
    # Create expert mapper
    expert_mapper = ExpertMapper(experts_to_keep, num_layers)
    
    # Verify mapping
    print("\n[2/5] Verifying expert mapping...")
    for layer_idx in [0, num_layers // 2, num_layers - 1]:
        kept = expert_mapper.get_kept_experts(layer_idx)
        print(f"  Layer {layer_idx}: keeping {len(kept)} experts")
        print(f"    First 5: {kept[:5]}")
        print(f"    Last 5: {kept[-5:]}")
    
    if verify_only:
        print("\n  Verify-only mode, exiting.")
        return
    
    # Load weight index
    print("\n[3/5] Loading weight index...")
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        weight_index = json.load(f)
    
    weight_map = weight_index['weight_map']
    
    # Group weights by shard
    shard_to_weights = defaultdict(list)
    for weight_name, shard in weight_map.items():
        shard_to_weights[shard].append(weight_name)
    
    print(f"  Total shards: {len(shard_to_weights)}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Config for processing
    process_config = {
        'layer_prefix': metadata.get('layer_prefix', 'model.language_model.layers'),
        'num_experts_original': num_experts_original,
    }
    
    # Process each shard
    print("\n[4/5] Processing shards...")
    new_weight_map = {}
    total_original = 0
    total_pruned = 0
    
    for shard_name, shard_weights in tqdm(shard_to_weights.items(), desc="  Shards"):
        shard_path = os.path.join(model_path, shard_name)
        output_shard_path = os.path.join(output_path, shard_name)
        
        total_original += len(shard_weights)
        
        # Process shard
        new_weights = process_shard(
            shard_path, shard_weights, expert_mapper, 
            process_config, output_shard_path
        )
        
        total_pruned += len(new_weights)
        
        # Update weight map
        for weight_name in new_weights.keys():
            new_weight_map[weight_name] = shard_name
        
        # Save new shard
        if new_weights:
            save_file(new_weights, output_shard_path)
    
    print(f"\n  Weights: {total_original} -> {total_pruned} ({total_pruned/total_original*100:.1f}%)")
    
    # Save new index
    print("\n[5/5] Saving metadata and configs...")
    
    new_index = {
        'metadata': weight_index.get('metadata', {}),
        'weight_map': new_weight_map
    }
    
    with open(os.path.join(output_path, 'model.safetensors.index.json'), 'w') as f:
        json.dump(new_index, f, indent=2)
    
    # Update and save config
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # Update expert count in text_config
    if 'text_config' in model_config:
        model_config['text_config']['num_experts'] = experts_kept_per_layer
    else:
        model_config['num_experts'] = experts_kept_per_layer
    
    # Add REAP metadata
    model_config['reap_pruning'] = {
        'version': metadata.get('reap_version', 'v12.4'),
        'original_experts': num_experts_original,
        'kept_experts': experts_kept_per_layer,
        'prune_ratio': metadata.get('prune_ratio', 0.4),
        'formula': metadata.get('formula', 'REAP'),
    }
    
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Copy other config files
    for fname in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                  'chat_template.json', 'preprocessor_config.json', 'generation_config.json',
                  'vocab.json', 'merges.txt']:
        src = os.path.join(model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, output_path)
    
    # Save pruning metadata
    with open(os.path.join(output_path, 'reap_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  PRUNING COMPLETE!")
    print(f"  Output: {output_path}")
    print(f"  Experts: {num_experts_original} -> {experts_kept_per_layer}")
    print(f"  Weights: {total_original} -> {total_pruned}")
    print(f"{'='*70}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='REAP Model Pruning V12.4')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to reap_metadata.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: {metadata_dir}/pruned)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify mapping only, do not prune')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.metadata), 'pruned')
    
    prune_model(args.metadata, args.output, args.verify)


if __name__ == "__main__":
    main()