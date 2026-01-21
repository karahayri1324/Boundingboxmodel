#!/usr/bin/env python3
"""
Test script to inspect VLM model structure before running REAP.
This helps identify the correct layer paths and MoE configuration.
"""

import sys
import json
import torch
from pathlib import Path

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./models/Qwen3-VL-235B-A22B-Thinking-AWQ"

    print("=" * 60)
    print("VLM Model Structure Inspector")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print()

    # Check config
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        print(f"ERROR: config.json not found at {config_path}")
        return 1

    with open(config_path) as f:
        config = json.load(f)

    print("Config Info:")
    print(f"  Architecture: {config.get('architectures', ['Unknown'])}")
    print(f"  Model type: {config.get('model_type', 'Unknown')}")

    # Text config (MoE info)
    text_config = config.get("text_config", {})
    print(f"  Num experts: {text_config.get('num_experts', 'N/A')}")
    print(f"  Experts per token: {text_config.get('num_experts_per_tok', 'N/A')}")
    print(f"  Num layers: {text_config.get('num_hidden_layers', 'N/A')}")
    print(f"  Hidden size: {text_config.get('hidden_size', 'N/A')}")

    # Quantization
    quant = config.get("quantization_config", {})
    if quant:
        print(f"  Quantization: {quant.get('quant_method', 'None')} {quant.get('bits', '')}bit")

    print()

    # Check GPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {gpu_mem:.1f} GB")
    else:
        print("WARNING: No GPU detected!")

    print()

    # Try loading processor only (lightweight test)
    print("Testing processor load...")
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("  Processor: OK")
    except Exception as e:
        print(f"  Processor FAILED: {e}")
        return 1

    print()
    print("=" * 60)
    print("Pre-flight check complete!")
    print()
    print("Expected REAP configuration:")
    print(f"  - 128 experts -> 64 experts (50% compression)")
    print(f"  - {text_config.get('num_hidden_layers', 94)} layers to process")
    print(f"  - Estimated model size: ~125GB (AWQ)")
    print()

    # Estimate if it will fit
    model_size_gb = 125  # AWQ compressed
    if torch.cuda.is_available() and gpu_mem < model_size_gb:
        print(f"WARNING: Model (~{model_size_gb}GB) may not fit in GPU ({gpu_mem:.1f}GB)")
        print("Consider using a machine with more VRAM (H200 recommended)")
    else:
        print("GPU memory should be sufficient for this model.")

    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
