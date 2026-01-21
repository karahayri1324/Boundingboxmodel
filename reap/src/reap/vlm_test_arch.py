"""
Test script to verify VLM model architecture and MoE structure.
Run this before full REAP to ensure compatibility.
"""

import torch
from transformers import AutoProcessor, AutoModel
import sys


def test_vlm_architecture(model_path: str):
    """Test VLM model architecture and MoE structure."""
    print("=" * 60)
    print("VLM Architecture Test")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print()

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    print(f"Processor type: {type(processor).__name__}")
    print()

    # Load model (CPU only for testing)
    print("Loading model (CPU only for architecture check)...")
    model = AutoModel.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"Model class: {model.__class__.__name__}")
    print()

    # Check model structure
    print("Model structure:")
    print("-" * 40)

    # Check if it has language_model
    if hasattr(model, "language_model"):
        lm = model.language_model
        print(f"✓ Has language_model: {type(lm).__name__}")

        if hasattr(lm, "layers"):
            layers = lm.layers
            print(f"✓ Has layers: {len(layers)} layers")

            # Check first layer for MoE
            layer0 = layers[0]
            print(f"  Layer 0 attributes: {[a for a in dir(layer0) if not a.startswith('_')]}")

            if hasattr(layer0, "mlp"):
                mlp = layer0.mlp
                print(f"  ✓ Has mlp: {type(mlp).__name__}")

                # Check for MoE components
                if hasattr(mlp, "experts"):
                    experts = mlp.experts
                    print(f"    ✓ Has experts: {len(experts)} experts")
                    print(f"    Expert type: {type(experts[0]).__name__}")

                if hasattr(mlp, "gate"):
                    gate = mlp.gate
                    print(f"    ✓ Has gate (router): {type(gate).__name__}")
                    if hasattr(gate, "weight"):
                        print(f"    Router shape: {gate.weight.shape}")

                if hasattr(mlp, "top_k"):
                    print(f"    ✓ top_k: {mlp.top_k}")

    # Check if it has model.model.layers (alternative path)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        print("✓ Has model.model.layers structure")
        layers = model.model.layers
        print(f"  Num layers: {len(layers)}")

    # Check visual encoder
    if hasattr(model, "visual"):
        print(f"✓ Has visual encoder: {type(model.visual).__name__}")
    elif hasattr(model, "vision_model"):
        print(f"✓ Has vision_model: {type(model.vision_model).__name__}")

    # Check config
    print()
    print("Model config:")
    print("-" * 40)
    config = model.config
    print(f"Model type: {getattr(config, 'model_type', 'N/A')}")

    if hasattr(config, "text_config"):
        tc = config.text_config
        print(f"Text config:")
        print(f"  num_experts: {getattr(tc, 'num_experts', 'N/A')}")
        print(f"  num_experts_per_tok: {getattr(tc, 'num_experts_per_tok', 'N/A')}")
        print(f"  num_hidden_layers: {getattr(tc, 'num_hidden_layers', 'N/A')}")
        print(f"  hidden_size: {getattr(tc, 'hidden_size', 'N/A')}")

    if hasattr(config, "quantization_config"):
        qc = config.quantization_config
        print(f"Quantization config: {qc}")

    print()
    print("=" * 60)
    print("Architecture test complete!")
    print("=" * 60)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return True


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/vault/Boundingboxmodel/Qwen3-VL-235B-A22B-Thinking-AWQ"
    test_vlm_architecture(model_path)
