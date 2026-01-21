"""
VLM REAP - Vision Language Model Expert Pruning with REAP criterion
Supports Qwen3-VL and similar VLM architectures with MoE layers.
"""

from __future__ import annotations
import json
import logging
import pathlib
import time
import gc
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoModel,
    HfArgumentParser,
)
from accelerate.utils import set_seed

# Try to import qwen_vl_utils for better image processing
try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False

# Try to import autoawq for AWQ support
try:
    import awq
    HAS_AWQ = True
except ImportError:
    HAS_AWQ = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# VLM Model Attributes Configuration
# ============================================================================

VLM_MODEL_ATTRS = {
    "Qwen3VLMoeForConditionalGeneration": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
        # VLM-specific: path to language model layers
        "layers_path": "model.language_model.layers",
    },
}


# ============================================================================
# Arguments
# ============================================================================

@dataclass
class VLMReapArgs:
    """Arguments for VLM REAP pruning."""
    model_path: str = field(
        metadata={"help": "Path to the VLM model"}
    )
    calibration_data_path: str = field(
        metadata={"help": "Path to calibration data directory containing calibration_data.json and images/"}
    )
    output_path: str = field(
        default="./artifacts/vlm_pruned",
        metadata={"help": "Output path for pruned model"}
    )
    compression_ratio: float = field(
        default=0.5,
        metadata={"help": "Compression ratio (0.5 = prune 50% of experts)"}
    )
    prune_method: str = field(
        default="reap",
        metadata={"help": "Pruning method: reap, frequency, ean_mean, weighted_ean_sum"}
    )
    num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to use for calibration (None = all)"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    max_pixels: int = field(
        default=1280 * 28 * 28,
        metadata={"help": "Maximum pixels for image processing"}
    )
    min_pixels: int = field(
        default=256 * 28 * 28,
        metadata={"help": "Minimum pixels for image processing"}
    )
    preserve_super_experts: bool = field(
        default=True,
        metadata={"help": "Preserve super experts (high activation) from pruning"}
    )
    renormalize_router_weights: bool = field(
        default=True,
        metadata={"help": "Renormalize router weights after top-k selection"}
    )
    save_observer_data: bool = field(
        default=True,
        metadata={"help": "Save observer data for analysis"}
    )


# ============================================================================
# VLM Dataset Loader
# ============================================================================

class VLMCalibrationDataset:
    """
    Dataset loader for VLM calibration data.
    Each region in an image becomes a separate sample.
    """

    def __init__(
        self,
        data_path: str,
        processor: AutoProcessor,
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 256 * 28 * 28,
        num_samples: Optional[int] = None,
    ):
        self.data_path = pathlib.Path(data_path)
        self.processor = processor
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        # Load calibration data
        json_path = self.data_path / "calibration_data.json"
        with open(json_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        # Expand regions into individual samples
        self.samples = []
        for item in self.raw_data:
            image_path = self.data_path / item["image"]
            for region in item["regions"]:
                self.samples.append({
                    "id": item["id"],
                    "image_path": str(image_path),
                    "region": region,
                })

        # Limit samples if specified
        if num_samples is not None:
            self.samples = self.samples[:num_samples]

        logger.info(f"Loaded {len(self.samples)} samples from {len(self.raw_data)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def prepare_input(self, sample: dict, device: torch.device) -> dict:
        """Prepare a single sample for model input."""
        image_path = sample["image_path"]
        region_text = sample["region"]

        # Create conversation format for Qwen3-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": f"Bu görselde şu bölgeyi analiz et: {region_text}"}
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process images using qwen_vl_utils if available
        if HAS_QWEN_VL_UTILS:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image = Image.open(image_path).convert("RGB")
            image_inputs = [image]
            video_inputs = None

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        return inputs


# ============================================================================
# VLM MoE Observer
# ============================================================================

class OnlineStatsTracker:
    """Track online statistics (mean) with weighted updates."""

    def __init__(self, shape, count_shape, device="cpu", dtype=torch.float32):
        self.mean = torch.zeros(shape, device=device, dtype=dtype)
        self.count = torch.zeros(count_shape, device=device, dtype=torch.float64)

    def update(self, value, weight):
        """Update with new value and weight."""
        value = value.to(self.mean.device, self.mean.dtype)
        weight = weight.to(self.count.device, self.count.dtype)

        # Weighted online mean update
        new_count = self.count + weight
        # Avoid division by zero
        mask = new_count > 0
        if mask.any():
            delta = value - self.mean
            self.mean = torch.where(
                mask if mask.dim() == self.mean.dim() else mask.unsqueeze(-1).expand_as(self.mean),
                self.mean + delta * weight / new_count.clamp(min=1e-10),
                self.mean
            )
        self.count = new_count


class VLMMoEObserver:
    """
    Observer for collecting MoE activation statistics in VLM models.
    Hooks into MoE layers to collect router logits and expert activations.
    """

    def __init__(
        self,
        model: nn.Module,
        model_attrs: dict,
        renormalize_router_weights: bool = True,
    ):
        self.model = model
        self.model_attrs = model_attrs
        self.renormalize = renormalize_router_weights
        self.hooks = []
        self.state = {}

        self._hook_model()
        logger.info(f"VLMMoEObserver initialized for {model.__class__.__name__}")

    def _get_layers(self):
        """Get the transformer layers from the VLM model."""
        layers_path = self.model_attrs.get("layers_path", "model.language_model.layers")
        obj = self.model
        for attr in layers_path.split("."):
            obj = getattr(obj, attr)
        return obj

    def _get_moe(self, layer_idx: int):
        """Get MoE module from a specific layer."""
        layers = self._get_layers()
        moe_attr = self.model_attrs["moe_block"]
        return getattr(layers[layer_idx], moe_attr)

    def _initialize_state(self, num_experts: int, hidden_dim: int, layer_idx: int):
        """Initialize state for a layer."""
        device = "cpu"
        state = {}

        # Token counts
        state["total_tokens"] = torch.tensor(0, device=device, dtype=torch.long)
        state["expert_frequency"] = torch.zeros(num_experts, device=device, dtype=torch.long)

        # Expert Activation Norm (EAN)
        state["ean_sum"] = torch.zeros(num_experts, device=device, dtype=torch.float64)
        state["weighted_ean_sum"] = torch.zeros(num_experts, device=device, dtype=torch.float64)
        state["ean_mean"] = OnlineStatsTracker(
            shape=(num_experts,),
            count_shape=(num_experts,),
            device=device,
            dtype=torch.float32,
        )

        # REAP criterion
        state["reap"] = OnlineStatsTracker(
            shape=(num_experts,),
            count_shape=(num_experts,),
            device=device,
            dtype=torch.float32,
        )

        # Weighted frequency
        state["weighted_expert_frequency_sum"] = torch.zeros(
            num_experts, device=device, dtype=torch.float64
        )

        # Super experts tracking
        state["max_activations"] = torch.zeros(
            num_experts, device=device, dtype=torch.float32
        )

        return state

    def _hook_model(self):
        """Register hooks on all MoE layers."""
        layers = self._get_layers()
        moe_attr = self.model_attrs["moe_block"]

        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, moe_attr):
                moe = getattr(layer, moe_attr)
                # Check if this is actually an MoE layer (has experts)
                if hasattr(moe, self.model_attrs["experts"]):
                    hook_fn = self._create_hook(layer_idx, moe)
                    hook = moe.register_forward_hook(hook_fn)
                    self.hooks.append(hook)
                    logger.debug(f"Hooked MoE layer {layer_idx}")

        logger.info(f"Registered {len(self.hooks)} MoE hooks")

    def _create_hook(self, layer_idx: int, moe: nn.Module):
        """Create a forward hook for an MoE layer."""
        num_experts = len(getattr(moe, self.model_attrs["experts"]))
        router = getattr(moe, self.model_attrs["router"])
        experts = getattr(moe, self.model_attrs["experts"])
        top_k = getattr(moe, "top_k", 8)  # Default for Qwen3-VL

        @torch.no_grad()
        def hook_fn(module, args, output):
            try:
                input_tensor = args[0]  # (batch, seq, hidden)
                device = input_tensor.device

                batch_size, seq_len, hidden_dim = input_tensor.shape
                flat_input = input_tensor.view(-1, hidden_dim)
                num_tokens = flat_input.shape[0]

                # Initialize state if needed
                if layer_idx not in self.state:
                    self.state[layer_idx] = self._initialize_state(
                        num_experts, hidden_dim, layer_idx
                    )

                # Get router logits
                router_logits = router(flat_input)  # (num_tokens, num_experts)
                routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

                # Get top-k experts
                _, selected_experts = torch.topk(router_logits, top_k, dim=-1)

                # Expert frequency
                expert_frequency = torch.bincount(
                    selected_experts.view(-1), minlength=num_experts
                ).to(device)

                # Update counts
                self.state[layer_idx]["total_tokens"] += num_tokens
                self.state[layer_idx]["expert_frequency"] += expert_frequency.cpu().long()

                # Renormalize if needed
                if self.renormalize:
                    topk_weights = torch.gather(routing_weights, 1, selected_experts)
                    routing_weights = routing_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)

                # Compute activations for each expert
                ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
                ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
                weighted_ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
                reap = torch.zeros(num_experts, device=device, dtype=torch.float32)
                weighted_freq_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)

                prior_max = self.state[layer_idx]["max_activations"]

                for i in range(num_experts):
                    active_mask = (selected_experts == i).any(dim=-1)
                    if not active_mask.any():
                        continue

                    active_inputs = flat_input[active_mask]
                    active_weights = routing_weights[active_mask, i]

                    # Compute expert output
                    expert_out = experts[i](active_inputs)
                    ean_norm = torch.linalg.norm(expert_out, dim=-1)

                    ean_sum[i] = ean_norm.sum()
                    ean_mean[i] = ean_norm.mean()
                    weighted_freq_sum[i] = active_weights.sum()
                    weighted_ean_sum[i] = (ean_norm * active_weights).sum()
                    reap[i] = (ean_norm * active_weights).mean()

                    # Track max activations for super expert detection
                    max_act = expert_out.abs().max()
                    if max_act > prior_max[i]:
                        self.state[layer_idx]["max_activations"][i] = max_act.cpu()

                # Update state
                self.state[layer_idx]["ean_sum"] += ean_sum.cpu()
                self.state[layer_idx]["ean_mean"].update(ean_mean.cpu(), expert_frequency.cpu())
                self.state[layer_idx]["weighted_ean_sum"] += weighted_ean_sum.cpu()
                self.state[layer_idx]["reap"].update(reap.cpu(), expert_frequency.cpu())
                self.state[layer_idx]["weighted_expert_frequency_sum"] += weighted_freq_sum.cpu()

            except Exception as e:
                logger.warning(f"Hook error at layer {layer_idx}: {e}")

        return hook_fn

    def get_state(self) -> dict:
        """Get the collected statistics."""
        result = {}
        for layer_idx, layer_state in self.state.items():
            result[layer_idx] = {}
            for key, value in layer_state.items():
                if isinstance(value, OnlineStatsTracker):
                    result[layer_idx][key] = value.mean
                else:
                    result[layer_idx][key] = value
        return result

    def reset(self):
        """Reset observer state."""
        del self.state
        gc.collect()
        self.state = {}

    def close(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# Super Expert Detection
# ============================================================================

def get_super_expert_indices(observer_data: dict, quantile: float = 99.5) -> torch.Tensor:
    """Identify super experts based on max activation values."""
    logger.info("Identifying super experts...")

    all_max_activations = []
    for layer_data in observer_data.values():
        all_max_activations.append(layer_data["max_activations"])

    all_max = torch.cat(all_max_activations)
    threshold = torch.quantile(all_max, quantile / 100.0).item()

    # Also use absolute threshold (max / 10)
    abs_threshold = all_max.max().item() / 10
    final_threshold = max(threshold, abs_threshold)

    # Find super expert indices
    super_experts = []
    for layer_idx, layer_data in observer_data.items():
        max_acts = layer_data["max_activations"]
        super_mask = max_acts > final_threshold
        super_indices = torch.where(super_mask)[0]
        for expert_idx in super_indices:
            super_experts.append([layer_idx, expert_idx.item()])

    logger.info(f"Found {len(super_experts)} super experts with threshold {final_threshold:.4f}")
    return torch.tensor(super_experts) if super_experts else torch.empty((0, 2), dtype=torch.long)


# ============================================================================
# Pruning
# ============================================================================

def prune_vlm_model(
    model: nn.Module,
    observer_data: dict,
    model_attrs: dict,
    n_experts_to_prune: int,
    prune_method: str = "reap",
    preserve_super_experts: bool = True,
) -> nn.Module:
    """
    Prune experts from a VLM model based on REAP criterion.
    """
    logger.info(f"Pruning {n_experts_to_prune} experts per layer using {prune_method}")

    # Get super experts if needed
    super_expert_idx = None
    if preserve_super_experts:
        super_expert_idx = get_super_expert_indices(observer_data)

        # Set super expert saliency to infinity (never prune)
        for layer_idx in observer_data:
            super_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer_idx][:, 1]
            if len(super_in_layer) > 0:
                for metric in ["reap", "ean_mean", "weighted_ean_sum", "expert_frequency"]:
                    if metric in observer_data[layer_idx]:
                        data = observer_data[layer_idx][metric]
                        if isinstance(data, torch.Tensor):
                            observer_data[layer_idx][metric][super_in_layer] = float("inf")

    # Get layers path
    layers_path = model_attrs.get("layers_path", "model.language_model.layers")
    layers = model
    for attr in layers_path.split("."):
        layers = getattr(layers, attr)

    # Prune each layer
    retained_experts_count = None
    for layer_idx in tqdm(observer_data, desc="Pruning layers"):
        moe = getattr(layers[layer_idx], model_attrs["moe_block"])
        num_experts = len(getattr(moe, model_attrs["experts"]))

        # Get saliency scores
        if prune_method == "reap":
            saliency = observer_data[layer_idx]["reap"]
        elif prune_method == "frequency":
            saliency = observer_data[layer_idx]["expert_frequency"].float()
        elif prune_method == "ean_mean":
            saliency = observer_data[layer_idx]["ean_mean"]
        elif prune_method == "weighted_ean_sum":
            saliency = observer_data[layer_idx]["weighted_ean_sum"]
        else:
            raise ValueError(f"Unknown prune method: {prune_method}")

        # Select experts to prune (lowest saliency)
        _, experts_to_prune = torch.topk(saliency, n_experts_to_prune, largest=False)
        retained_indices = [i for i in range(num_experts) if i not in experts_to_prune]
        retained_experts_count = len(retained_indices)

        # Prune experts
        experts = getattr(moe, model_attrs["experts"])
        retained_experts = torch.nn.ModuleList([experts[i] for i in retained_indices])
        setattr(moe, model_attrs["experts"], retained_experts)

        # Prune router
        router = getattr(moe, model_attrs["router"])
        router.weight.data = router.weight.data[retained_indices, :]
        if hasattr(router, "bias") and router.bias is not None:
            router.bias.data = router.bias.data[retained_indices]
        router.out_features = len(retained_indices)

        logger.debug(f"Layer {layer_idx}: {num_experts} -> {len(retained_indices)} experts")

    # Update model config
    if hasattr(model.config, "text_config"):
        model.config.text_config.num_experts = retained_experts_count

    logger.info(f"Pruning complete: {retained_experts_count} experts per layer retained")
    return model


# ============================================================================
# Main
# ============================================================================

def main():
    parser = HfArgumentParser((VLMReapArgs,))
    args, = parser.parse_args_into_dataclasses()

    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("VLM REAP - Vision Language Model Expert Pruning")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Calibration data: {args.calibration_data_path}")
    logger.info(f"Compression ratio: {args.compression_ratio}")
    logger.info(f"Prune method: {args.prune_method}")

    # Load processor
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )

    # Load model - try different auto classes for compatibility
    logger.info("Loading model (this may take a while for large models)...")
    model = None

    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU memory: {gpu_mem:.1f} GB")

    # Try to load with different strategies
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "attn_implementation": "sdpa",  # Use SDPA instead of flash attention
        "low_cpu_mem_usage": True,
    }

    # For AWQ models, we need to keep everything on GPU
    # If model doesn't fit, we'll need to use a different approach
    try:
        # First try: GPU only (required for AWQ)
        logger.info("Attempting to load model on GPU only (required for AWQ)...")
        model = AutoModel.from_pretrained(
            args.model_path,
            device_map="cuda:0",
            **load_kwargs,
        )
        logger.info("Loaded model with AutoModel on GPU")
    except Exception as e:
        logger.warning(f"GPU-only loading failed: {e}")

        # Second try: Use sequential loading with manual device management
        try:
            logger.info("Attempting sequential layer loading...")
            model = AutoModel.from_pretrained(
                args.model_path,
                device_map="sequential",
                max_memory={0: "90GiB"},  # Leave some headroom
                **load_kwargs,
            )
            logger.info("Loaded model with sequential device map")
        except Exception as e2:
            logger.warning(f"Sequential loading failed: {e2}")

            # Third try: Load without quantization config for inspection
            try:
                logger.info("Attempting to load without device_map restrictions...")
                # Try using AutoModelForVision2Seq
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                    args.model_path,
                    device_map="cuda:0",
                    **load_kwargs,
                )
                logger.info("Loaded model with AutoModelForImageTextToText")
            except Exception as e3:
                logger.error(f"All loading methods failed: {e3}")
                raise RuntimeError(
                    f"Could not load model from {args.model_path}. "
                    f"AWQ models require full GPU memory. Your GPU has {gpu_mem:.1f}GB "
                    f"but model may need more. Consider using a smaller model or "
                    f"a non-AWQ version with CPU offloading support."
                )

    model.eval()

    # Get model class and attributes
    model_class = model.__class__.__name__
    logger.info(f"Model class: {model_class}")

    if model_class not in VLM_MODEL_ATTRS:
        raise ValueError(
            f"Model {model_class} not supported. "
            f"Supported: {list(VLM_MODEL_ATTRS.keys())}"
        )

    model_attrs = VLM_MODEL_ATTRS[model_class]

    # Load calibration dataset
    logger.info("Loading calibration dataset...")
    dataset = VLMCalibrationDataset(
        args.calibration_data_path,
        processor,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        num_samples=args.num_samples,
    )

    # Create observer
    logger.info("Creating MoE observer...")
    observer = VLMMoEObserver(
        model,
        model_attrs,
        renormalize_router_weights=args.renormalize_router_weights,
    )

    # Run calibration
    logger.info(f"Running calibration on {len(dataset)} samples...")
    device = next(model.parameters()).device

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Calibrating"):
            sample = dataset[idx]
            try:
                inputs = dataset.prepare_input(sample, device)
                _ = model(**inputs)
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue

            # Clear cache periodically
            if idx % 50 == 0:
                torch.cuda.empty_cache()

    # Get observer data
    observer_data = observer.get_state()
    observer.close()

    # Save observer data if requested
    output_dir = pathlib.Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_observer_data:
        observer_path = output_dir / "observer_data.pt"
        torch.save(observer_data, observer_path)
        logger.info(f"Observer data saved to {observer_path}")

    # Calculate number of experts to prune
    num_experts = observer_data[0]["expert_frequency"].shape[0]
    n_experts_to_prune = int(num_experts * args.compression_ratio)
    logger.info(f"Pruning {n_experts_to_prune} of {num_experts} experts ({args.compression_ratio*100:.0f}%)")

    # Prune model
    logger.info("Pruning model...")
    start_time = time.time()
    model = prune_vlm_model(
        model,
        observer_data,
        model_attrs,
        n_experts_to_prune,
        prune_method=args.prune_method,
        preserve_super_experts=args.preserve_super_experts,
    )
    prune_time = time.time() - start_time
    logger.info(f"Pruning completed in {prune_time:.2f} seconds")

    # Save pruned model
    logger.info(f"Saving pruned model to {output_dir}...")
    start_time = time.time()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    save_time = time.time() - start_time
    logger.info(f"Model saved in {save_time:.2f} seconds")

    # Save config
    config_info = {
        "original_model": args.model_path,
        "compression_ratio": args.compression_ratio,
        "prune_method": args.prune_method,
        "original_experts": num_experts,
        "retained_experts": num_experts - n_experts_to_prune,
        "num_calibration_samples": len(dataset),
        "preserve_super_experts": args.preserve_super_experts,
    }
    with open(output_dir / "reap_config.json", "w") as f:
        json.dump(config_info, f, indent=2)

    logger.info("=" * 60)
    logger.info("VLM REAP Complete!")
    logger.info(f"Original experts: {num_experts}")
    logger.info(f"Retained experts: {num_experts - n_experts_to_prune}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
