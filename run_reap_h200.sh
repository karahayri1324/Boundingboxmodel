#!/bin/bash
# Qwen3-VL-235B REAP Pruning Script for H200
# NVIDIA H200 (141GB VRAM) - Optimized

set -e

echo "============================================================"
echo "Qwen3-VL-235B-A22B REAP Expert Pruning"
echo "============================================================"
echo "GPU: NVIDIA H200 (141GB VRAM)"
echo "Model: Qwen3-VL-235B-A22B-Thinking-AWQ (~125GB)"
echo "============================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Paths - Relative to script directory
MODEL_PATH="${MODEL_PATH:-./models/Qwen3-VL-235B-A22B-Thinking-AWQ}"
CALIBRATION_PATH="${CALIBRATION_PATH:-./reap_calibration_data}"
OUTPUT_PATH="${OUTPUT_PATH:-./output/Qwen3-VL-235B-A22B-REAP-50}"

# Configuration
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"  # Prune 50% of experts (128 -> 64)
PRUNE_METHOD="${PRUNE_METHOD:-reap}"           # REAP criterion
NUM_SAMPLES="${NUM_SAMPLES:-}"                 # Empty = use all samples

# Memory optimizations for H200
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Add REAP src to path
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/reap/src"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo ""
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please run: bash download_model.sh"
    echo ""
    exit 1
fi

# Check if calibration data exists
if [ ! -f "$CALIBRATION_PATH/calibration_data.json" ]; then
    echo ""
    echo "ERROR: Calibration data not found at $CALIBRATION_PATH"
    echo ""
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

echo ""
echo "Configuration:"
echo "  Model:       $MODEL_PATH"
echo "  Calibration: $CALIBRATION_PATH"
echo "  Output:      $OUTPUT_PATH"
echo "  Compression: ${COMPRESSION_RATIO} (128 -> 64 experts)"
echo "  Method:      ${PRUNE_METHOD}"
echo "  Samples:     ${NUM_SAMPLES:-all}"
echo ""

# Show GPU info
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Build command
CMD="python -m reap.vlm_reap \
    --model_path \"${MODEL_PATH}\" \
    --calibration_data_path \"${CALIBRATION_PATH}\" \
    --output_path \"${OUTPUT_PATH}\" \
    --compression_ratio ${COMPRESSION_RATIO} \
    --prune_method ${PRUNE_METHOD} \
    --preserve_super_experts true \
    --renormalize_router_weights true \
    --save_observer_data true"

# Add num_samples if specified
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

echo "Running REAP..."
echo ""

# Run
eval $CMD

echo ""
echo "============================================================"
echo "REAP Complete!"
echo "============================================================"
echo "Pruned model saved to: ${OUTPUT_PATH}"
echo ""
echo "Expected results:"
echo "  Original: 128 experts/layer, ~125GB (AWQ)"
echo "  Pruned:   64 experts/layer,  ~65GB (AWQ)"
echo "============================================================"
