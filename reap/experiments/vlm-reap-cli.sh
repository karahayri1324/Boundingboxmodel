#!/bin/bash
# VLM REAP - Vision Language Model Expert Pruning
# Usage: bash experiments/vlm-reap-cli.sh [CUDA_DEVICES] [MODEL_PATH] [CALIBRATION_PATH] [OUTPUT_PATH] [COMPRESSION_RATIO] [NUM_SAMPLES]

set -e

# Arguments
CUDA_DEVICES="${1:-0}"
MODEL_PATH="${2:-/mnt/vault/Boundingboxmodel/Qwen3-VL-235B-A22B-Thinking-AWQ}"
CALIBRATION_PATH="${3:-/mnt/vault/Boundingboxmodel/reap_calibration_data}"
OUTPUT_PATH="${4:-./artifacts/Qwen3-VL-235B-REAP}"
COMPRESSION_RATIO="${5:-0.5}"
NUM_SAMPLES="${6:-}"  # Empty = use all samples
PRUNE_METHOD="${7:-reap}"

echo "============================================================"
echo "VLM REAP - Vision Language Model Expert Pruning"
echo "============================================================"
echo "CUDA Devices:      $CUDA_DEVICES"
echo "Model Path:        $MODEL_PATH"
echo "Calibration Path:  $CALIBRATION_PATH"
echo "Output Path:       $OUTPUT_PATH"
echo "Compression Ratio: $COMPRESSION_RATIO"
echo "Num Samples:       ${NUM_SAMPLES:-all}"
echo "Prune Method:      $PRUNE_METHOD"
echo "============================================================"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Memory optimizations for large models
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Build command
CMD="python -m reap.vlm_reap \
    --model_path $MODEL_PATH \
    --calibration_data_path $CALIBRATION_PATH \
    --output_path $OUTPUT_PATH \
    --compression_ratio $COMPRESSION_RATIO \
    --prune_method $PRUNE_METHOD \
    --preserve_super_experts true \
    --renormalize_router_weights true \
    --save_observer_data true"

# Add num_samples if specified
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

echo ""
echo "Running command:"
echo "$CMD"
echo ""

# Run
cd /mnt/vault/Boundingboxreap/reap
$CMD

echo ""
echo "============================================================"
echo "VLM REAP Complete!"
echo "Output saved to: $OUTPUT_PATH"
echo "============================================================"
