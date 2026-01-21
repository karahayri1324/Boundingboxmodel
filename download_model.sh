#!/bin/bash
# Qwen3-VL-235B-A22B-Thinking-AWQ Model Download Script
# For RunPod H200 environment

set -e

echo "============================================================"
echo "Qwen3-VL-235B-A22B-Thinking-AWQ Model Downloader"
echo "============================================================"

# Configuration
MODEL_ID="tclf90/Qwen3-VL-235B-A22B-Thinking-AWQ"
MODEL_DIR="./models/Qwen3-VL-235B-A22B-Thinking-AWQ"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install -U huggingface_hub[cli]
fi

# Login to HuggingFace if token provided
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "WARNING: HF_TOKEN not set. If model is gated, set it with:"
    echo "  export HF_TOKEN=your_token_here"
    echo ""
fi

# Create model directory
mkdir -p "$MODEL_DIR"

echo ""
echo "Downloading model: $MODEL_ID"
echo "Target directory: $MODEL_DIR"
echo "This may take a while (~125GB)..."
echo ""

# Download model
huggingface-cli download "$MODEL_ID" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False \
    --resume-download

echo ""
echo "============================================================"
echo "Download Complete!"
echo "Model saved to: $MODEL_DIR"
echo "============================================================"

# Verify download
echo ""
echo "Verifying download..."
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "OK - config.json found"
else
    echo "WARNING: config.json not found!"
fi

SAFETENSOR_COUNT=$(ls -1 "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)
echo "Found $SAFETENSOR_COUNT safetensor files"

echo ""
echo "Done! You can now run REAP with:"
echo "  bash run_reap_h200.sh"
