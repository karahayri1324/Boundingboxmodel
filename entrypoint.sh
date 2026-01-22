#!/bin/bash
# =============================================================================
# REAP V10 Docker Entrypoint
# =============================================================================

set -e

echo "=============================================="
echo "  REAP V10 - Expert Pruning Environment"
echo "  H200 GPU Optimized"
echo "=============================================="

# GPU Check
echo ""
echo "[GPU Check]"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""

    # Get GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "  GPU Memory: ${GPU_MEM} MB"

    # Check if H200
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    if [[ "$GPU_NAME" == *"H200"* ]] || [[ "$GPU_NAME" == *"H100"* ]]; then
        echo "  Detected: High-end GPU - Full optimizations enabled"
        export REAP_PARALLEL_LAYERS=4
        export REAP_BATCH_SIZE=4
    else
        echo "  Detected: Standard GPU - Conservative settings"
        export REAP_PARALLEL_LAYERS=2
        export REAP_BATCH_SIZE=2
    fi
else
    echo "  WARNING: nvidia-smi not found!"
fi

# Python Check
echo ""
echo "[Python Environment]"
python --version
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  CUDA Version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "  GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "  GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# Directory Check
echo ""
echo "[Directories]"
echo "  Working: $(pwd)"
echo "  Data: /app/data"
echo "  Output: /app/output"
echo "  Cache: /app/cache"

# Check mounted volumes
if [ -d "/app/data/models" ]; then
    echo "  Models: /app/data/models (mounted)"
    ls -la /app/data/models 2>/dev/null | head -5
fi

if [ -d "/app/data/calibration" ]; then
    echo "  Calibration: /app/data/calibration (mounted)"
fi

# Available commands
echo ""
echo "=============================================="
echo "  Available Commands"
echo "=============================================="
echo ""
echo "  # Run REAP scoring"
echo "  python reap.py"
echo ""
echo "  # Prune model after scoring"
echo "  python prune_model.py --verify"
echo ""
echo "  # Test pruned model"
echo "  python test_reapv8_docker.py"
echo ""
echo "  # Monitor GPU"
echo "  watch -n 1 nvidia-smi"
echo "  gpustat -i 1"
echo ""
echo "  # Check VRAM usage"
echo "  python -c \"import torch; print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB')\""
echo ""
echo "=============================================="

# Execute command
exec "$@"
