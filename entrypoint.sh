#!/bin/bash
# =============================================================================
# REAP V11 Docker Entrypoint
# =============================================================================

set -e

echo "=============================================="
echo "  REAP V11 - Fixed VLM Expert Pruning"
echo "=============================================="

# GPU Check
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "[GPU]"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Python Check
echo ""
echo "[Python]"
python --version
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Config
echo ""
echo "[Config]"
echo "REAP_SCORING_MODE: ${REAP_SCORING_MODE:-full}"
echo "REAP_PRUNE_RATIO: ${REAP_PRUNE_RATIO:-0.40}"

echo ""
echo "=============================================="
echo "Commands:"
echo "  python reap.py           # REAP scoring"
echo "  python prune_model.py    # Prune model"
echo "  python test_pruned_model.py  # Test"
echo "=============================================="

exec "$@"
