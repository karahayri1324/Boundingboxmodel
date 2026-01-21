#!/bin/bash
# REAP Setup Script for RunPod H200
# Installs all dependencies needed for VLM REAP pruning

set -e

echo "============================================================"
echo "REAP Setup for RunPod H200"
echo "============================================================"

# Update system
echo "Updating system packages..."
apt-get update -qq

# Install system dependencies
echo "Installing system dependencies..."
apt-get install -y -qq git curl wget htop nvtop

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CUDA 12.1 for H200)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "Installing core dependencies..."
pip install transformers>=4.45.0
pip install accelerate>=0.26.0
pip install huggingface_hub[cli]
pip install safetensors
pip install sentencepiece
pip install protobuf

# Install VLM dependencies
echo "Installing VLM dependencies..."
pip install pillow
pip install qwen-vl-utils

# Install AWQ support
echo "Installing AWQ support..."
pip install autoawq

# Install additional utilities
echo "Installing additional utilities..."
pip install tqdm
pip install pyyaml
pip install scipy
pip install scikit-learn

# Install flash-attention (optional but recommended)
echo "Installing flash-attention..."
pip install flash-attn --no-build-isolation || echo "Flash-attention installation failed (optional)"

# Verify installations
echo ""
echo "============================================================"
echo "Verifying installations..."
echo "============================================================"

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"

# Check GPU memory
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Download model:  bash download_model.sh"
echo "  2. Run REAP:        bash run_reap_h200.sh"
echo ""
