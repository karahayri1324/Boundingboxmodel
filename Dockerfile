# REAP VLM Expert Pruning - Docker Image
# For RunPod H200 / High VRAM GPUs
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install transformers and core deps
RUN pip install \
    transformers>=4.45.0 \
    accelerate>=0.26.0 \
    huggingface_hub[cli] \
    safetensors \
    sentencepiece \
    protobuf \
    pillow \
    tqdm \
    pyyaml \
    scipy \
    scikit-learn

# Install qwen-vl-utils for Qwen3-VL
RUN pip install qwen-vl-utils

# Install AutoAWQ for quantized models
RUN pip install autoawq

# Install flash-attention (optional, may fail on some systems)
RUN pip install flash-attn --no-build-isolation || echo "Flash-attn not installed"

# Create working directory
WORKDIR /workspace

# Copy REAP source code
COPY reap/ /workspace/reap/

# Copy calibration data
COPY reap_calibration_data/ /workspace/reap_calibration_data/

# Copy scripts
COPY run_reap_h200.sh /workspace/
COPY download_model.sh /workspace/

# Make scripts executable
RUN chmod +x /workspace/*.sh

# Set PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/workspace/reap/src"

# Set memory optimizations
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
ENV TOKENIZERS_PARALLELISM=false

# Default command
CMD ["/bin/bash"]
