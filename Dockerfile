# =============================================================================
# REAP V10 - Qwen3-VL Expert Pruning Docker Image
# =============================================================================
# H200 GPU Optimized (140GB VRAM)
# CerebrasResearch/reap Compatible
# =============================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL maintainer="REAP V10"
LABEL description="Expert Pruning for Qwen3-VL-235B-AWQ"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# CUDA Environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Python Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# =============================================================================
# System Dependencies
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    # Python
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # System libraries
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libwebp-dev \
    zlib1g-dev \
    # Utilities
    htop \
    nvtop \
    tmux \
    vim \
    nano \
    tree \
    jq \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# =============================================================================
# PyTorch + CUDA 12.4
# =============================================================================
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# =============================================================================
# Core ML Dependencies
# =============================================================================
RUN pip install \
    # Transformers ecosystem
    transformers>=4.45.0 \
    accelerate>=0.34.0 \
    datasets>=2.20.0 \
    tokenizers>=0.19.0 \
    sentencepiece>=0.2.0 \
    # Model loading
    safetensors>=0.4.5 \
    huggingface-hub>=0.25.0 \
    # Scientific computing
    numpy>=1.26.0 \
    scipy>=1.13.0 \
    pandas>=2.2.0 \
    scikit-learn>=1.5.0 \
    # Image processing
    Pillow>=10.4.0 \
    opencv-python-headless>=4.10.0 \
    # Visualization
    matplotlib>=3.9.0 \
    seaborn>=0.13.0 \
    # Progress & logging
    tqdm>=4.66.0 \
    rich>=13.7.0 \
    colorama>=0.4.6 \
    # Utilities
    pyyaml>=6.0.0 \
    python-dotenv>=1.0.0 \
    requests>=2.32.0

# =============================================================================
# Flash Attention 2 (H200 optimized)
# =============================================================================
RUN pip install flash-attn==2.6.3 --no-build-isolation

# =============================================================================
# vLLM for Testing (Optional but recommended)
# =============================================================================
RUN pip install vllm>=0.6.0

# =============================================================================
# Qwen3-VL specific dependencies
# =============================================================================
RUN pip install \
    qwen-vl-utils>=0.0.8 \
    torchvision \
    av>=12.0.0

# =============================================================================
# Additional tools for monitoring
# =============================================================================
RUN pip install \
    nvidia-ml-py>=12.560.0 \
    gpustat>=1.1.0 \
    psutil>=6.0.0 \
    memory-profiler>=0.61.0

# =============================================================================
# Working Directory
# =============================================================================
WORKDIR /app

# =============================================================================
# Copy REAP scripts
# =============================================================================
COPY reap.py /app/reap.py
COPY prune_model.py /app/prune_model.py
COPY test_reapv8_docker.py /app/test_reapv8_docker.py

# =============================================================================
# Create directories
# =============================================================================
RUN mkdir -p /app/data \
    && mkdir -p /app/output \
    && mkdir -p /app/logs \
    && mkdir -p /app/cache

# =============================================================================
# Environment for HuggingFace
# =============================================================================
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV HF_DATASETS_CACHE=/app/cache/datasets

# =============================================================================
# CUDA Memory Settings for H200
# =============================================================================
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
ENV CUDA_VISIBLE_DEVICES=0

# =============================================================================
# Entrypoint script
# =============================================================================
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]
