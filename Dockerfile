# =============================================================================
# REAP V11 - VLM Expert Pruning Docker Image
# =============================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL maintainer="REAP V11"
LABEL description="Fixed Expert Pruning for Qwen3-VL"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libssl-dev libffi-dev libjpeg-dev libpng-dev \
    htop nvtop tmux vim nano \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch
RUN pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ML deps
RUN pip install \
    transformers>=4.45.0 \
    accelerate>=0.34.0 \
    safetensors>=0.4.5 \
    huggingface-hub>=0.25.0 \
    numpy>=1.26.0 \
    Pillow>=10.4.0 \
    tqdm>=4.66.0 \
    pyyaml>=6.0.0 \
    python-dotenv>=1.0.0

# Flash Attention
RUN pip install flash-attn==2.6.3 --no-build-isolation

# Qwen-VL
RUN pip install qwen-vl-utils>=0.0.8

# vLLM for testing
RUN pip install vllm>=0.6.0

WORKDIR /app

COPY reap.py /app/
COPY prune_model.py /app/
COPY test_pruned_model.py /app/
COPY entrypoint.sh /app/

RUN mkdir -p /app/data /app/output /app/cache \
    && chmod +x /app/entrypoint.sh

ENV HF_HOME=/app/cache/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]
