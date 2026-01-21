#!/bin/bash
# REAP VLM - Tek Komutla Çalıştır
# RunPod H200 için optimize edilmiş

set -e

echo "============================================================"
echo "REAP VLM Expert Pruning - Otomatik Kurulum"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

# GPU kontrolü
echo ""
echo "GPU Kontrolü..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    print_status "GPU bulundu"
else
    print_error "nvidia-smi bulunamadı!"
    exit 1
fi

# Docker kontrolü
echo ""
echo "Docker Kontrolü..."
if command -v docker &> /dev/null; then
    print_status "Docker bulundu: $(docker --version)"
else
    print_warning "Docker bulunamadı, direkt kurulum yapılacak..."

    # Docker yoksa direkt pip ile kur
    echo ""
    echo "Bağımlılıklar kuruluyor..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers>=4.45.0 accelerate>=0.26.0 huggingface_hub[cli]
    pip install safetensors sentencepiece protobuf pillow tqdm pyyaml scipy scikit-learn
    pip install qwen-vl-utils autoawq
    pip install flash-attn --no-build-isolation 2>/dev/null || true

    print_status "Bağımlılıklar kuruldu"

    # PYTHONPATH ayarla
    export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/reap/src"
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
    export TOKENIZERS_PARALLELISM=false

    # Model kontrolü
    MODEL_PATH="./models/Qwen3-VL-235B-A22B-Thinking-AWQ"
    if [ ! -d "$MODEL_PATH" ]; then
        echo ""
        print_warning "Model bulunamadı, indiriliyor..."
        mkdir -p ./models

        if [ -z "$HF_TOKEN" ]; then
            print_error "HF_TOKEN ayarlanmamış!"
            echo "Şunu çalıştır: export HF_TOKEN=hf_xxxxx"
            exit 1
        fi

        huggingface-cli download "Quyen/Qwen3-VL-235B-A22B-Thinking-AWQ" \
            --local-dir "$MODEL_PATH" \
            --local-dir-use-symlinks False \
            --resume-download

        print_status "Model indirildi"
    else
        print_status "Model mevcut: $MODEL_PATH"
    fi

    # REAP çalıştır
    echo ""
    echo "============================================================"
    echo "REAP Başlatılıyor..."
    echo "============================================================"

    mkdir -p ./output

    python -m reap.vlm_reap \
        --model_path "$MODEL_PATH" \
        --calibration_data_path "./reap_calibration_data" \
        --output_path "./output/Qwen3-VL-235B-A22B-REAP-50" \
        --compression_ratio 0.5 \
        --prune_method reap \
        --preserve_super_experts true \
        --renormalize_router_weights true \
        --save_observer_data true

    echo ""
    print_status "REAP tamamlandı!"
    echo "Çıktı: ./output/Qwen3-VL-235B-A22B-REAP-50"
    exit 0
fi

# Docker varsa container ile çalıştır
echo ""
echo "Docker Image Oluşturuluyor..."
docker build -t reap-vlm:latest .
print_status "Docker image oluşturuldu"

# Dizinleri oluştur
mkdir -p ./models ./output

# HF_TOKEN kontrolü
if [ -z "$HF_TOKEN" ]; then
    print_warning "HF_TOKEN ayarlanmamış!"
    echo "Model indirmek için: export HF_TOKEN=hf_xxxxx"
fi

# Container çalıştır
echo ""
echo "============================================================"
echo "Container Başlatılıyor..."
echo "============================================================"

docker run --rm -it \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024 \
    -e TOKENIZERS_PARALLELISM=false \
    -e HF_TOKEN="${HF_TOKEN}" \
    -v "${SCRIPT_DIR}/models:/workspace/models" \
    -v "${SCRIPT_DIR}/output:/workspace/output" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    reap-vlm:latest \
    bash -c '
        echo "Container içinde..."

        # Model kontrolü
        MODEL_PATH="/workspace/models/Qwen3-VL-235B-A22B-Thinking-AWQ"

        if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
            echo "Model indiriliyor..."
            mkdir -p /workspace/models

            if [ -z "$HF_TOKEN" ]; then
                echo "HATA: HF_TOKEN ayarlanmamış!"
                echo "Container dışında çalıştır: export HF_TOKEN=hf_xxxxx"
                exit 1
            fi

            huggingface-cli login --token "$HF_TOKEN"
            huggingface-cli download "Quyen/Qwen3-VL-235B-A22B-Thinking-AWQ" \
                --local-dir "$MODEL_PATH" \
                --local-dir-use-symlinks False \
                --resume-download
        else
            echo "Model mevcut: $MODEL_PATH"
        fi

        echo ""
        echo "============================================================"
        echo "REAP Başlatılıyor..."
        echo "============================================================"

        python -m reap.vlm_reap \
            --model_path "$MODEL_PATH" \
            --calibration_data_path "/workspace/reap_calibration_data" \
            --output_path "/workspace/output/Qwen3-VL-235B-A22B-REAP-50" \
            --compression_ratio 0.5 \
            --prune_method reap \
            --preserve_super_experts true \
            --renormalize_router_weights true \
            --save_observer_data true

        echo ""
        echo "============================================================"
        echo "REAP Tamamlandı!"
        echo "Çıktı: /workspace/output/Qwen3-VL-235B-A22B-REAP-50"
        echo "============================================================"
    '

print_status "İşlem tamamlandı!"
echo "Pruned model: ./output/Qwen3-VL-235B-A22B-REAP-50"
