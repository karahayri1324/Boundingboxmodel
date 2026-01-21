#!/bin/bash
# RunPod H200 - Tek Komut ile REAP
# Bu script RunPod ortamında direkt çalışır

set -e

echo "============================================================"
echo "  REAP VLM Expert Pruning - RunPod H200"
echo "  Qwen3-VL-235B -> 50% Pruned"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# GPU bilgisi
echo ""
echo "[1/5] GPU Kontrolü..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Python paketleri
echo "[2/5] Paketler Kuruluyor..."
pip install --quiet --upgrade pip

# PyTorch (CUDA 12.1 - RunPod default)
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers & deps
pip install --quiet \
    "transformers>=4.45.0" \
    "accelerate>=0.26.0" \
    "huggingface_hub[cli]" \
    safetensors \
    sentencepiece \
    protobuf \
    pillow \
    tqdm \
    pyyaml \
    scipy \
    scikit-learn

# Qwen-VL utils
pip install --quiet qwen-vl-utils

# AutoAWQ
pip install --quiet autoawq

# Flash attention (opsiyonel)
pip install --quiet flash-attn --no-build-isolation 2>/dev/null || echo "Flash-attn skip"

echo "Paketler tamam!"
echo ""

# Environment
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/reap/src"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# Model path
MODEL_DIR="${SCRIPT_DIR}/models"
MODEL_PATH="${MODEL_DIR}/Qwen3-VL-235B-A22B-Thinking-AWQ"
OUTPUT_PATH="${SCRIPT_DIR}/output/Qwen3-VL-235B-A22B-REAP-50"

mkdir -p "$MODEL_DIR" "$SCRIPT_DIR/output"

# Model indirme
echo "[3/5] Model Kontrolü..."
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Model indiriliyor... (~125GB, uzun sürebilir)"

    if [ -z "$HF_TOKEN" ]; then
        echo ""
        echo "!!! HF_TOKEN gerekli !!!"
        echo "Şunu çalıştır ve tekrar dene:"
        echo "  export HF_TOKEN=hf_xxxxxxxxxxxxx"
        echo "  bash runpod_start.sh"
        echo ""
        exit 1
    fi

    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

    huggingface-cli download "Quyen/Qwen3-VL-235B-A22B-Thinking-AWQ" \
        --local-dir "$MODEL_PATH" \
        --local-dir-use-symlinks False \
        --resume-download

    echo "Model indirildi!"
else
    echo "Model mevcut: $MODEL_PATH"
fi
echo ""

# Kalibrasyon verisi kontrolü
echo "[4/5] Kalibrasyon Verisi Kontrolü..."
CALIB_PATH="${SCRIPT_DIR}/reap_calibration_data"
if [ ! -f "${CALIB_PATH}/calibration_data.json" ]; then
    echo "HATA: Kalibrasyon verisi bulunamadı!"
    echo "Beklenen: ${CALIB_PATH}/calibration_data.json"
    exit 1
fi
SAMPLE_COUNT=$(python -c "import json; print(len(json.load(open('${CALIB_PATH}/calibration_data.json'))))")
echo "Kalibrasyon: $SAMPLE_COUNT görsel"
echo ""

# REAP çalıştır
echo "[5/5] REAP Başlatılıyor..."
echo "============================================================"
echo "  Model:       $MODEL_PATH"
echo "  Calibration: $CALIB_PATH"
echo "  Output:      $OUTPUT_PATH"
echo "  Compression: 0.5 (128 -> 64 experts)"
echo "============================================================"
echo ""

python -m reap.vlm_reap \
    --model_path "$MODEL_PATH" \
    --calibration_data_path "$CALIB_PATH" \
    --output_path "$OUTPUT_PATH" \
    --compression_ratio 0.5 \
    --prune_method reap \
    --preserve_super_experts true \
    --renormalize_router_weights true \
    --save_observer_data true

echo ""
echo "============================================================"
echo "  TAMAMLANDI!"
echo "============================================================"
echo "  Pruned model: $OUTPUT_PATH"
echo "  Original:     128 experts (~125GB)"
echo "  Pruned:       64 experts (~65GB)"
echo "============================================================"
