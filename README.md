# REAP V10 - Expert Pruning for Qwen3-VL-235B

CerebrasResearch/reap uyumlu expert pruning sistemi. H200 GPU (140GB VRAM) icin optimize edilmistir.

## Hizli Baslangic

```bash
# 1. Docker image olustur
./run.sh build

# 2. Interaktif shell'e gir
./run.sh shell

# 3. REAP scoring calistir
python reap.py

# 4. Model'i prune et
python prune_model.py --verify
```

## Docker Compose Kullanimi

```bash
# Image olustur
docker compose build

# Interaktif shell
docker compose run --rm reap bash

# REAP calistir
docker compose run --rm reap python reap.py

# Prune calistir
docker compose run --rm reap python prune_model.py --verify
```

## Dizin Yapisi

```
/app/
├── reap.py              # REAP scoring
├── prune_model.py       # Model pruning
├── test_reapv8_docker.py # Testing
├── data/                # Mount: /mnt/vault/boundingboxtest
│   ├── models/          # Model dosyalari
│   └── calibration/     # Kalibrasyon verileri
├── output/              # Cikti dizini
├── cache/               # HuggingFace cache
└── logs/                # Log dosyalari
```

## Konfigurasyion

### Environment Variables

```bash
# Model yollari
REAP_MODEL_PATH=/app/data/models/Qwen3-VL-235B-A22B-Thinking-AWQ
REAP_OUTPUT_PATH=/app/output/reap-v10
REAP_CALIBRATION_PATH=/app/data/calibration/calibration_data.json

# Pruning ayarlari
REAP_PRUNE_RATIO=0.40      # %40 prune
REAP_MAX_SAMPLES=300       # Kalibrasyon ornekleri
REAP_PARALLEL_LAYERS=4     # Paralel layer sayisi
REAP_BATCH_SIZE=4          # Batch boyutu

# H200 VRAM
REAP_VRAM_GB=140.0         # Toplam VRAM
REAP_VRAM_BUFFER=0.10      # %10 buffer
```

### .env Dosyasi

```bash
cp .env.example .env
nano .env
```

## REAP Metodolojisi

CerebrasResearch/reap ile uyumlu REAP score hesaplama:

```
REAP Score = mean(gate_weight * ||expert_output||_L2)
```

- **Welford's Algorithm**: Numerically stable online mean
- **Kahan Summation**: Precision loss prevention
- **Parallel Processing**: 4 layer ayni anda

## GPU Gereksinimleri

| GPU | VRAM | Parallel Layers | Batch Size |
|-----|------|-----------------|------------|
| H200 | 140GB | 4 | 4 |
| H100 | 80GB | 2 | 2 |
| A100 | 80GB | 2 | 2 |
| A100 | 40GB | 1 | 1 |

## Cikti

### REAP Metadata

```json
{
  "reap_version": "v10_cerebras_compatible",
  "methodology": "REAP = mean(gate_weight * ||expert_output||_L2)",
  "experts_to_keep": {"0": [1,5,8,...], "1": [2,6,9,...], ...},
  "mean_scores": [[...], [...], ...]
}
```

### Pruned Model

```
output/
├── Qwen3-VL-235B-Pruned/
│   ├── config.json           # Updated config (num_experts: 77)
│   ├── model-*.safetensors   # Pruned weights
│   ├── reap_metadata.json    # REAP info
│   └── tokenizer files...
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Parallel layer sayisini azalt
export REAP_PARALLEL_LAYERS=2
export REAP_BATCH_SIZE=2
```

### Slow Performance

```bash
# Flash Attention aktif mi kontrol et
python -c "from flash_attn import flash_attn_func; print('OK')"
```

### Model Yukleme Hatasi

```bash
# Model dosyalarini kontrol et
ls -la /app/data/models/Qwen3-VL-235B-A22B-Thinking-AWQ/
```

## Lisans

MIT License
