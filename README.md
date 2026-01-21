# Qwen3-VL-235B REAP Expert Pruning

Bu repo, Qwen3-VL-235B-A22B modelini REAP (Representation Efficient Activation Pruning) yontemiyle prune etmek icin gerekli tum dosyalari icerir.

## Gereksinimler

- **GPU:** NVIDIA H200 (141GB VRAM) veya benzeri
- **RAM:** 64GB+ onerilir
- **Disk:** ~300GB (model + output)

## Hizli Baslangic (RunPod H200)

```bash
# 1. Repo'yu klonla
git clone https://github.com/KULLANICI_ADI/Boundingboxmodel.git
cd Boundingboxmodel

# 2. Bagimliliklari kur
bash setup.sh

# 3. Modeli indir (~125GB, uzun surebilir)
export HF_TOKEN=your_huggingface_token
bash download_model.sh

# 4. REAP calistir
bash run_reap_h200.sh
```

## Dosya Yapisi

```
Boundingboxmodel/
├── setup.sh                 # Bagimliliklari kurar
├── download_model.sh        # Modeli HuggingFace'den indirir
├── run_reap_h200.sh         # Ana REAP calistirma scripti
├── README.md                # Bu dosya
├── reap/                    # REAP kaynak kodu
│   └── src/reap/
│       ├── vlm_reap.py      # VLM REAP ana modulu
│       ├── observer.py      # MoE observer
│       ├── prune.py         # Pruning fonksiyonlari
│       ├── metrics.py       # Metrik hesaplamalari
│       └── model_util.py    # Model yardimcilari
├── reap_calibration_data/   # Kalibrasyon verisi
│   ├── calibration_data.json
│   └── images/              # Kalibrasyon gorselleri
├── models/                  # (Indirilecek) Model dosyalari
└── output/                  # (Olusturulacak) Pruned model
```

## Yapilandirma

`run_reap_h200.sh` icindeki degiskenler:

| Degisken | Varsayilan | Aciklama |
|----------|------------|----------|
| MODEL_PATH | ./models/Qwen3-VL-235B-A22B-Thinking-AWQ | Model yolu |
| CALIBRATION_PATH | ./reap_calibration_data | Kalibrasyon verisi |
| OUTPUT_PATH | ./output/Qwen3-VL-235B-A22B-REAP-50 | Cikti yolu |
| COMPRESSION_RATIO | 0.5 | Sikistrima orani (0.5 = %50 prune) |
| PRUNE_METHOD | reap | Pruning yontemi |
| NUM_SAMPLES | (bos = tumu) | Kalibrasyon ornegi sayisi |

## Pruning Yontemleri

| Yontem | Aciklama |
|--------|----------|
| `reap` | Router weight * Expert Activation Norm (varsayilan) |
| `frequency` | Expert secilme sikligi |
| `ean_mean` | Ortalama aktivasyon normu |
| `weighted_ean_sum` | Agirlikli aktivasyon toplami |

## Beklenen Sonuclar

- **Orijinal model:** 128 expert/katman, ~125GB (AWQ)
- **Pruned model:** 64 expert/katman, ~65GB (AWQ)
- **Islem suresi:** ~2-4 saat (H200'de)

## Sorun Giderme

### CUDA Out of Memory
```bash
# Daha az ornek kullan
NUM_SAMPLES=1000 bash run_reap_h200.sh
```

### Model bulunamadi
```bash
# Modeli indirdiginizden emin olun
bash download_model.sh
```

### HuggingFace token hatasi
```bash
# Token'i ayarlayin
export HF_TOKEN=hf_xxxxxxxxxxxxx
huggingface-cli login --token $HF_TOKEN
```

## Referanslar

- [REAP Paper](https://arxiv.org/abs/...)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B)
