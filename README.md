# REAP V11 - Fixed VLM Expert Pruning

Router-weighted Expert Activation Pruning for Vision-Language MoE models.

**Based on [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)**

## 🚨 Critical Fix from V10

**V10 had a critical bug**: `compute_all_experts=False` meant only top-k experts were scored, giving **wrong importance values**.

**V11 fixes this**: Now computes ALL 128 expert outputs for accurate REAP scoring.

```python
# V10 (WRONG)
compute_all_experts=False  # Only scored ~8 experts per token

# V11 (CORRECT)  
compute_all_experts=True   # Scores ALL 128 experts
```

## Quick Start

```bash
# 1. Build Docker
./run.sh build

# 2. Run REAP scoring (FULL mode - accurate)
./run.sh reap

# 3. Prune model
./run.sh prune

# 4. Test
./run.sh test
```

## Scoring Modes

| Mode | Accuracy | Speed | Use Case |
|------|----------|-------|----------|
| `full` | ✅ High | 🐢 Slow | Production pruning |
| `router` | ⚠️ Approximate | 🚀 Fast | Quick experiments |

Set mode via environment:
```bash
export REAP_SCORING_MODE=full    # Default, recommended
export REAP_SCORING_MODE=router  # Fast approximation
```

## How REAP Works

```
REAP Score = mean(router_probability × ||expert_output||_L2)
```

1. **Router Probability**: How likely the router selects this expert
2. **Output Norm**: How much the expert contributes when selected
3. **Combined**: Experts with high prob AND high output are important

## Configuration

```bash
# Copy and edit
cp .env.example .env

# Key settings
REAP_MODEL_PATH=/path/to/model
REAP_PRUNE_RATIO=0.40        # Prune 40% of experts
REAP_SCORING_MODE=full       # full or router
REAP_MAX_SAMPLES=300         # Calibration samples
```

## Directory Structure

```
reap-vlm-fixed/
├── reap.py              # Main REAP scoring (FIXED)
├── prune_model.py       # Model pruning
├── test_pruned_model.py # Testing
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.sh
```

## Output

After running:

```
output/
├── reap_metadata.json    # REAP scores & experts to keep
└── Pruned-Model/
    ├── config.json       # Updated (num_experts reduced)
    ├── model-*.safetensors
    └── reap_metadata.json
```

## Troubleshooting

### "a a a a a" output from pruned model

This usually means:
1. Wrong experts were pruned (V10 bug - use V11)
2. Gate weights corrupted
3. Calibration data too narrow

**Solution**: Re-run with V11's `REAP_SCORING_MODE=full`

### CUDA OOM

```bash
# Reduce batch size
export REAP_BATCH_SIZE=1

# Use router-only mode (less accurate but faster)
export REAP_SCORING_MODE=router
```

### Zero-count experts warning

If analysis shows many zero-count experts:
- Ensure `REAP_SCORING_MODE=full`
- Check calibration data has diverse samples

## Differences from CerebrasResearch/reap

| Feature | CerebrasResearch | This Repo |
|---------|------------------|-----------|
| Model Type | Text-only LLM | Vision-Language |
| Vision Encoder | ❌ | ✅ Preserved |
| Calibration | Text | Text + Images |
| Quantization | FP16/BF16 | AWQ support |

## Citation

```bibtex
@misc{reap-vlm,
    title = {REAP V11: Fixed VLM Expert Pruning},
    note = {Based on CerebrasResearch/reap},
    url = {https://github.com/CerebrasResearch/reap}
}
```

## License

MIT License
