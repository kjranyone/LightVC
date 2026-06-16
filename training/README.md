# LightVC Training Pipeline

Python training pipeline for the LightVC converter model.
Trained via mean-flow matching — no VC teacher needed.

## Setup

```bash
cd training
uv sync
```

## Pipeline

```
Phase A: encode_corpus.py        Multi-speaker speech → DAC latents
Phase B: train_warmstart.py      Bottleneck autoencoder warm-start
Phase C: train_flow.py           Mean-flow matching (core training)
Phase D: export_weights.py       PyTorch → safetensors for Rust
```

## Quick Start

```bash
# 1. Generate TTS corpus (Edge TTS, 17 speakers, ~170 utterances)
uv run python generate_tts_corpus.py

# 2. Encode to DAC latents (needs DAC weights at ../models/dac_44khz.safetensors)
uv run python encode_corpus.py \
    --source ../data/tts_corpus \
    --output data/latents

# 3. Phase B: Warm-start
uv run python train_warmstart.py \
    --config configs/phase_b.yaml \
    --data data/latents \
    --output checkpoints/phase_b

# 4. Phase C: Flow matching
uv run python train_flow.py \
    --config configs/phase_c.yaml \
    --data data/latents \
    --output checkpoints/phase_c

# 5. Export
uv run python export_weights.py \
    --checkpoint checkpoints/phase_c/best.pt \
    --output ../models/converter.safetensors \
    --model-type flow

# 6. Inference
uv run python infer_flow.py \
    --source source.wav --reference reference.wav \
    --output converted.wav \
    --converter checkpoints/phase_c/best.pt
```

## Configs

| File | Description |
|------|-------------|
| `configs/phase_b.yaml` | Warm-start (bottleneck autoencoder) |
| `configs/phase_c.yaml` | Flow matching (core, init from phase_b) |

## Files

| File | Purpose |
|------|---------|
| `converter.py` | Converter + FlowConverter (PyTorch, mirrors Rust) |
| `encode_corpus.py` | Phase A: speech corpus → DAC latents |
| `train_warmstart.py` | Phase B: bottleneck autoencoder |
| `train_flow.py` | Phase C: mean-flow matching |
| `timbre_shifter.py` | Signal-processing augmentation (not a teacher) |
| `generate_tts_corpus.py` | Generate Edge TTS multi-speaker corpus |
| `infer_flow.py` | One-step inference test |
| `export_weights.py` | Export to safetensors |
