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

## Quick Start (smoke test)

Quick sanity check on a small Edge TTS corpus (~5 min encode + ~30 min train).
Use this to verify the pipeline runs end-to-end before committing to a full
training run.

```bash
# 1. Generate TTS corpus (Edge TTS, 17 speakers, ~170 utterances)
uv run python generate_tts_corpus.py

# 2. Encode to DAC latents (needs DAC weights at ../models/dac_44khz.safetensors)
uv run python encode_corpus.py \
    --source ../data/tts_corpus \
    --output data/latents

# 3. Phase B: Warm-start (smoke)
uv run python train_warmstart.py \
    --config configs/phase_b_smoke.yaml \
    --data data/latents \
    --output checkpoints/phase_b_smoke

# 4. Phase C: Flow matching (smoke)
uv run python train_flow.py \
    --config configs/phase_c_smoke.yaml \
    --data data/latents \
    --output checkpoints/phase_c_smoke

# 5. Export
uv run python export_weights.py \
    --checkpoint checkpoints/phase_c_smoke/best.pt \
    --output ../models/converter.safetensors \
    --model-type flow
```

## Production training

Full-scale training on LibriTTS / VCTK (100+ speakers). Expected wall time on
a single Arc B580: Phase B ~2 h, Phase C ~5-7 days. See MODEL_TRAINING.md for
corpus download instructions.

```bash
# Phase B: 50K steps, batch_size=8
uv run python train_warmstart.py \
    --config configs/phase_b.yaml \
    --data data/latents_libritts \
    --output checkpoints/phase_b

# Phase C: 200K steps, bf16, batch_size=8
uv run python train_flow.py \
    --config configs/phase_c.yaml \
    --data data/latents_libritts \
    --output checkpoints/phase_c

uv run python export_weights.py \
    --checkpoint checkpoints/phase_c/best.pt \
    --output ../models/converter.safetensors \
    --model-type flow
```

## Inference (Python reference)

```bash
uv run python infer_flow.py \
    --source source.wav --reference reference.wav \
    --output converted.wav \
    --converter checkpoints/phase_c/best.pt
```

## Evaluation

Offline metrics (SECS / UTMOS / WER) for a trained model. The metric models
are heavy, so they live in an optional dependency group:

```bash
uv sync --extra eval
```

Build a manifest of (source, reference, optional ground-truth text) pairs,
then run:

```bash
uv run python evaluate.py \
    --converter checkpoints/phase_c/best.pt \
    --manifest eval_manifest.json \
    --output eval_results.json
```

Targets (MODEL_TRAINING.md §Validation Protocol): SECS > 0.70, UTMOS > 3.5,
WER < 5%, WER degradation (src vs converted) < 2%. Any metric whose model
fails to load is reported as `null` and skipped without aborting the run.

## Configs

| File | Steps | Corpus | Purpose |
|------|-------|--------|---------|
| `configs/phase_b_smoke.yaml` | 10K | Edge TTS (17 spk) | Smoke test warm-start |
| `configs/phase_c_smoke.yaml` | 30K | Edge TTS (17 spk) | Smoke test flow matching |
| `configs/phase_b.yaml` | 50K | LibriTTS/VCTK (100+ spk) | Production warm-start |
| `configs/phase_c.yaml` | 200K | LibriTTS/VCTK (100+ spk) | Production flow matching (bf16) |

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
| `evaluate.py` | Offline metrics: SECS / UTMOS / WER |
| `download_corpus.py` | Fetch LibriTTS/VCTK from HuggingFace |
| `export_weights.py` | Export to safetensors |
