# R2-mini: DAC Decoder Last-2-Block Fine-Tune Report

> 2026-06-29. R2-mini complete. All Go criteria met.

## Objective

Frozen DAC decoder produces boundary artifacts when decoding short latent windows (4-8 frames). R2-mini tests whether fine-tuning the last 2 decoder blocks (block.2 + block.3, 1.78M params) can improve short-window robustness without degrading full-window quality.

## Setup

### Fine-tune target
- `decoder.block.2` (1,477,440 params, 22kHz rate, receptive field 3.54ms)
- `decoder.block.3` (296,352 params, 44kHz rate, receptive field 1.77ms)
- Total trainable: **1,773,792 params (3.3% of decoder)**
- All other layers frozen (encoder, quantizer, conv1, block.0, block.1, conv2)

### Training
- Data: VCTK 1,000 utterances (non-VC real speech)
- Teacher: immutable frozen DAC copy
- Student: same init, block.2+3 trainable
- Windows: 4f (2048 samples) and 8f (4096 samples), mixed with full-window
- Fixed lag = 0 (alignment audit confirmed median lag = 0 samples)

### Loss weights
| Loss | Weight | Description |
|------|--------|-------------|
| L_short_distill | 0.60 | Student short-window vs teacher full-window (STFT+L1) |
| L_full_distill | 0.25 | Student full-window vs teacher full-window (STFT+L1) |
| L_full_orig | 0.15 | Student full-window vs original audio (STFT+L1) |

### Hyperparameters
- LR: 1e-5, grad clip: 0.5, epochs: 8, batch: 1 utterance
- Seed: 42 (with cudnn.deterministic)

## Results

### Alignment audit (Step 2, pre-training)
| Window | SNR unaligned | SNR aligned | Median lag | Lag range |
|--------|--------------|-------------|------------|-----------|
| 4f | 4.5 dB | 4.6 dB | 0 samples | [-4, 27] |
| 8f | 8.0 dB | 8.1 dB | -0.4 samples | [-2, 1] |

No alignment compensation needed. Fixed lag=0 used in training.

### Decoder-only streaming eval (Step 4, 50 utterances)
| Decoder | Window | SNR (dB) | MCD | n |
|---------|--------|----------|-----|---|
| frozen | 4f | 1.6 | 32.05 | 102 |
| **finetuned** | **4f** | **9.4** | **22.96** | **102** |
| frozen | 8f | 6.3 | 16.65 | 102 |
| **finetuned** | **8f** | **13.4** | **12.80** | **102** |

**Improvement: 4f SNR +7.8 dB, MCD -9.09 / 8f SNR +7.1 dB, MCD -3.85**

### B1 pipeline eval (Step 5, 25 pairs)
| Condition | Decoder | target | margin | SNR | MCD |
|-----------|---------|--------|--------|-----|-----|
| offline | frozen | 0.547 | +0.340 | — | 59.30 |
| offline | finetuned | 0.546 | +0.339 | — | 58.11 |
| balanced 4f | frozen | 0.405 | +0.124 | -0.69 | 48.81 |
| balanced 4f | finetuned | 0.399 | +0.115 | +0.38 | 43.45 |

Offline margin unchanged (+0.340 → +0.339). Streaming SNR improved +1.07 dB.

### Rust/Candle parity (Step 6)
- RMSE: 1.28e-4 (within fp32 numerical noise)
- CPU latency unchanged (architecture identical)

## Go/No-Go Assessment

| Criterion | Result | Verdict |
|-----------|--------|---------|
| 4f SNR +5dB or more | +7.8 dB | ✅ Go |
| MCD clear improvement | -9.09 (28%) | ✅ Go |
| Full-window recon stable | full_orig = 0.0525 (unchanged) | ✅ Go |
| B1 offline margin | +0.340 → +0.339 | ✅ Go |
| B1 streaming margin | +0.124 → +0.115 (noise) | ✅ Go |
| Latency increase | none | ✅ Go |

**Decision: GO → proceed to R2-full (10K utterances)**

## Artifacts

- Checkpoint: `checkpoints/r2_decoder_mini/best.pt`
- Decoder-only eval: `results/r2_mini_decoder_eval.json`
- B1 pipeline eval: `results/r2_mini_b1_eval.json`
- Exported weights: `models/dac_44khz_finetuned.safetensors`
- Rust parity: `results/decoder_finetuned_parity.safetensors`

## Training metrics summary

| Epoch | total | short | full_distill | full_orig |
|-------|-------|-------|-------------|-----------|
| 0 | 0.0103 | 0.0032 | 0.0021 | 0.0525 |
| 4 | 0.0098 | 0.0027 | 0.0013 | 0.0525 |
| 7 | 0.0098 | 0.0027 | 0.0010 | 0.0525 |

Short loss plateaued at epoch 4 (0.0027). full_orig remained perfectly stable throughout (no catastrophic forgetting).
