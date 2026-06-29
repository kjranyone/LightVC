# LightVC Concept

> **Last updated: 2026-06-29.** This document reflects the validated architecture and C2 research direction.

## What LightVC Is

LightVC is a real-time voice conversion system that operates entirely in neural-codec (DAC) latent space. A lightweight adapter transforms the source speaker's codec representation toward a target speaker, conditioned on an ECAPA speaker embedding. The frozen DAC decoder reconstructs the waveform.

```
Mic (cpal) → DAC encoder → soft RVQ → B1 adapter (ECAPA cross-attn) → DAC decoder → Speaker
                                                              ↑
                                              Fine-tuned decoder (R2: short-window robust)
```

**Pure Rust inference** via Candle. No Python at deployment. <50ms latency on CUDA.

## Validated Architecture (v1.0)

### B1 UTTE Adapter

The production VC model: ECAPA 192-dim → 32 tokens → 4-head cross-attention.

```
z_q [B, 1024, T]
  → Conv1d(1024→256, k3)
  → ECAPA → Linear(192, 32×256) → 32 tokens [B, 32, 256]
  → 4-head cross-attention(256)
  → GELU
  → Conv1d(256→1024, k3) [zero-init]
  → z_q + delta
```

- Parameters: 3.5M
- Training: 10K VCTK same-text pairs, adapter-only (no generator)
- Results (200-pair bootstrap CI):
  - **Cross-text target SECS: 0.508**
  - **Cross-text margin: +0.323** (target - source)
  - Oracle ratio: 79.2% (vs src_K1 oracle +0.542)

### R2 Fine-Tuned Decoder

Frozen DAC decoder has boundary artifacts in short-window streaming (4f: 1.6 dB SNR). R2 fine-tunes the last 2 decoder blocks (1.78M params) to be short-window-tolerant.

```
block.2 (1.48M, 22kHz rate) + block.3 (0.30M, 44kHz rate) = 1.78M trainable
```

- Training: 10K VCTK utterances, immutable teacher, 5 epochs
- Results:
  - **4f SNR: +9.1 dB improvement** (1.6 → 10.7 dB)
  - **4f MCD: -11.93** (32.05 → 20.11)
  - Full-window reconstruction: unchanged (no catastrophic forgetting)
  - B1 offline margin: maintained (+0.339)
  - Rust/Candle parity: RMSE = 7.3e-3

### Two Runtime Controls (Knobs)

| Control | Range | Effect |
|---------|-------|--------|
| **wet/dry mix** | 0.0–1.0 | VC amount (0 = bypass, 1 = full conversion) |
| **τ (tau)** | 0.1–10.0 | Soft RVQ temperature (low = hard quantization, high = smooth) |

---

## Research Findings (What We Learned)

### Proven

1. **Cross-text generalization**: B1 adapter trained on same-text pairs generalizes to cross-text with ZERO degradation (200-pair, Δmargin = +0.002). Same-text supervision bias does not exist.

2. **RVQ speaker/content structure exists**: Depth surgery (200-pair) revealed speaker information distribution:
   - d1-3: 68% of speaker info (the "speaker core")
   - d0: 12% (mixed content + speaker)
   - d5-8: <6% each (fine detail, minimal speaker)

3. **ECAPA is the optimal conditioning signal**: Pre-trained speaker embeddings outperform any codec-derived representation (raw latent, depth-filtered latent, codebook histograms, co-occurrence lookup tables).

4. **Decoder fine-tune works**: Last 2 blocks fine-tune with immutable teacher and distillation losses dramatically improves short-window robustness without degrading full-window quality.

### Disproven (Negative Results)

1. **Learning-free VC is impossible**: Code choice is jointly (content, speaker)-dependent. Statistical methods (histogram, co-occurrence lookup) cannot factorize these dimensions.

2. **ref_latent token bank fails**: Reference audio's codec representation (raw or depth-filtered) is a worse conditioning signal than ECAPA's pre-extracted speaker vector.

3. **Per-depth discrete code modification is optimization-hostile**: Depth-aware adapters (V1: 8-dim code projection, V2: 1024-dim per-depth delta) both plateau below ECAPA B1 performance.

4. **Naive RVQ token swap fails**: Token validity does not imply residual-chain validity.

5. **Continuous latent L1/MSE regression collapses**: All latent-domain losses converge to codebook centroid.

---

## C2 Research Direction: Concept Voice Synthesis

v1.0 is a functional zero-shot VC (source → target speaker). The next goal is **concept voice synthesis**: generating natural-sounding voices from abstract presets rather than speaker references.

### The Gap

"Male input → sensual female voice" requires more than speaker conversion:
- **Natural female manifold**: The decoder/adapter must have seen diverse natural female speech
- **Attribute control**: "breathy", "warm", "intimate" must be controllable dimensions
- **Source timbre suppression**: Male vocal characteristics must be actively removed

### C2 Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **C2-0** | Female manifold dataset (Irodori-TTS 25K utterances, 500 speakers × 5 captions) | Generating |
| **C2-1** | Concept embedding from caption labels (neutral/soft/breathy/warm/low_tension) | Planned |
| **C2-2** | Domain bridge (source timbre leakage suppression, q0 correction) | Planned |
| **C2-3** | Concept preset system (voice synth knob controls) | Planned |

### Irodori-TTS Corpus

Using `Irodori-TTS-600M-v3-VoiceDesign` (MIT, Flow Matching TTS with caption control) to generate a diverse female voice corpus from 500 speakers. Caption text acts as the attribute label:

```
neutral:     "落ち着いた自然な女性の声で、普通の速さで読み上げてください。"
soft:        "柔らかく穏やかな声で、優しく語りかけるように読み上げてください。"
breathy:     "息多めの甘い声で、囁くように親密な距離感で読み上げてください。"
warm:        "温かく包容力のある声で、安心させるようにゆっくりと読み上げてください。"
low_tension: "リラックスして力の抜いた声で、少し低めのトーンで読み上げてください。"
```

This simultaneously provides:
- Large-scale female manifold for RVQ analysis
- Attribute-labeled pairs for concept embedding training
- Pairwise comparison material for perceptual studies

---

## Historical Context (Lessons Learned)

### What Was Tried and Abandoned

| Approach | Result | Lesson |
|----------|--------|--------|
| Continuous latent flow matching (Phase B/C) | Collapse to centroid | L1/MSE on latents always collapses |
| VC teacher distillation (SynthVC-style) | License risk + quality ceiling | Teacher-free policy adopted |
| Naive RVQ depth swap | Decoder-invalid trajectories | Residual chain must be preserved |
| Cross-text target-bank retrieval | Content/speaker inseparable | Frame-level retrieval insufficient |
| ref_latent token bank (true UTTE) | ECAPA strictly better | Pre-trained speaker extraction >> raw codec |
| Per-depth knob adapter (V1/V2) | Plateau below B1 | Per-depth structure is optimization-hostile |
| Learning-free code statistics | Margin -0.43 to -0.54 | Code choice is (content, speaker) joint function |

### What Endured

- **Codec-space VC**: VC happens in DAC latent space, not waveform space ✅
- **One-step conversion**: No ODE loop, no diffusion sampling ✅
- **Pure Rust inference**: Candle, no Python at deployment ✅
- **Frozen codec backbone**: Encoder and decoder are frozen (decoder last-2-blocks fine-tuned for streaming) ✅
- **MIT license**: All components MIT/Apache/ISC ✅

---

## Architecture Summary

```
┌─ Encoder (frozen) ──────────────────────────────────────┐
│ DAC encoder: Conv1d(1→64→128→512→1024) + ResBlocks     │
│ 44.1kHz PCM → [B, 1024, T] latent @ 86 Hz              │
└──────────────────────────────────────────────────────────┘
         │
┌─ Soft RVQ ───────────────────────────────────────────────┐
│ Sequential re-quantization with softmax(τ=5.0)          │
│ q0 (source) fixed → depths 1-8 soft-quantized           │
│ z_q = q0 + Σ soft_q_d                                   │
└──────────────────────────────────────────────────────────┘
         │                                    ┌─ ECAPA ─────┐
┌─ B1 Adapter ─────────────────────────────────│  192-dim   │
│ Conv1d(1024→256) + cross-attn(32 tokens)   │  speaker    │
│ + GELU + Conv1d(256→1024) [zero-init]       │  embedding │
│ 3.5M params                                 └────────────┘
└────────────────────────────────────────────────────────────┘
         │
┌─ Decoder (R2 fine-tuned) ────────────────────────────────┐
│ DAC decoder: conv1 + block.0-1 (frozen)                  │
│              + block.2-3 (fine-tuned for short-window)    │
│              + snake + conv2                              │
│ [B, 1024, T] → 44.1kHz PCM                               │
└──────────────────────────────────────────────────────────┘
```

## Deployment

- **Platform**: Windows (ASIO/WASAPI), macOS (CoreAudio), Linux (ALSA)
- **GUI**: egui/eframe (pure Rust)
- **Audio**: cpal ring-buffer streaming
- **Latency**: <50ms on CUDA (Balanced 4f), ~185ms on CPU (not real-time viable)
- **Weights**: `dac_44khz_finetuned.safetensors` (306MB), `dac_quantizer.safetensors` (4.7MB), `utte_adapter_b1.safetensors` (7.1MB)
