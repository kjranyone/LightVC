# LightVC Design Document

> **Last updated: 2026-06-29.** Reflects B1 adapter + R2 decoder fine-tune + C2 roadmap.

## TL;DR

**LightVC** is a real-time voice conversion system operating in DAC codec latent space. The current architecture consists of:

1. **Frozen DAC encoder** — source audio → 1024-dim latent @ 86 Hz
2. **Soft RVQ** — source q0 anchor + soft re-quantization (τ=5.0)
3. **B1 UTTE adapter** — ECAPA-conditioned cross-attention, 3.5M params
4. **R2 fine-tuned DAC decoder** — last 2 blocks fine-tuned for short-window streaming

Pure Rust inference via Candle. <50ms latency on CUDA.

```
Mic (cpal) → resample → DAC encoder → soft RVQ
                                          ↓
                               B1 adapter ← ECAPA (target speaker)
                                          ↓
                            R2 decoder → resample → Speaker (cpal)
```

---

## Design Decisions

### 1. Inference Framework: Candle (Pure Rust)

| Option | Verdict |
|--------|---------|
| **Candle** | **CHOSEN.** Native DAC reimplementation. Loads `.safetensors` directly. No C++/ONNX dependency. |
| ort (ONNX) | Rejected. Neural codec ONNX export is non-trivial. |
| Burn | Fallback for CPU kernels. No codec implementations. |

### 2. Codec Backbone: DAC + R2 Fine-Tune

| Property | Value |
|----------|-------|
| Codec | Descript Audio Codec (`descript/dac_44khz`) |
| Sample rate | 44,100 Hz |
| Frame rate | ~86 Hz (hop = 512) |
| Latent dim | 1,024 |
| RVQ | 9 codebooks × 1024 entries × 8 dims |
| Encoder | Frozen (~23M params) |
| Decoder block.0-1 | Frozen (~41M params) |
| **Decoder block.2-3** | **Fine-tuned (1.78M params)** — short-window robust |
| License | MIT |

**R2 Fine-Tune Details:**
- Training: 10K VCTK utterances, immutable teacher, 5 epochs, LR 5e-6
- Loss: 0.60 short-window distill + 0.25 full-window distill + 0.15 original reconstruction
- Result: 4f SNR +9.1 dB, full-window quality maintained
- Rust parity: RMSE = 7.3e-3

### 3. VC Model: B1 UTTE Cross-Attention Adapter

```
z_q [B, 1024, T]
  → Conv1d(1024→256, k3)
  → ECAPA(192-dim) → Linear(192, 32×256) → 32 tokens
  → 4-head MultiheadAttention(256, batch_first)
  → GELU
  → Conv1d(256→1024, k3) [zero-init residual]
  → z_q + delta
```

| Property | Value |
|----------|-------|
| Parameters | 3.5M |
| Conditioning | ECAPA speaker embedding (192-dim, speechbrain) |
| Training data | 10K VCTK same-text pairs (adapter-only, no generator) |
| Cross-text margin | +0.323 (200-pair bootstrap CI) |
| Oracle ratio | 79.2% |

**Key finding**: B1 adapter generalizes from same-text to cross-text with ZERO degradation. Same-text supervision bias was empirically disproven.

### 4. Training: VC-Teacher-Free

- **No VC teacher distillation** (project rule)
- No L1/MSE-only latent regression (proven to collapse)
- Target latent = real target speaker's DAC-encoded recording
- Auxiliary models (ECAPA, Whisper) used for loss/eval only, not at inference
- Seed-fixed for reproducibility (--seed 42, cudnn.deterministic)

### 5. Streaming Architecture

| Mode | Chunk | Lookahead | Decode Window | Use Case |
|------|-------|-----------|---------------|----------|
| Strict | 1f (512 smp) | 0f | 1f | Disabled (quality C/D, see #7) |
| **Balanced** | **4f (2048 smp)** | **4f (2048)** | **4f** | **Default** |
| Quality | 8f (4096 smp) | 8f (4096) | 8f | Offline/High-quality |

**Encoder overlap**: 2048 samples (4 frames) for context continuity.
**Decoder**: R2 fine-tuned (short-window tolerant, 4f SNR = 10.7 dB).

### 6. Target Application

- **Platform**: Windows (ASIO/WASAPI), macOS (CoreAudio), Linux (ALSA)
- **UI**: egui/eframe desktop app (3 tabs: offline/realtime/catalog)
- **Latency target**: <50ms (CUDA Balanced), <150ms (CPU fallback)
- **CLI**: `lightvc convert-b1` for offline WAV→VC→WAV

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    Rust Client (binary)                        │
│                                                                │
│  ┌──────────┐  ring buf  ┌──────────────────┐  ring buf  ┌──────────┐
│  │ cpal     │───────────►│  Inference       │───────────►│ cpal     │
│  │ capture  │            │  Thread          │            │ playback │
│  └──────────┘            │                  │            └──────────┘
│                          │  ┌────────────┐  │                                  │
│                          │  │ resample   │  │
│                          │  │ dev↔44.1k  │  │
│                          │  ├────────────┤  │
│                          │  │ DAC enc    │  │  frozen
│                          │  │ (frozen)   │  │
│                          │  ├────────────┤  │
│                          │  │ soft RVQ   │  │  τ=5.0
│                          │  │ (q0 fixed) │  │
│                          │  ├────────────┤  │
│                          │  │ B1 adapter │  │  3.5M, trained
│                          │  │ (ECAPA x-  │  │
│                          │  │  attn)     │  │
│                          │  ├────────────┤  │
│                          │  │ DAC dec    │  │  block.0-1 frozen
│                          │  │ (R2 fine-  │  │  block.2-3 fine-tuned
│                          │  │  tuned)    │  │
│                          │  ├────────────┤  │
│                          │  │ wet/dry τ  │  │  runtime knobs
│                          │  ├────────────┤  │
│                          │  │ resample   │  │
│                          │  │ 44.1k↔dev  │  │
│                          │  └────────────┘  │
│                          └──────────────────┘
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ egui UI                                                  ││
│  │  Tab 1: Offline (WAV file conversion)                    ││
│  │  Tab 2: Realtime (mic → VC → speaker)                    ││
│  │    - Device selection, B1 adapter/quantizer/timbre load  ││
│  │    - τ slider, wet/dry mix                               ││
│  │    - Strict/Balanced/Quality mode (Strict disabled #7)   ││
│  │    - Real-time level meters, latency/RTF display         ││
│  │  Tab 3: Catalog (voice reference library)                ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

---

## Implementation Status

| Component | Status | Details |
|-----------|--------|---------|
| DAC encoder/decoder (Rust) | ✅ Complete | `dac_model.rs`, parity MSE≈0 |
| DAC quantizer (Rust) | ✅ Complete | `soft_rvq.rs`, parity MSE≈0 |
| B1 UTTE adapter (Rust) | ✅ Complete | `utte_adapter.rs`, parity MSE≈0 |
| B1 streaming pipeline (Rust) | ✅ Complete | `b1_pipeline.rs`, CUDA <50ms |
| R2 fine-tuned decoder | ✅ Complete | block.2+3, 4f SNR +9.1dB |
| Rust/Candle parity test | ✅ Pass | RMSE 1.3e-4 (mini) / 7.3e-3 (full) |
| Desktop GUI (egui) | ✅ Complete | 3-tab app, B1 controls |
| CLI convert-b1 | ✅ Complete | Offline WAV→VC→WAV |
| ECAPA timbre extraction | ✅ Complete | `extract_timbre.py` (Python, offline) |
| Cross-text eval | ✅ Complete | 200-pair CI, Δmargin=+0.002 |
| Depth surgery | ✅ Complete | d1-3 = 68% speaker info |
| Streaming eval | ✅ Complete | F0/CER/MCD + SECS/SNR |
| Strict mode | ⚠️ Disabled | Quality C/D, warning in GUI/CLI (#7) |
| cpal on-device test | ❌ Pending | Issue #1 |
| C2 female manifold | 🔄 Generating | 25K Irodori-TTS utterances |
| C2 concept embedding | ❌ Planned | From caption labels |
| C2 domain bridge | ❌ Planned | Source timbre suppression |

---

## Research Backlog (v2.0+)

### R1: Per-Depth Knob (Deferred)
Post-hoc decomposition of B1 delta into depth components, or distillation from B1 teacher. Learning-free approaches (histogram, lookup) are proven impossible.

### R2: ✅ Decoder Fine-Tune (Complete)
Short-window streaming quality improved from C/D to A/B via last-2-block fine-tune.

### R3: Mimi Migration (Optional)
Causal codec for true low-latency + CPU viability. Deferred while DAC + R2 works.

### R4: Content Tokenizer (Phase 4)
WavLM-based content encoder to replace q0 anchor, eliminating source speaker leakage (SECS_source = 0.238).

### R5: Paper
Key contributions: cross-text generalization discovery, depth surgery map, learning-free impossibility, practical <50ms Rust/Candle VC.

---

## Key Files

| File | Purpose |
|------|---------|
| `crates/lightvc-core/src/dac_model.rs` | DAC encoder/decoder (Rust/Candle) |
| `crates/lightvc-core/src/soft_rvq.rs` | Soft RVQ + quantize_q0 |
| `crates/lightvc-core/src/utte_adapter.rs` | B1 UTTE cross-attention adapter |
| `crates/lightvc-core/src/b1_pipeline.rs` | B1Streaming + B1Offline pipeline |
| `crates/lightvc-core/src/lib.rs` | Backend enum (Legacy/B1) dispatch |
| `crates/lightvc-app/src/app.rs` | GUI app (3 tabs) |
| `crates/lightvc-app/src/cli.rs` | CLI (convert-b1, roundtrip, gui) |
| `training/train_phase3c_adapter.py` | B1 adapter training |
| `training/eval_cross_text.py` | Cross-text generalization eval |
| `training/eval_depth_surgery.py` | RVQ depth speaker attribution |
| `training/train_decoder_finetune.py` | R2 decoder fine-tune |
| `training/eval_decoder_streaming.py` | Decoder-only streaming eval |
| `training/generate_female_corpus_fast.py` | Irodori-TTS female corpus generation |
| `models/dac_44khz_finetuned.safetensors` | R2 fine-tuned DAC (306MB) |
| `models/dac_quantizer.safetensors` | DAC quantizer weights (4.7MB) |
| `models/utte_adapter_b1.safetensors` | B1 adapter weights (7.1MB) |
| `checkpoints/phase3c_ao_b1_ecapa/best.pt` | B1 best checkpoint (epoch 9, margin +0.282) |
| `checkpoints/r2_decoder_full/best.pt` | R2-full decoder fine-tune checkpoint |
