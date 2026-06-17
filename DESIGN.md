# LightVC Design Document

> Codec-space one-step streaming voice conversion, designed as a Rust client application.

## TL;DR

**LightVC** is a real-time voice conversion system that transforms audio in a pretrained neural codec's continuous latent space, using a lightweight one-step converter. The entire inference stack runs in **pure Rust** via [Candle](https://github.com/huggingface/candle), with no Python runtime, no ONNX export, and no external service dependency.

```
Mic (cpal) → DAC encoder (frozen) → continuous latent
           → one-step converter (~10M params, our model)
           → DAC decoder (frozen) → Speaker (cpal)
```

---

## Design Decisions

### 1. Inference framework: Candle (pure Rust)

| Option | Verdict |
|--------|---------|
| **Candle** | **CHOSEN.** Native implementations of DAC, EnCodec, Mimi, SNAC, Mamba2. Loads `.safetensors` directly. No C++ dependency. StreamingModule abstraction exists. |
| ort (ONNX) | Rejected. Neural codec ONNX export is "highly non-trivial" per HuggingFace engineers (`optimum` #1545, open for years). Mamba/SSM ONNX export actively failing (`mamba` #751). |
| Burn | Viable for converter CPU kernels (burn-flex), but no codec implementations. Consider as fallback for CPU optimization. |

**Evidence**: See [RESEARCH.md](RESEARCH.md) section 1.

### 2. Codec backbone: DAC (Descript Audio Codec)

| Property | Value |
|----------|-------|
| Sample rate | 44,100 Hz |
| Frame rate | ~86 Hz (hop = 512 samples, ~11.6 ms/frame) |
| Latent dim | 1,024 |
| RVQ depth | 9 codebooks |
| Codebook | 1,024 entries × 8 dims (factorized) |
| Parameters | ~76.6M (encoder + decoder) |
| License | **MIT** |
| HF model | `descript/dac_44khz` |

**Rationale**: MIT license (commercial-safe), 44.1 kHz quality. The DAC
architecture (encoder + decoder + Snake) is reimplemented natively in
`dac_model.rs` rather than reusing `candle-transformers::models::dac`,
because the HuggingFace checkpoint uses transformers-style safetensors
key names that the upstream module does not match.

**Known challenges and mitigations** (see [ARCHITECTURE.md](ARCHITECTURE.md) section 4 for details):

| Challenge | Mitigation |
|-----------|------------|
| Upstream Candle DAC assumes PyTorch-original key names; HF `descript/dac_44khz` uses transformers-style keys | Full native reimplementation in `dac_model.rs` (~400 LOC). See ARCHITECTURE §3.3, §6.3. |
| **Non-causal** architecture (future context required) | Chunked processing with bounded lookahead (40-120 ms). Map to quality/latency modes per CONCEPT.md. |
| No **StreamingModule** implementation | Implement streaming wrapper with conv-state caching + overlap-add. |
| 86 Hz frame rate (6.9x Mimi's 12.5 Hz) | Converter stays lightweight (Conv1d-only, ~10M params). At 86 Hz this is ~860 MFLOP/s for a 10M model — well within CPU budget. |

### 3. Converter architecture: Causal Conv1d flow-matching latent converter

- **Rectified flow formulation (1-NFE): single forward pass at inference, no ODE loop.**
- Phase B (warm-start): AutoVC-style bottleneck autoencoder in DAC latent space.
- Phase C (core): Rectified / linear flow matching, target = real speaker latent.
- Phase 2: Add MeanVC2-style Universal Timbre Token Encoder via cross-attention.
- Phase 3: Progressive RVQ-depth factorized FM heads (novel contribution).
- Parameters: 5-15M (Phase B/C), 15-30M (Phase 2+).

### 4. Training: Direct flow matching (NO VC teacher)

- **No teacher distillation.** Trained from scratch via flow matching.
- Target latent = DAC encoding of a *real* target-speaker recording (any text).
- Timbre-shifter augmentation (signal processing, not a neural teacher).
- Rationale: survey of 16 SOTA zero-shot VC systems shows 14 are teacher-free.

### 5. Target application: Desktop real-time VC client

- **Platform priority**: Windows (primary), macOS, Linux.
- **UI**: egui/eframe (pure Rust, immediate mode, real-time level meters).
- **Audio I/O**: cpal (WASAPI/CoreAudio/ALSA), optional ASIO on Windows.
- **Latency target**: 80-150 ms total round-trip (mode-dependent).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Rust Client (binary)                      │
│                                                             │
│  ┌──────────┐  ring buf  ┌─────────────────┐  ring buf  ┌──────────┐
│  │ cpal     │───────────►│  Inference      │───────────►│ cpal     │
│  │ capture  │            │  Thread         │            │ playback │
│  │ (mic)    │            │                 │            │ (spk)    │
│  └──────────┘            │  ┌───────────┐  │            └──────────┘
│                          │  │ resample  │  │
│                          │  │ device↔44k│  │
│                          │  ├───────────┤  │
│                          │  │ DAC enc   │  │  frozen
│                          │  │ (frozen)  │  │
│                          │  ├───────────┤  │
│                          │  │ Converter │  │  trained
│                          │  │ (ours)    │  │
│                          │  ├───────────┤  │
│                          │  │ DAC dec   │  │  frozen
│                          │  │ (frozen)  │  │
│                          │  ├───────────┤  │
│                          │  │ resample  │  │
│                          │  │ 44k↔device│  │
│                          │  └───────────┘  │
│                          └─────────────────┘
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ egui UI (main thread)                                 │  │
│  │  - device selection, target voice load               │  │
│  │  - latency/quality mode toggle                       │  │
│  │  - real-time input/output level meters               │  │
│  │  - prosody/rhythm controls (Phase 4)                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

Detailed design: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Model Data Creation Summary

The converter model is trained **from scratch via flow matching** (no VC teacher):

```
Phase A: Corpus Encoding (offline, Python/XPU)
  1. Collect multi-speaker speech (LibriTTS / VCTK, non-parallel OK)
  2. DAC-encode all utterances → latent shards
  (No teacher. No pairing. No synthetic generation.)

Phase B: Warm-start (bottleneck autoencoder, ~2h on B580)
  3. Converter learns to autoencode DAC latents with speaker bottleneck
  4. Target speaker re-injected from reference encoder

Phase C: Flow Matching (core training, ~5-7 days on B580)
  5. Converter learns: z_src(content) + speaker(ref) → z_tgt(real)
     where z_tgt is the DAC latent of a REAL target-speaker recording
  6. Losses: flow-matching velocity MSE + latent L1 + speaker SECS + content MI
  7. Export converter weights as .safetensors

Phase D: Deploy (Rust client)
  8. Load DAC frozen weights + converter weights in Candle
  9. Run real-time streaming inference (one-step, no ODE loop)
```

Detailed pipeline: [MODEL_TRAINING.md](MODEL_TRAINING.md)

---

## Implementation Phases

| Phase | Goal | Key Deliverable |
|-------|------|-----------------|
| **0** | DAC streaming proof-of-concept | Rust binary: WAV → DAC encode → DAC decode → WAV (round-trip) ✅ |
| **B** | Bottleneck warm-start | AutoVC-in-DAC-space converter (audible VC, low quality) |
| **C** | Flow matching training | From-scratch SOTA-tier converter (no teacher) |
| **2** | Universal Timbre Token Encoder | Zero-shot target voice from 5-30s reference |
| **3** | Progressive RVQ-depth factorized FM heads | Novel contribution: depth-axis control |
| **4** | Prosody/rhythm factorization | Preserve / blend / imitate / flatten modes |
| **5** | Real-time client | egui desktop app with cpal streaming |

---

## Key Research Insights (from literature survey)

| Insight | Source | Application in LightVC |
|---------|--------|--------------------------|
| VC = codec-space translation, not waveform generation | X-VC concept | Converter operates on DAC latents only |
| One-step conversion via rectified flow (1-NFE) | Lipman 2023, Liu 2022 | Single forward pass, no ODE loop |
| **14/16 SOTA VC systems trained without teacher** | R-VC, REF-VC, EZ-VC, Seed-VC, MeanVC2 | Direct flow matching, no distillation |
| Flow target = real recording latent, not teacher output | Flow matching theory | Phase C trains on real target-speaker DAC latents |
| Timbre-shifter augmentation (signal processing, not neural) | Seed-VC training recipe | Phase C data augmentation |
| Bounded future context >> strict causal for quality | MeanVC2 (FRC, 110ms) | Latency/quality mode switch |
| Universal timbre token bank >> single speaker embedding | MeanVC2 (UTTE, K=32) | Cross-attention timbre conditioning |
| Content/prosody/rhythm factorization adds product value | Discl-VC, R-VC | Separate control streams |
| Progressive depth-wise codec decoding | DiFlow-TTS, Mimi progressive | RVQ depth as latency/fidelity axis |

Full evidence: [RESEARCH.md](RESEARCH.md)
