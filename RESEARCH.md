# LightVC Research Evidence Base

Summary of the literature survey and technology evaluation that informed the design decisions.

---

## 1. Rust ML Inference Framework Comparison

### Decision: Candle (pure Rust)

| Framework | Maturity | Codec Models | Streaming | GPU Support | Verdict |
|-----------|----------|-------------|-----------|-------------|---------|
| **Candle** (HF) | 0.10.2, very active | **DAC, EnCodec, Mimi, SNAC, Mamba2 native** | StreamingModule trait | CUDA, Metal, CPU(MKL/Accel) | **CHOSEN** |
| ort (ONNX) | 2.0-rc.12 | Requires ONNX export (blocked) | Manual state mgmt | CUDA, DirectML, CoreML | Rejected for codecs |
| Burn | 0.21.0, active | No codec implementations | No | wgpu, CUDA, CPU | Viable for converter CPU path |

### Critical Finding: ONNX Export is Blocked for Codecs

- **EnCodec**: HuggingFace engineer states export is "highly non-trivial" (`optimum` #1545, open for years). RVQ argmin loop + weight-norm + causal convs break tracing.
- **Mimi**: No official ONNX. Only Candle has a production implementation.
- **DAC**: Same RVQ issues. No official ONNX.
- **Mamba/SSM**: ONNX export actively failing (`mamba` #751, #200). Selective scan custom kernel can't be traced.

**Conclusion**: For leveraging frozen pretrained codec weights (core to LightVC), Candle is the only viable path. ONNX is viable only for models authored from scratch with export-friendly ops.

### Candle's Existing Codec Implementations

```
candle-transformers/src/models/
├── encodec.rs       ← EnCodec (full)
├── dac.rs           ← DAC (decode-only, PyTorch-original key names)
├── mimi/            ← Mimi (full, streaming)
├── snac.rs          ← SNAC
├── mamba.rs         ← Mamba SSM
├── mamba2.rs        ← Mamba2 (SSD algorithm, chunked)
```

> **Note**: LightVC does **not** use `candle-transformers::models::dac`.
> The HuggingFace `descript/dac_44khz` checkpoint uses transformers-style
> safetensors key names that do not match the upstream `dac.rs` (which
> assumes PyTorch-original names). We reimplemented the full DAC natively
> in `crates/lightvc-core/src/dac_model.rs` to match the HF weight keys.
> See [ARCHITECTURE.md §3.3](ARCHITECTURE.md) and §6.3 for details.

---

## 2. Voice Conversion Literature Survey

### SynthVC (NCMMSC 2025)
- **Paper**: arXiv:2510.09245
- **Key contribution**: Streaming end-to-end VC via synthetic parallel distillation from Seed-VC teacher. 77.1ms latency, 14.7M params.
- **Architecture**: AudioDec backbone, latent-to-latent causal conv converter. Two-stage training (mel L1 → GAN).
- **Relevance to LightVC**: Validates the synthetic parallel distillation approach. Converter architecture (causal conv blocks) is directly applicable.
- **Code**: Anonymous repo (not yet public).

### MeanVC 2 (Interspeech 2026)
- **Paper**: arXiv:2606.09050
- **Key contribution**: Future-Receptive Chunking (FRC) + Universal Timbre Token Encoder (UTTE). 110ms latency, 18M params.
- **FRC**: Layer-wise attention masks. Only layer 1 needs 1 future chunk; rest causal. Removes teacher-forcing mismatch.
- **UTTE**: 32 learnable timbre key-value pairs with shared priors (universal prototypes). Cross-attention from content features. Robust to low-quality references.
- **Mean flows**: 1-NFE sampling (no ODE loop). Single forward pass.
- **Relevance**: UTTE pattern directly applied in Phase 2. FRC concept → latency/quality modes. Mean flows → one-step philosophy.

### Discl-VC (Interspeech 2025)
- **Key contribution**: Discrete content/prosody token separation via SimVQ. Prosody further split into Duration, F0, Formant. Non-autoregressive prosody prediction via mask transformer.
- **Relevance**: Phase 4 (prosody/rhythm factorization) borrows the discrete token separation concept.

### R-VC (ACL 2025)
- **Paper**: arXiv:2506.01014
- **Key contribution**: Rhythm-controllable zero-shot VC. Content token deduplication (collapses repeated tokens → separates content from rhythm). Mask Generative Transformer for duration. Shortcut flow matching (2-step).
- **Results**: 2-step inference, RTF 0.07, UTMOS 4.10, SECS 0.931. Beats FACodec-VC, Diff-HierVC, CosyVoice-VC.
- **Relevance**: Content token deduplication concept. Shortcut flow matching validates few-step generation.

### DiFlow-TTS (Sep 2025)
- **Paper**: arXiv:2509.09631
- **Key contribution**: First purely discrete flow matching for speech. Factorized prediction heads (prosody + acoustic detail). Up to 25.8x faster than baselines.
- **Code**: github.com/ishine/DiFlow-TTS (released Jan 2026)
- **Relevance**: Factorized heads concept → progressive RVQ-depth conversion. Code available for study.

### Seed-VC (Nov 2024)
- **Paper**: arXiv:2411.09943
- **Key contribution**: Zero-shot VC with DiT + in-context learning + timbre shifter.
- **Quality**: S-MOS 4.34, N-MOS 4.02, WER 2.28 (highest among non-streaming baselines).
- **Code**: github.com/Plachtaa/seed-vc (**GPL-3.0**, archived 2025-11, read-only)
- **Critical finding**: Seed-VC is **trained without a VC teacher** — its "timbre shifter"
  is signal-processing augmentation, not a neural model. 14/16 SOTA zero-shot VC systems
  are teacher-free. This eliminated the distillation plan.
- **Relevance**: Architecture/augmentation ideas borrowed (timbre shifter, in-context
  conditioning), but NOT used as teacher. No code/weights dependency.

### Astrape (the baseline to beat)
- **Repo**: github.com/stremtec/astrape-vc (22 stars)
- **Architecture**: 16kHz → streaming log-mel → causal ContentStudent (768d × 10 layers) → DirectWaveDecoder → 44.1kHz. VoiceBank single-reference (128d embedding).
- **Limitations**:
  1. Custom F³ encoder/decoder trained from scratch (doesn't leverage pretrained codecs)
  2. No factorized control (single content stream)
  3. Strict causal only (no quality/latency tradeoff)
  4. Single teacher (MioCodec)
  5. No fine-grained timbre retrieval

### X-VC (concept reference)
- **Note**: Could not be verified as a published paper. Treat as design spec from CONCEPT.md. Components are individually validated by SynthVC + Seed-VC + MeanVC2.

---

## 3. Neural Codec Comparison

| Codec | SR | Frame Rate | RVQ | Codebook | Params | License | Streaming | Candle |
|-------|-----|-----------|-----|----------|--------|---------|-----------|--------|
| **DAC** (chosen) | 44.1kHz | 86 Hz | 9 | 1024×8 | 77M | **MIT** | Non-causal | ✅ (decode-only) |
| Mimi | 24kHz | 12.5 Hz | 8 | 32 | 110M | Apache-2.0 | Causal | ✅ (full, streaming) |
| EnCodec | 24/48kHz | 75 Hz | ≤32 | 1024 | 28M | CC-BY-NC | Causal | ✅ |
| SoundStream | 16/24kHz | 50-100Hz | 3-4 | 1024 | 20-70M | Proprietary | Causal | ❌ |

### DAC Selection Tradeoffs

| Advantage | Cost |
|-----------|------|
| MIT license (commercial-safe) | Non-causal → bounded lookahead required |
| 44.1kHz high quality | 86 Hz frame rate → 6.9x more converter compute vs Mimi |
| Snake activation (music-grade) | Candle impl is decode-only (encoder needs wiring) |
| No StreamingModule | Must implement streaming wrapper |

### Mitigation Summary
- **Non-causal**: Overlap-add with bounded lookahead (40-120ms). Mapped to quality modes.
- **Decode-only**: Converter operates in continuous latent space → quantizer not needed.
- **86 Hz**: Lightweight Conv1d converter stays within CPU budget (~860 MFLOP/s for 10M model).
- **No streaming**: Implement conv-state caching + overlap-add in `lightvc-core`.

---

## 4. Streaming Implementation Evidence

### sherpa-onnx (production reference)
- `github.com/k2-fsa/sherpa-onnx` proves ONNX speech inference at scale.
- Streaming ASR with chunked state externalization (conv states as I/O tensors).
- int8 quantized models work well for speech in production.
- **Note**: sherpa-onnx runs ASR/TTS, not codecs. Codec ONNX export remains unsolved.

### cpal audio latency
| Platform | Backend | Typical Latency |
|----------|---------|-----------------|
| Windows | WASAPI shared | 10-30 ms |
| Windows | WASAPI exclusive / ASIO | 3-5 ms |
| macOS | CoreAudio | 5-15 ms |
| Linux | ALSA/PipeWire/JACK | 5-20 ms |

### rubato resampling
- v3.0: `process_into_buffer()` is zero-allocation in steady state (real-time safe).
- SIMD accelerated (AVX2, SSE3, NEON).
- Async mode allows on-the-fly ratio adjustment for clock drift.

---

## 5. Quantization Findings

| Method | Speedup | Quality Impact | Recommendation |
|--------|---------|----------------|----------------|
| FP16 | ~2x (GPU) | Negligible | Use on GPU always |
| INT8 dynamic | 2-4x (CPU) | <2% degradation | Safe for converter linear layers |
| INT8 static | 3-5x (CPU) | <2% with calibration | Best for production CPU |
| INT4 | 4-8x | Risky for audio | Avoid for codecs |

**Codec quantization rule**: Quantize linear/attention layers to INT8, **keep conv layers in FP32** (conv quantization introduces audible artifacts in audio codecs).

---

## 6. UI Framework

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **egui/eframe** | Pure Rust, 60fps, custom painting for meters, no JS | Not native-looking | **CHOSEN** |
| Tauri | Native web UI, self-updater | WebView overhead for real-time metering, IPC latency | Overkill |
| CLI | Simplest | No visual feedback | Ship as secondary mode |

---

## 7. Dataset Summary

| Dataset | Hours | Speakers | License | Cost |
|---------|-------|----------|---------|------|
| LibriTTS | 585h | 2,456 | ODC-BY | Free |
| VCTK | 44h | 110 | ODC-BY | Free |
| Libriheavy | 50,000h | 7,000 | CC-BY-4.0 | Free |
| MLS | 2,400h+ | thousands | CC-BY-4.0 | Free |
| Expresso | 24h | 4 | CC-BY-4.0 | Free |
| Emilia | 25,000h | — | CC-BY-NC-4.0 | Non-commercial |

**Primary recommendation**: LibriTTS (clean, multi-speaker, permissive license) for both source utterances and target references.

---

## 8. Key Architectural Insights

### "VC = codec-space translation"
The most important conceptual shift. Instead of generating waveforms, the converter transforms latents. The frozen codec decoder handles waveform synthesis. This reduces the VC model to a lightweight latent mapping function.

### "One-step > multi-step for latency"
Both MeanVC2 (mean flows, 1-NFE) and X-VC (one-step codec conversion) converge on this. Flow matching ODE loops (Astrape: 4-8 steps) are fundamentally slower than a single forward pass.

### "Bounded lookahead >> strict causal for quality"
MeanVC2's FRC shows that only the first attention layer needs future context. Strict causal (0ms lookahead) degrades quality significantly. The CONCEPT.md's 0/40/80ms mode switch is the right design.

### "Progressive depth as an axis"
Neural codecs have RVQ depth (9 for DAC). Most VC systems treat all depths equally. DiFlow-TTS's factorized heads + the streaming TTS progressive depth-wise decode concept suggest treating depth as a latency/fidelity axis — a genuinely novel contribution for VC.

---

## References

1. SynthVC — arXiv:2510.09245 (NCMMSC 2025)
2. MeanVC 2 — arXiv:2606.09050 (Interspeech 2026)
3. Discl-VC — Interspeech 2025 (wkd12345.github.io/disclvc)
4. R-VC — arXiv:2506.01014 (ACL 2025)
5. DiFlow-TTS — arXiv:2509.09631 (code: github.com/ishine/DiFlow-TTS)
6. Seed-VC — arXiv:2411.09943 (code: github.com/Plachtaa/seed-vc, GPL-3.0, archived 2025-11)
7. Astrape — github.com/stremtec/astrape-vc
8. DAC — github.com/descriptinc/descript-audio-codec (MIT)
9. Candle — github.com/huggingface/candle (Apache-2.0)
10. cpal — github.com/RustAudio/cpal
11. sherpa-onnx — github.com/k2-fsa/sherpa-onnx
12. ONNX EnCodec export blocker — huggingface/optimum#1545
13. ONNX Mamba export failure — state-spaces/mamba#751
14. AutoVC — arXiv:1905.05879 (bottleneck VC, teacher-free)
15. VQMIVC — arXiv:2106.10132 (MI disentanglement, teacher-free)
16. Diff-HierVC — arXiv:2311.04693 (hierarchical diffusion VC, teacher-free)
17. CoDiff-VC — arXiv:2411.18918 (codec-assisted diffusion, teacher-free)
18. EZ-VC — arXiv:2505.16691 (self-supervised NAR FM, teacher-free)
19. REF-VC — arXiv:2508.04996 (SSL + random erase, matches Seed-VC from scratch)
20. NaturalSpeech 3 / FACodec — arXiv:2403.03100 (factorized codec)

---

## 9. Training Paradigm Survey (2026-06 Revision)

### The Critical Finding

**Seed-VC itself is trained without a VC teacher.** Its "timbre shifter" is
signal-processing data augmentation (pitch/formant perturbation), not a neural
teacher. **14 of 16 surveyed SOTA zero-shot VC systems are teacher-free.**
Teacher distillation (SynthVC, FasterVoiceGrad) is a *latency compression*
trick, not a quality requirement.

This finding eliminated the Seed-VC dependency from LightVC's training plan.

### Teacher-Free Paradigm Comparison

| Paradigm | Teacher? | Data | Codec native? | 12GB B580? | Zero-shot? | Quality |
|----------|----------|------|---------------|------------|------------|---------|
| 1 Parallel direct | No | parallel | ✅ | ✅ | weak | mid |
| 2 Bottleneck + swap | No | non-parallel | ✅ | ✅ cheapest | ✅ | low-mid |
| 3 Contrastive/MI | No | non-parallel | ✅ (loss) | ✅ | ✅ | auxiliary |
| 4 In-context (DiT) | No | non-parallel | ✅ (LM) | ❌ heavy | ✅ | SOTA |
| 5 SSL disentangle | No (frozen) | non-parallel | ⚠️ bridge | ⚠️ borderline | ✅ | SOTA |
| 6 Flow matching (scratch) | No | non-parallel | ✅ + novel | ✅ if small | ✅ | SOTA |

### Decision: Paradigm 6 + Paradigm 2 Hybrid

- **Phase B:** AutoVC-style bottleneck warm-start (Paradigm 2)
- **Phase C:** Mean-flow / shortcut flow matching, target = real speaker latent (Paradigm 6)
- **Novelty:** Progressive RVQ-depth factorized FM heads (no prior VC system does this)

### Why Not Each Alternative

| Option | Rejection reason |
|--------|-----------------|
| Seed-VC teacher | GPL-3.0, archived 2025-11, quality ceiling, license contamination |
| Paradigm 1 only | Zero-shot generalization too weak |
| Paradigm 4 (full DiT) | >12GB training, conflicts with "small converter" goal |
| Paradigm 5 (HuBERT) | Conflicts with CONCEPT.md "HuBERT必須にしない" principle |
| Any distillation | Caps quality at teacher, inherits artifacts, adds dependency |
