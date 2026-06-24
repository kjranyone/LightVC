# Literature Update 2026-06-21

This note supersedes older design notes that treat continuous DAC latent flow matching as the active LightVC path.

## Current LightVC Position

LightVC should remain a codec-space, low-latency VC project, but the active hypothesis is no longer:

```text
source continuous DAC latent -> one-step continuous latent converter -> frozen DAC decoder
```

That path failed experimentally:

- continuous latent / STE / code CE / embedding MSE all hit RVQ-cascade sensitivity;
- latent cosine around `0.67` could still decode to SECS around `0.03`;
- naive RVQ token swaps were only random-mix quality;
- cross-text retrieval and subsequence DTW failed for free-conversation use.

The active hypothesis is:

```text
source q0 content anchor
+ generated target-like residual trajectory
+ residual-chain-preserving re-quantization
+ DAC/tolerant decoder
```

Same-text/content-aligned oracle remains strong (`SECS 0.656-0.686`, `CER 0.057-0.082`), but free conversation and singing require a generator or decoder-adapter path, not bank retrieval.

## Relevant Recent Work

### X-VC: codec-space streaming VC

- Paper: <https://arxiv.org/abs/2604.12456>
- Code/checkpoints: <https://github.com/Jerrister/X-VC>
- Summary: one-step conversion in pretrained codec latent space with dual conditioning, frame-level target acoustic conditions, adaptive normalization, generated paired data, role assignment, and chunkwise overlap smoothing.
- LightVC relevance:
  - Strongly supports codec-space one-step VC as a viable research direction.
  - Does **not** prove that arbitrary DAC continuous latent regression is safe.
  - Uses generated paired data; LightVC policy forbids VC-teacher synthetic targets, so its training recipe cannot be copied directly.

### MeanVC 2: bounded future context and universal timbre tokens

- Paper: <https://arxiv.org/abs/2606.09050>
- Summary: future-receptive chunking and universal timbre token encoder improve robust low-latency streaming zero-shot VC; reported latency reduces from 211 ms to 110 ms.
- LightVC relevance:
  - Bounded lookahead is a practical design variable; strict causality should not be treated as sacred.
  - Timbre-token cross-attention remains useful as conditioning, but LightVC experiments show conditioning alone does not solve decoder-valid trajectory generation.

### StreamVC / RT-VC: real-time VC via learned causal units or articulatory features

- StreamVC: <https://arxiv.org/abs/2401.03078>
- RT-VC: <https://arxiv.org/abs/2506.10289>
- Summary:
  - StreamVC shows low-latency VC with SoundStream-style codec and causal soft speech units plus pitch information.
  - RT-VC uses articulatory feature space and causal models for real-time zero-shot VC.
- LightVC relevance:
  - Supports the need for a lightweight causal content representation.
  - Suggests that content disentanglement cannot be delegated to raw DAC latent cosine distance.
  - Final inference should avoid Wav2Vec2/HuBERT, but a small causal content/unit encoder is likely necessary.

### VChangeCodec: codec-integrated voice changer

- Paper: <https://openreview.net/pdf?id=qDSfOQBrOD>
- Summary: integrates voice changer into codec with causal projection at token level; claims about 40 ms latency and under 1M parameters.
- LightVC relevance:
  - Strong evidence for sub-50 ms feasibility when VC is built into or tightly coupled with codec.
  - Suggests a possible long-term fallback: train or adapt a VC-aware codec instead of relying on frozen DAC decoder behavior.

### Recent zero-shot singing VC

- YingMusic-SVC: <https://arxiv.org/abs/2512.04793>
- HQ-SVC: <https://arxiv.org/abs/2511.08496>
- R2-SVC: <https://arxiv.org/abs/2510.20677>
- Singing Voice Conversion Challenge 2025: <https://arxiv.org/abs/2509.15629>

Common findings:

- Singing VC needs explicit handling of F0, melody, vibrato, energy, breath/noise, and singing style.
- Recent high-quality systems often use flow/diffusion, DDSP/NSF, or specialized singing encoders.
- Robust real-world singing conversion must handle accompaniment leakage, separation artifacts, broad F0 range, and style variation.

LightVC relevance:

- A speech-only converter is unlikely to generalize to singing by scaling data alone.
- Singing support should be a separate mode or at least use a mode token and singing-specific losses.
- The final <50 ms constraint conflicts with heavy diffusion/vocoder pipelines. Singing support must keep F0/prosody explicit and the decoder lightweight.

## Implications for LightVC

### Keep

- Codec-space VC as the core product/research identity.
- One-step or very few-step streaming inference.
- Rust/Candle deployment target.
- Bounded lookahead modes.
- VC-teacher-free constraint.
- RVQ residual-chain validity as a first-class constraint.

### Stop Treating as Active

- Continuous DAC latent flow matching as the main route.
- Naive RVQ depth swap.
- L1/MSE-only latent prediction.
- Frame-independent target-bank retrieval for free conversation.
- Global or subsequence DTW for arbitrary cross-text enrollment.
- Wav2Vec2/HuBERT in final runtime.

### Next Research Priority

The next high-value question is decoder tolerance:

```text
Can DAC decoding be made tolerant to approximate target-like residual latents
without losing round-trip quality?
```

Recommended experiments:

1. Noisy latent tolerance sweep.
2. Soft residual decoding without hard RVQ argmin.
3. Small decoder adapter before the frozen DAC decoder.
4. Partial decoder fine-tune only if adapter fails.
5. Singing-mode oracle with F0/vibrato preservation metrics.

## Updated Architecture Sketch

```text
source wav
  -> DAC encode
  -> q0_source content/prosody anchor
  -> lightweight content/F0/energy streams
  -> target timbre profile
  -> target-like residual trajectory generator
  -> residual-chain-preserving re-quantization or tolerant decoder adapter
  -> DAC decode
```

For singing:

```text
mode = speech | singing
preserve source F0/melody/vibrato
convert timbre/formant/breathiness
track lyric intelligibility and pitch RMSE
```

## Decision Table

| Direction | Status | Reason |
|---|---|---|
| Continuous DAC latent FM | Frozen | off-manifold and cascade sensitivity |
| WORLD mcep retrieval | Stopped | corrected 200-pair ceiling around 0.36-0.40 |
| Naive RVQ swap | Stopped | residual-chain-invalid |
| Same-text residual re-quantization | Valid oracle | high SECS and low CER |
| Cross-text bank retrieval | Stopped as main route | weak SECS and high CER |
| Target-like residual generator | Active, but blocked by decoder sensitivity | needs tolerant decoder/adapter |
| Decoder adapter/tolerant decoder | Next main experiment | addresses observed failure mode directly |
| Singing support | Separate mode required | F0/vibrato/style constraints differ from speech |

