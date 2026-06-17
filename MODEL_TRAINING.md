# LightVC Model Training & Data Creation

Complete pipeline for creating the converter model — **trained from scratch, no VC teacher**.

> **Design revision (2026-06):** Dropped the Seed-VC teacher-distillation plan after
> surveying 16 SOTA zero-shot VC systems. 14 of them (Seed-VC included) are trained
> *without any VC teacher*. Teacher distillation (SynthVC, FasterVoiceGrad) is a
> *latency compression* trick, not a quality requirement. LightVC now trains
> directly via **mean-flow / shortcut flow matching in DAC continuous latent space**,
> with a bottleneck-autoencoder warm-start. This removes the teacher dependency
> entirely and unlocks the novel progressive RVQ-depth factorized FM heads.

---

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  OFFLINE (Python / PyTorch / XPU)            │
│                                                             │
│  Phase A: Corpus Preparation                                │
│    Multi-speaker speech (non-parallel OK)                   │
│    → DAC encode all utterances → latent shards              │
│    (No VC teacher. No paired text. No synthetic pairs.)     │
│                                                             │
│  Phase B: Warm-start (AutoVC-in-DAC-space, Paradigm 2)      │
│    Converter learns to autoencode DAC latents                │
│    with an information bottleneck that drops speaker info.  │
│    Target speaker re-injected from reference encoder.       │
│                                                             │
│  Phase C: Mean-flow matching (Paradigm 6, the core)         │
│    Converter learns z_src + speaker(ref) → z_tgt_real       │
│    where z_tgt_real is the DAC latent of a REAL recording   │
│    of the target speaker (any text).                        │
│    Loss: flow-matching velocity MSE + speaker consistency.  │
│                                                             │
│  Phase D: Export                                            │
│    Converter weights → converter.safetensors                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                  DEPLOY (Rust / Candle)                     │
│    converter.safetensors + dac_44khz.safetensors            │
│    → LightVC desktop client (one-step inference)          │
└─────────────────────────────────────────────────────────────┘
```

**Key contrast with the old plan:**

| Old (teacher distillation) | New (direct flow matching) |
|----------------------------|----------------------------|
| Generate 300K synthetic pairs with Seed-VC (~7 days A100) | Encode real speech with DAC (~hours on B580) |
| `z_tgt = DAC.encode(teacher.convert(src, ref))` | `z_tgt = DAC.encode(real_target_utterance)` |
| Quality ceiling = teacher quality | Quality ceiling = data + architecture |
| Teacher license contaminates output | No external model dependency |
| Multi-day Phase A blocking step | Phase A is just data encoding |

---

## Phase A: Corpus Preparation

### A.1 Data Philosophy

Voice conversion is non-parallel: we don't need `(speaker_A_saying_X, speaker_B_saying_X)`.
We need `(any_speaker_saying_anything, different_speaker_saying_anything_else)`.
The flow-matching objective (Phase C) learns the conditional distribution
`p(z_target_speaker | content(source), speaker(reference))` from non-parallel data.

### A.2 Data Sources

| Dataset | Hours | Speakers | License | Role |
|---------|-------|----------|---------|------|
| **LibriTTS** (train-clean-100) | 100h | 2,456 | ODC-BY | Primary training corpus |
| **VCTK** | 44h | 110 | ODC-BY | Accent diversity + parallel validation |
| **LibriTTS** (train-clean-360) | 360h | 2,456 | ODC-BY | Scale-up (Phase C+) |

**Parallel data note:** VCTK's same-text multi-speaker utterances are reserved for
*validation only* (clean content-preservation metric). Training uses non-parallel sampling.

### A.3 Augmentation (timbre shifter, no teacher)

Borrowed from Seed-VC's training recipe — this is **signal-processing augmentation**,
not a neural VC teacher:

```python
def timbre_shift(wav, sr, pitch_ratio_range=(0.8, 1.25), formant_shift_range=(0.85, 1.18)):
    """Perturb source timbre during training to prevent speaker leakage."""
    pitch_ratio = random.uniform(*pitch_ratio_range)
    formant_ratio = random.uniform(*formant_shift_range)
    wav = psola_pitch_shift(wav, sr, pitch_ratio)
    wav = formant_filter(wav, sr, formant_ratio)
    return wav
```

This forces the converter to rely on the reference embedding for speaker identity
rather than copying source timbre.

### A.4 Encoding Pipeline

```python
# training/encode_corpus.py

for utterance in corpus:
    wav_44k = resample(utterance, native_sr, 44100)
    z = dac.encode(wav_44k)               # [1024, T_frames]
    save(f"latents/{speaker_id}/{utt_id}.npy", z)
```

**No teacher, no pairing, no synthetic generation.** Just encode real speech.

---

## Phase B: Warm-start (Bottleneck Autoencoder)

### B.1 Why Warm-start

Jumping directly into flow matching on a randomly-initialized converter produces
unstable gradients (the model can't even autoencode). We warm-start with the
AutoVC-style bottleneck objective (Paradigm 2):

- **Content path:** latent → bottleneck Conv1d (force speaker info out) → content code
- **Speaker path:** reference latent → speaker encoder → speaker embedding
- **Reconstruction:** content code + speaker embedding → predicted latent
- **Loss:** L1(predicted, original) + speaker consistency

### B.2 Bottleneck Design

```
z_src [B, 1024, T]
    │
    ▼
┌─────────────────────────┐
│ Content Bottleneck       │
│  Conv1d(1024 → 256, k=1) │   ← 4× channel reduction
│  + causal Conv1d(k=7)    │     drops speaker info
│  + Conv1d(256 → 1024)    │
└────────────┬────────────┘
             │ content_code [B, 256, T]
             ▼
┌─────────────────────────┐
│ Speaker Injection (FiLM) │
│  γ, β = MLP(speaker_emb) │
│  z = γ * content + β     │
└────────────┬────────────┘
             │
             ▼
         z_pred [B, 1024, T]
```

The bottleneck dimension (256) is deliberately smaller than what's needed to
encode speaker identity, forcing the model to take speaker info from the reference.

### B.3 Training Config

```yaml
# configs/phase_b.yaml
model:
  latent_dim: 1024
  bottleneck_dim: 256        # force speaker info out
  hidden_dim: 1024
  n_conv_blocks: 4
  speaker_embed_dim: 256

training:
  batch_size: 8
  learning_rate: 3.0e-4
  max_steps: 50000           # ~2 hours on B580
  gradient_clip: 1.0

losses:
  reconstruction_l1: 1.0
  speaker_consistency: 0.5   # pred speaker ≈ ref speaker ([04-8])
  content_preservation: 0.3  # content_code(pred) ≈ content_code(src) ([04-9])
  speaker_classify: 0.5      # auxiliary CE, prevents encoder collapse ([04-11])

# Role assignment ([04-10]): teacher-free paradigm.
# cross_speaker is a REGULARIZER, not a target producer.
role_assignment:
  cross_speaker: 0.0         # 0 = pure AutoVC reconstruction (recommended)
```

---

## Phase C: Flow Matching (Core Training)

### C.1 The Key Insight

**Flow matching doesn't need a teacher.** The training target is a *real recording*
of the target speaker, not a teacher's output:

```
z_src  = DAC.encode(source_speaker_saying_anything)      # content source
z_ref  = DAC.encode(target_speaker_saying_anything_else) # reference for speaker
z_tgt  = DAC.encode(target_speaker_saying_yet_another)   # ← REAL target latent

# The flow-matching objective:
# Learn velocity field v(z_t, t | content(z_src), speaker(z_ref))
# such that integrating from z_0 = z_src arrives at z_1 ≈ z_tgt.
```

**Critical difference from old plan:** `z_tgt` is the DAC encoding of a real
target-speaker utterance. No VC teacher generated it. The model learns the
*actual distribution* of target-speaker latents, not a teacher's approximation.

### C.2 Rectified Flow Matching (1-NFE)

> **Note ([04-7] revision):** The implementation uses **rectified / linear flow
> matching** (Lipman 2023, Liu 2022), not MeanVC2's mean-flow formulation.
> The 1-NFE property arises from the flow being *linear* (constant velocity),
> so a single forward pass at `t=1` recovers the endpoint. MeanVC2 is credited
> for FRC (future-receptive chunking) and UTTE (universal timbre token
> encoder), which LightVC adopts, but the velocity-matching formulation is
> standard rectified flow.

A **linear** (rectified) flow has constant velocity, so a single forward pass
at `t=1` gives the endpoint — no ODE loop, no multi-step distillation:

**Training:**
```python
# Sample time t ~ U[0, 1]
t = torch.rand(B, 1, 1)

# Source-conditioned interpolation (z_0 = z_src, NOT noise — see C.3)
z_t = (1 - t) * z_src + t * z_tgt    # linear interp

# Target velocity (constant for a linear flow)
v_target = z_tgt - z_src              # = dz/dt, independent of t

# Predict velocity from (z_t, t, content, speaker)
v_pred = converter(z_t, t, content_code, speaker_embedding)

# Loss
loss = F.mse_loss(v_pred, v_target)
```

**Inference (1-step, no ODE loop):**
```python
# Linear flow → constant velocity → single step suffices.
# z_converted = z_src + v_pred(z_src, t=1, ref)
z_converted = converter.one_step(z_src, content_code, speaker_embedding)
```

This gives us the **one-step inference** we need for real-time, *without distilling
from a multi-step teacher*.

### C.3 Source-Conditioned Flow (Not Pure Noise)

A pure noise→target flow is inefficient for VC (most of the latent is already
correct — only speaker-related dimensions need to change). We use
**source-conditioned initialization**:

```python
# z_0 = z_src (not noise) perturbed by timbre shifter
z_0 = dac.encode(timbre_shift(source_wav))

# Flow target: z_tgt (real target-speaker latent)
# The model learns the residual transformation z_0 → z_tgt
```

This makes the flow short and stable — the converter only needs to learn the
*delta*, not reconstruct from scratch.

### C.4 Loss Functions

```python
def compute_losses(v_pred, v_target, z_converted, z_tgt,
                   speaker_pred, speaker_tgt, content_src, content_pred):
    losses = {}

    # 1. Flow-matching velocity MSE (primary)
    losses["fm_velocity"] = F.mse_loss(v_pred, v_target)

    # 2. Latent L1 at t=1 (endpoint check)
    losses["latent_l1"] = F.l1_loss(z_converted, z_tgt)

    # 3. Speaker similarity (SECS)
    losses["speaker_sim"] = 1.0 - F.cosine_similarity(
        speaker_pred, speaker_tgt, dim=-1
    ).mean()

    # 4. Content preservation (content code should not depend on target speaker)
    losses["content_inv"] = F.l1_loss(content_pred, content_src)

    # 5. Mutual information regularization (VQMIVC-style disentanglement)
    #    Optional, via gradient reversal on speaker classification of content_code

    total = (
        1.0 * losses["fm_velocity"] +
        2.0 * losses["latent_l1"] +
        1.0 * losses["speaker_sim"] +
        0.5 * losses["content_inv"]
    )
    return total, losses
```

### C.5 Training Config

```yaml
# configs/phase_c.yaml
model:
  latent_dim: 1024
  bottleneck_dim: 256
  hidden_dim: 1024
  n_conv_blocks: 4
  speaker_embed_dim: 256
  n_timbre_tokens: 32        # Phase 2 UTTE
  enable_timbre: true

training:
  init_from: checkpoints/phase_b/best.pt
  batch_size: 8
  learning_rate: 1.5e-4
  optimizer_betas: [0.8, 0.99]
  weight_decay: 0.01
  lr_scheduler_gamma: 0.9999
  max_steps: 200000
  gradient_clip: 1.0

  # Timbre shifter augmentation (signal processing, NOT a teacher)
  timbre_shift_prob: 0.5

losses:
  fm_velocity: 1.0
  latent_l1: 2.0
  speaker_sim: 1.0
  content_inv: 0.5
  content_mi: 0.1            # GRL disentanglement ([04-4])

data:
  sample_rate: 44100
  latent_frame_rate: 86
  max_utterance_frames: 600
  min_utterance_frames: 30

hardware:
  device: auto               # resolves to xpu on B580 ([04-13])
  mixed_precision: bf16      # B580 supports BF16
```

### C.6 Data Sampling Strategy

For each training step:

```python
def sample_training_batch(corpus):
    # 1. Pick a random source utterance (any speaker, any text)
    src = corpus.random_utterance()

    # 2. Pick a DIFFERENT target speaker
    tgt_speaker = corpus.random_speaker(exclude=src.speaker)
    tgt_utt = corpus.random_utterance_from(tgt_speaker)  # any text
    ref_utt = corpus.random_utterance_from(tgt_speaker)  # different utterance

    # 3. Encode all three (pre-computed in Phase A)
    z_src = load_latent(src)
    z_tgt = load_latent(tgt_utt)   # ← REAL target, no teacher
    z_ref = load_latent(ref_utt)

    return z_src, z_tgt, z_ref
```

---

## Phase D: Export

(Unchanged from before — the converter architecture is the same, only training
objective changed.)

```python
# training/export_weights.py
from safetensors.torch import save_file

model.load_state_dict(torch.load("checkpoints/phase_c_flow/best.pt")["model"])
save_file(model.state_dict(), "../models/converter.safetensors")
```

---

## Compute Requirements (Revised)

| Phase | Hardware | Time | Notes |
|-------|----------|------|-------|
| A: Corpus encoding | 1× B580 | ~12h for 100h audio | DAC encode only |
| B: Warm-start | 1× B580 | ~2h | 50k steps, bottleneck autoencoder |
| C: Flow matching | 1× B580 | ~5-7 days | 200k steps, the main training |
| D: Export | CPU | seconds | |

**Total: ~1 week on a single Intel Arc B580.** No A100 cluster needed.
No teacher model to download or run.

---

## Validation Protocol

### Offline Metrics

| Metric | Tool | Target |
|--------|------|--------|
| **SECS** | ECAPA-TDNN / WavLM | > 0.70 |
| **UTMOS** | UTMOS predictor | > 3.5 |
| **WER** | Whisper ASR | < 5% |
| **Content preservation** | WER(src) vs WER(converted) | < 2% degradation |

### VCTK Parallel Validation (held-out)

VCTK has same-text utterances across speakers. Use held-out pairs to measure:

```python
# For each (speaker_A_text_X, speaker_B_text_X) pair:
z_src = encode(speaker_A_text_X)
z_ref = encode(speaker_B_text_anything)
z_converted = converter(z_src, z_ref)

# Content preservation: ASR(z_converted) should match text_X
# Speaker similarity: SECS(z_converted, speaker_B) should be high
```

---

## Quick Start: Minimal Viable Training

```bash
# 0. Prepare environment (uv, already done)
cd training && uv sync

# 1. Download VCTK (~44h, 110 speakers, ODC-BY)
#    (manual download or use huggingface datasets)

# 2. Encode corpus to DAC latents (~1-2 hours on B580)
uv run python encode_corpus.py \
    --source /path/to/vctk/wav48 \
    --output data/latents/

# 3. Phase B: Warm-start (~2 hours)
uv run python train_warmstart.py \
    --config configs/phase_b.yaml \
    --data data/latents/ \
    --output checkpoints/phase_b/

# 4. Phase C: Flow matching training (~5-7 days)
uv run python train_flow.py \
    --config configs/phase_c.yaml \
    --data data/latents/ \
    --output checkpoints/phase_c/

# 5. Export
uv run python export_weights.py \
    --checkpoint checkpoints/phase_c/best.pt \
    --output ../models/converter.safetensors

# 6. Test in Rust
cd ..
./target/release/lightvc-app convert \
    -i source.wav -r reference.wav -o converted.wav \
    --dac-weights models/dac_44khz.safetensors \
    --converter-weights models/converter.safetensors
```

---

## Why This Beats Teacher Distillation

| Criterion | Teacher Distillation | Direct Flow Matching |
|-----------|---------------------|---------------------|
| Quality ceiling | Teacher's quality | Data + architecture (higher) |
| Teacher artifacts | Inherits teacher's quirks | None |
| License contamination | Teacher license applies | Clean |
| Phase A time | ~7 days (A100) | ~2 hours (B580) |
| Novelty | Incremental | Progressive RVQ-depth FM is new |
| SOTA potential | Capped at teacher | Uncapped |

---

## References (Teacher-Free VC Training)

| Paper | arXiv | Paradigm | Key contribution |
|-------|-------|----------|------------------|
| AutoVC | 1905.05879 | Bottleneck | Information-bottleneck VC |
| VQMIVC | 2106.10132 | MI disentangle | VQ + mutual information |
| Diff-HierVC | 2311.04693 | Flow matching | Hierarchical diffusion VC |
| CoDiff-VC | 2411.18918 | Codec + diffusion | Codec-assisted dual-CFG |
| Seed-VC | 2411.09943 | In-context + FM | Timbre shifter aug (NOT a teacher) |
| R-VC | 2506.01014 | SSL + shortcut FM | 2-step, HuBERT content |
| EZ-VC | 2505.16691 | SSL + NAR FM | "Purely self-supervised" |
| REF-VC | 2508.04996 | SSL + random erase | Matches Seed-VC from scratch |
| MeanVC 2 | 2606.09050 | FRC + UTTE | Future-receptive chunking + universal timbre tokens (LightVC adopts FRC/UTTE; the flow formulation is rectified FM, not mean-flow — see §C.2) |
| DiFlow-TTS | 2509.09631 | Discrete FM, factorized heads | Progressive depth pattern |
