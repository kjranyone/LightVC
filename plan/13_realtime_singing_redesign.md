# Plan 13: Realtime × Singing VC Redesign

> Critical redesign of LightVC for free-conversation zero-shot VC and singing,
> addressing the structured generator-error collapse discovered in Phase 3.
>
> Status: revised after T0 tolerance sweep and empirical generator-error diagnostic.
> Supersedes: Phase 3 generator-only attempts in plan/12.
> Does NOT supersede: plan/12 Phase 1b-2e results (these are the evidence base).

---

## 0. Problem Statement

### 0.1 The mathematical object

DAC defines:

```
E: Audio → R^{1024×T}           (encoder)
Q: R^{1024×T} → M_q ⊂ R^{1024×T} (RVQ: 9 codebooks, each 1024×8, factorized)
D: M_q → Audio                   (decoder)
```

M_q is discrete: at most 1024^9 possible frame-level quantizations (far fewer are
speech-valid). The decoder D was trained only on M_q inputs from real speech.

### 0.2 What failed and why (revised unified view)

Every Phase 3 attempt tried to produce z_t_like ∈ R^{1024×T} that approximates
a target encoder output, then hard-quantize it:

```
z_t_like → (z_t_like - q0_s) → RVQ residual chain → z_q → D(z_q)
```

The original hypothesis was that the RVQ argmin chain was the primary failure:
small generator noise would flip early codes and cascade into catastrophic
speaker loss. T0 falsified this as the main explanation.

**T0 result**: adding random Gaussian noise to the exact target latent does not
break the path. Even at σ=0.5, hard RVQ SECS changes only about `0.664 → 0.659`.
Soft RVQ is consistently slightly better (`τ=5`: about `0.698`), but it is not
a rescue mechanism for the failed generator.

**Empirical generator-error result**: the trained embedding generator is wrong
in the speaker-bearing residual directions:

| depth | cosine | code acc | oracle-code rank |
|-------|--------|----------|------------------|
| d1 | 0.322 | 0.6% | 319 / 1024 |
| d2 | 0.257 | 0.8% | 327 / 1024 |
| d3-d8 | 0.047-0.165 | ~random | 381-472 / 1024 |

The decoded model output is closer to the source than to the target:

| condition | SECS target | SECS source | target-source |
|-----------|-------------|-------------|---------------|
| oracle | 0.664 | 0.191 | +0.473 |
| model hard | 0.041 | 0.126 | -0.086 |
| model soft τ=5 | 0.069 | 0.231 | -0.162 |

Therefore the dominant Phase 3 failure is **structured generator error**:
latent-domain MSE/cosine/CE learns a codebook-centroid or source-like solution
that does not encode target speaker directions. The decoder is tolerant; the
generator objective is wrong.

### 0.3 The oracle vs generator gap

```
Oracle (exact z_t):     z_t_aligned → RVQ → D → SECS 0.686
Generator (≈67% cos):   z_t_like   → RVQ → D → SECS 0.03
```

The 0.656 gap is not "the generator needs more capacity." It is the cascade
amplification of the wrong kind of error: structured collapse in speaker
directions, not random local noise around a valid target trajectory.

### 0.4 What this plan solves

Two problems, in order:

1. **Objective**: train the generator through decoded audio so that speaker
   directions are rewarded directly.
2. **Decode path**: use soft RVQ as the default differentiable path and minor
   quality improvement. Keep decoder adapter as a fallback, not the next step.

---

## 1. Decoder Tolerance Findings

### 1.1 Design space

Five approaches, ordered by invasiveness:

| ID | Approach | What changes | Frozen DAC? | New params |
|----|----------|-------------|-------------|------------|
| T0 | Noisy latent tolerance sweep | Nothing (diagnostic) | Yes | 0 |
| T1 | Soft RVQ (no hard argmin) | Quantization path | Yes | 0 |
| T2 | Small adapter before decoder | Pre-decoder projection | Yes | ~0.5-2M |
| T3 | Partial decoder fine-tune | Last 2-3 decoder blocks | Partially | ~5-10M unfrozen |
| T4 | VC-aware codec retrain | Encoder + decoder + VC | No | Full retrain |

T0 and the empirical diagnostic are complete enough to reprioritize the plan:
T1 soft RVQ remains useful, but T2 adapter is no longer the next experiment.
T3/T4 remain fallback paths only if audio-domain generator training fails with
a valid soft-RVQ decode path.

### 1.1.1 Completed diagnostics

**T0 noisy latent tolerance sweep**:

- Hard RVQ is robust to random noise up to σ=0.5.
- Soft RVQ improves exact/oracle decoding by about +0.01 to +0.03 SECS.
- Decoder adapter is not indicated by the current evidence.

**Empirical generator-error diagnostic**:

- The failed Phase 3 checkpoint predicts depth embeddings with random-level code
  accuracy across all depths.
- Soft RVQ does not rescue the model prediction; it increases source similarity
  more than target similarity.
- The failure mode is target-speaker collapse, not decoder intolerance.

### 1.2 Experiment T0: Noisy Latent Tolerance Sweep (DIAGNOSTIC — do first)

**Purpose**: characterize where the sensitivity lives. Without this, all
downstream choices are blind.

**Data**: existing 200 eval pairs in `data/phase3/eval/`.

**Procedure**:

For each eval pair, for each σ ∈ {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}:

```python
ε ~ N(0, I)  in R^{1024×T}
z_noisy = z_t_aligned + σ * ε * (||z_t_aligned||_F / ||ε||_F)

# Path A: direct decode (skip RVQ entirely)
y_A = dac.decoder(z_noisy)

# Path B: hard RVQ re-quantize (current pipeline)
residual = z_noisy - q0_s
z_q_B = q0_s.clone()
for d in 1..8:
    q_d, _, _, codes_d, _ = dac.quantizer.quantizers[d](residual)
    z_q_B += q_d
    residual -= q_d
y_B = dac.decoder(z_q_B)

# Path C: soft RVQ (Section 1.3)
z_q_C = soft_rvq_requantize(dac, q0_s, z_noisy, tau)
y_C = dac.decoder(z_q_C)
# Sweep tau ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
```

Measure on each path:
- **SECS**: ECAPA cosine(y, target_timbre)
- **Content**: Whisper CER(y, source_text)
- **UTMOS/DNSMOS**: audio quality
- **F0 corr**: log-F0 Pearson(y, source)

Also measure **code flip rate** per depth for Path B:
```python
codes_clean = hard_rvq_residual(q0_s, z_t_aligned)  # depths 1..8 with q0_s fixed
codes_noisy = hard_rvq_residual(q0_s, z_noisy)      # same residual-chain prefix
flip_rate_d = mean(codes_clean[d] != codes_noisy[d])
```

Sanity checks:
- Path B at `σ=0` should reproduce the known `src_K1` oracle, SECS about `0.686`.
- Path A at `σ=0` should be close to continuous target decode, not quantized target decode.
- Path C with very small `τ` should approach Path B; if it does not, the soft RVQ implementation is wrong.
- Add one empirical-noise condition if a Phase 3 checkpoint exists: `z_noisy = z_t_aligned + (z_pred - z_t_aligned)`. Gaussian noise alone may not match generator error structure.

**Observed outcomes and interpretation**:

| Observation | Interpretation | Next step |
|-------------|---------------|-----------|
| Path A/B/C all remain stable under large random σ | Decoder/RVQ are tolerant to local random perturbation | Do not prioritize adapter |
| Path C is consistently above hard RVQ | Soft RVQ is a useful differentiable decode path | Use T1 in training |
| Empirical generator output fails under hard and soft RVQ | Generator predicts wrong speaker residual directions | Replace latent-domain loss |

**Time**: unknown until benchmarked. The full grid is closer to
`200 pairs × 8 σ × (direct + hard + 5 soft τ)` decode/evaluate passes.
Start with a 50-pair smoke run and SECS/flip-rate only, then run CER/UTMOS on
the shortlisted paths.

**Decision**: T0 is complete. The next step is `train_phase3b.py`: generator
training through soft RVQ and decoded-audio losses. Adapter work is deferred.

### 1.3 Experiment T1: Soft RVQ

**Mathematical formulation**:

Replace hard argmin with differentiable soft assignment:

```
For depth d with codebook cb_d ∈ R^{1024×8}:

z_e = project_in^{(d)}(residual)         # [B, 8, T]
distances = ||z_e^T - cb_d||²             # [B, T, 1024]
weights = softmax(-distances / τ)         # [B, T, 1024]
z_q_soft = (weights @ cb_d)^T             # [B, 8, T]
z_q_1024 = project_out^{(d)}(z_q_soft)   # [B, 1024, T]
residual = residual - z_q_1024
```

τ controls tradeoff:
- τ → 0⁺: approaches hard RVQ (discontinuous, cascade-sensitive)
- τ → ∞: uniform weights → z_q_soft = mean(cb_d) → information loss
- Intermediate τ: smooth, differentiable, cascade-damped

The key property: **no discrete jump**. Small perturbation in z → small
perturbation in weights → small perturbation in z_q_soft → small perturbation
in residual for next depth. The cascade does not amplify.

**Observed risk status**: soft embeddings are acceptable for the current DAC
decoder. T0 shows `τ=1-5` improves the oracle path. However, soft RVQ does not
fix a collapsed generator, so it must be paired with an objective change.

**Implementation**: ~50 LOC in Python. No new parameters. No training needed
(diagnostic). Just change the eval decode path.

**Rust/Candle portability**: softmax + matmul are trivial in Candle. No new
ops. The quantizer loop structure is already in `dac_model.rs:426-513`.

**Go**: use soft RVQ as the differentiable Phase 3b decode path.
**No-Go**: do not use soft RVQ as the sole fix for existing Phase 3 checkpoints.

### 1.4 Experiment T2: Adapter Before Frozen Decoder

**When**: only if Phase 3b audio-domain generator training fails while oracle
soft RVQ remains strong. T0 does not justify building the adapter first.

**Architecture**:

```
z_q_or_soft [B, 1024, T]   # output of hard/soft residual quantization
    ↓
Adapter: Conv1d(1024, 1024, k=3, pad=1) + GELU + Conv1d(1024, 1024, k=3, pad=1)
    ↓
z_adapted [B, 1024, T]
    ↓
frozen DAC decoder
    ↓
audio
```

The adapter is **pre-decoder**, not pre-RVQ. A pre-RVQ adapter would still feed
the hard cascade and may not solve the observed failure. Use a bottleneck
causal Conv1d adapter: 1024 → 256 → 1024, about 1.6M parameters.

**Training data**: same 1800 train pairs. The adapter sees the decoder input
produced by the selected tolerant path (`z_q_hard`, `z_q_soft`, or noised
variants), not raw `z_t_like` before RVQ.

**Loss**:

```
L = α_speaker * (1 - SECS(y, target_timbre))
  + α_content * content_feature_loss(y, source_audio)  # train-only frozen model
  + α_stft * MultiScaleSTFTLoss(y, target_audio)       # same-text/aligned only
  + α_recon * ||Adapter(z_t_exact) - z_t_exact||²  # round-trip preservation
  + α_leak * max(0, cos(ECAPA(y), e_source))       # anti-leakage
```

Critical: **the dominant losses are decoded-audio-domain or decoded-audio
representation losses**, not latent-domain L1. Plain CER from decoded text is
not differentiable. Use a differentiable frozen content model loss
(for example CTC logits / SSL feature cosine) during training, and keep
CER/WER as evaluation metrics. STFT to target audio is valid only for same-text
or aligned supervision; do not apply it to arbitrary cross-text pairs as if
they were parallel. The reconstruction term (α_recon, low weight) prevents the
adapter from destroying round-trip quality.

**Training protocol**:
1. Phase T2a: train adapter on (noised/soft-quantized z_t → audio) with frozen generator=identity.
   Adapter learns to project noisy latents to decoder-valid manifold.
2. Phase T2b: train adapter + generator jointly. Generator produces z_t_like,
   adapter projects it, decoder synthesizes audio, losses are audio-domain.

**Catastrophic forgetting mitigation**:
- DAC decoder is fully frozen. No forgetting possible in decoder.
- Adapter's α_recon term ensures it preserves exact latents when input is clean.
- Monitor: round-trip SECS (encode → adapter → decode vs original) must stay > 0.95.

**Rust/Candle portability**: 2-layer Conv1d is trivially portable. No attention,
no SSM. Direct mapping to `candle-nn::Conv1d`.

**Go**: SECS(adapter) ≥ 0.45 at generator noise, UTMOS ≥ 3.5, round-trip SECS ≥ 0.95
**No-Go**: adapter destroys round-trip quality, or doesn't improve over T1

### 1.5 Experiment T3: Partial Decoder Fine-tune (fallback)

**When**: only if T1 + T2 are both insufficient.

Unfreeze the last 2 decoder blocks of DAC (the final upsampling layers). Keep
encoder, quantizer, and earlier decoder blocks frozen.

**Risk**: catastrophic forgetting — the decoder may lose general audio quality.

**Mitigation**:
- Mix 50% original reconstruction audio + 50% VC audio in each batch
- Learning rate ≤ 1e-5 (100x smaller than adapter)
- EMA weights for rollback safety
- Monitor UTMOS on held-out non-VC audio (LibriTTS test-clean)

**Go**: same as T2, with UTMOS(decode-only audio) drop < 0.1
**No-Go**: UTMOS drop > 0.3 on non-VC audio (catastrophic forgetting)

### 1.6 Experiment T4: VC-Aware Codec (last resort)

**When**: if T0 shows all frozen-DAC paths fail, or T1-T3 all fail.

**Design**: take DAC architecture, add a timbre conditioning module to the
decoder (AdaIN or cross-attention), retrain end-to-end on multi-speaker speech
with VC objective.

This is the VChangeCodec direction. It requires:
- Full codec retraining (~100 GPU-hours on LibriTTS)
- Large dataset (>500 hours multi-speaker)
- Careful curriculum (reconstruction → VC)

**Risk**: the retrained codec may not match DAC's audio quality.
**Rust portability**: same architecture as DAC (already in Rust), just
different weights + conditioning module.

**Migration trigger**: see Section 7.

---

## 2. Free Conversation Zero-Shot VC

### 2.1 Why previous generators failed (root cause)

All Phase 3 generator variants used **latent-domain losses**:

| Variant | Loss | Failure |
|---------|------|---------|
| STE + latent_cos | cosine(z_t_like, z_t_aligned) | target speaker not learned |
| Code CE | cross_entropy(codes_pred, codes_oracle) | random-level code accuracy |
| Embedding MSE | MSE(emb_pred, emb_oracle) | centroid/source-like collapse |

The latent-domain losses optimize a proxy that **does not identify the
speaker-bearing residual directions**. Empirical error analysis shows low
cosine and random-level code accuracy at every residual depth, with decoded
output closer to source than target. The failure is not local decoder
sensitivity; it is the conditional mean/collapse mode:

```
argmin_f E[||f(x) - y||²] = E[y | x]
```

For multi-speaker residual targets, `E[y | x]` is close to a codebook centroid
or generic/source-like residual. That output can be numerically close under a
global proxy while carrying little target identity.

The fix: **train through the tolerant decode path with audio-domain losses**.

### 2.2 Generator architecture: Causal Conformer-lite

```
Inputs:
  z_s [B, 1024, T]    (source DAC latent, content+speaker mixed)
  f0  [B, T, 1]       (normalized log-F0)
  energy [B, T, 1]    (normalized log-RMS)
  timbre [B, 192]     (target ECAPA embedding)
  timbre_tokens [B, 32, 192]  (optional UTTE bank, if enable_timbre)

Backbone:
  proj = Conv1d(1024+1+1, hidden_dim)     # content + prosody streams
  h = proj(cat(z_s, f0, energy))
  h = h + pos_emb

  # Timbre conditioning via FiLM
  gamma, beta = timbre_film(timbre)
  h = gamma * h + beta

  # Causal Conformer blocks × N
  for block in conformer_blocks:
      h = h + block(h)     # each: MHSA(causal) + Conv + FFN

  # Output projection (zero-init)
  delta_z = out_proj(h)                   # [B, 1024, T]
  z_t_like = z_s + delta_z                # residual prediction
```

Parameters: hidden=512, N=6 layers, ~15M params. All causal Conv1d + causal
attention. No future context in Strict mode.

**Why not from scratch (z_t_like = G(features))**: residual prediction
(z_t_like = z_s + Δz) constrains the output to be close to z_s, which is
already near the codec manifold. The generator only needs to modify the
speaker-bearing components, not reconstruct the entire latent.

**Why not Mamba/SSM**: Conformer has stronger content modeling for this frame
rate (86 Hz). Mamba's advantage at long sequences is less relevant at 86 Hz
with chunked streaming. If latency budget is tight, Mamba can be substituted
without architecture change to the loss.

### 2.3 Loss design (the critical section)

```
# Forward pass (training, with tolerant decode path)
z_t_like = generator(z_s, f0, energy, timbre)

# Tolerant decode (soft RVQ by default; adapter only if later justified)
z_q = tolerant_quantize(q0_s, z_t_like)
y = dac.decoder(z_q)               # [B, 1, T_samples]
y_16k = resample(y, 44100, 16000)

# Decoded-audio losses
L_speaker = 1 - cos(ECAPA(y_16k), timbre)           # speaker similarity
L_content = frozen_content_feature_loss(y_16k, source_audio_16k)
L_stft = MultiScaleSTFT(y, target_audio_44k)         # same-text/aligned only
L_leak = max(0, cos(ECAPA(y_16k), e_source_16k) - 0.3)  # anti-leakage

# Latent auxiliary (low weight — regularization only, not primary)
L_cos = 1 - cos(z_t_like, z_t_aligned)               # trajectory guidance

# Total
L = 1.0*L_speaker + 0.5*L_content + 0.3*L_stft + 0.2*L_leak + 0.1*L_cos
```

**Why this avoids each known collapse mode**:

| Collapse mode | Why this loss avoids it |
|---------------|------------------------|
| Median collapse (L1/MSE → E[y\|x]) | L_speaker is cosine, not L1. No median target. |
| Code CE sparsity | No code prediction. Audio-domain loss only. |
| Structured speaker-direction miss | Speaker loss is computed after decode, where the failure is observed. |
| Source leakage | L_leak explicitly penalizes source similarity. |
| Off-manifold | Decoder is in the loop; gradient pushes toward on-manifold outputs. |

**Content loss at training time**: do not use decoded CER as a loss. Use a
differentiable frozen ASR/SSL representation or CTC logits if needed. Whisper
decoding can remain an evaluation metric, not the main backprop path. This is
train-only and not a VC teacher.

**Project-rule note**: using a frozen speaker or content model as a training
loss is a perceptual-loss experiment, not VC-teacher distillation. If the
project interprets "Teacher distillation" as banning all frozen external-model
losses, run Phase 3b as research-only and replace `L_speaker` with a native
speaker metric before productizing.

### 2.4 Training protocol

**Phase 3b-warmup**:
1. Use soft RVQ (`τ≈5` initially, with sweep) as the fixed decode path
2. Train generator on 1800 pairs with decoded-audio losses
3. Evaluate on 200 eval pairs with bootstrap CI

**Phase 3b result**: decoded-audio training reduces source leakage but plateaus
at low target similarity. Across multiple hyperparameter settings, eval
target SECS stays around `0.12-0.14` while source SECS remains higher. This is
not fixed by τ, learning rate, or increasing the latent auxiliary weight.

Gradient audit (`training/phase3b_grad_audit.py`) shows:

- ECAPA speaker loss is differentiable; gradient is not zero.
- The soft-RVQ + decoder path attenuates audio gradients by roughly four orders
  of magnitude before they reach `z_pred`.
- On eval, speaker and anti-leak gradients are partially opposed, so training
  can learn "not source" without learning "specific target".

This means an **unconditioned partial decoder fine-tune is not the preferred
next step**. It may increase gradient magnitude, but the decoder still has no
direct target-speaker conditioning except whatever weak signal the generator
already failed to encode.

**Phase 3c recommendation**: add a small **timbre-conditioned pre-decoder
adapter** before unfreezing DAC decoder blocks:

```
z_q_soft
  ↓
Adapter(z_q_soft, target_timbre) = z_q_soft + ConvFiLM(z_q_soft, target_timbre)
  ↓
frozen DAC decoder
```

Train the adapter first with the generator frozen or using oracle/soft-RVQ
inputs, then jointly train generator + adapter if target-source margin becomes
positive. This creates a direct, low-latency target-speaker path while keeping
the DAC decoder frozen. Partial decoder fine-tune becomes the fallback only if
the conditioned adapter cannot reach positive target-source margin.

**Phase 3c full-eval result** (`checkpoints/phase3c/best.pt`, 200 eval pairs):

| metric | mean | 95% CI |
|--------|------|--------|
| target SECS | 0.378 | [0.357, 0.399] |
| source SECS | 0.426 | [0.408, 0.442] |
| margin | -0.048 | [-0.078, -0.018] |
| oracle target SECS | 0.595 | [0.576, 0.615] |
| oracle margin | +0.401 | [0.373, 0.428] |

The adapter is a real improvement over Phase 3b (`0.14 → 0.38`), but it still
does not cross the speaker-disentanglement line: source similarity remains
significantly higher than target similarity on the full 200-pair evaluation.
The 50-pair/best-epoch value was mildly optimistic.

**Next decision**: do not scale this exact setup blindly. The next experiment
must separate:

1. **capacity/data overfit**: train margin is strongly positive while eval
   margin is negative.
2. **source leakage objective weakness**: eval source SECS remains high.
3. **adapter-only vs generator+adapter**: determine whether the generator is
   helping or just overfitting the train pairs.

Run `adapter_only` and stronger leak-weight ablations before partial decoder
fine-tune or data expansion.

**Phase 3c-B1 result**: replacing FiLM-only conditioning with ECAPA-token
cross-attention is the current winning design.

| metric | A0 FiLM | B1 ECAPA-token cross-attn | oracle |
|--------|---------|---------------------------|--------|
| target SECS | 0.420 [0.403, 0.436] | 0.508 [0.493, 0.522] | 0.641 |
| source SECS | 0.311 [0.291, 0.335] | 0.238 [0.220, 0.259] | 0.108 |
| margin | +0.108 [0.080, 0.136] | +0.269 [0.242, 0.294] | +0.533 |
| oracle ratio | 0.657 | 0.792 | 1.000 |

The bottleneck was the FiLM conditioning mechanism, not ECAPA information
capacity. Expanding the same ECAPA 1-vector into 32 timbre tokens and applying
frame-level cross-attention raises target similarity and reduces source
leakage simultaneously.

**Latency estimate** (`training/bench_phase3c_adapter.py`, CUDA, checkpoint
`phase3c_ao_b1_ecapa/best.pt`):

| frames | mean adapter time |
|--------|-------------------|
| 1 | 0.337 ms |
| 2 | 0.353 ms |
| 8 | 0.365 ms |
| 32 | 0.346 ms |
| 128 | 0.325 ms |
| 256 | 0.366 ms |

The adapter is not the latency bottleneck. For real-time deployment, the
dominant terms remain DAC encode/decode, soft RVQ, resampling, and audio I/O.
Rust/Candle implementation should preserve this architecture directly:

```
z_q [B,1024,T]
  -> Conv1d(1024,256,k=3,pad=1)
  -> FiLM(gamma,beta from ECAPA)
  -> ECAPA MLP -> 32 tokens [B,32,256]
  -> MHA(query=frames, key/value=tokens, heads=4)
  -> GELU
  -> Conv1d(256,1024,k=3,pad=1)
  -> residual add
```

Do not reuse the older Rust `CrossAttnBlock` verbatim without checking weight
layout: the existing block has separate q/k/v/o projections and LayerNorm,
while the PyTorch B1 adapter uses `nn.MultiheadAttention` inside the bottleneck
stream and no adapter LayerNorm. The export/import path must preserve exact key
names and tensor transposes.

**Rust/Candle implementation status**:

| artifact | path | status |
|----------|------|--------|
| B1 weight export | `training/export_b1_adapter.py` → `models/utte_adapter_b1.safetensors` | done |
| UTTE adapter module | `crates/lightvc-core/src/utte_adapter.rs` | done |
| Rust microbench | `crates/lightvc-core/examples/bench_utte.rs` | done |

Rust CPU adapter benchmark:

| frames | mean |
|--------|------|
| 1 | 0.53 ms |
| 2 | 1.09 ms |
| 8 | 1.70 ms |
| 32 | 3.04 ms |
| 128 | 5.60 ms |
| 256 | 8.26 ms |

Adapter overhead is well below the 50 ms budget. The next implementation risk is
not UTTE; it is integrating soft RVQ and DAC encode/decode into a streaming
pipeline without adding avoidable buffer copies.

**Next implementation order**:

1. Implement/benchmark Rust soft-RVQ residual chain (`q0_source + depths 1-8`).
2. Build an offline Rust parity path: `z_s + q0_s + ECAPA -> soft RVQ -> UTTE adapter -> DAC decoder`.
3. Compare Rust output with Python output on a fixed eval pair (latent MSE and waveform/STFT sanity).
4. Benchmark full offline path by component: soft RVQ, adapter, DAC decode.
5. Only then wire into strict real-time audio callback.

**Soft RVQ + adapter implementation status**:

| artifact | path | status |
|----------|------|--------|
| Soft RVQ | `crates/lightvc-core/src/soft_rvq.rs` | parity MSE = 0 |
| Python reference | `training/generate_rust_parity_ref.py` | done |
| Parity test | `crates/lightvc-core/examples/parity_test.rs` | pass |
| Component bench | `crates/lightvc-core/examples/bench_pipeline.rs` | done |

Rust CPU component benchmark:

| frames | soft RVQ | UTTE adapter | soft RVQ + adapter |
|--------|----------|--------------|--------------------|
| 1 | 0.23 ms | 0.57 ms | 0.90 ms |
| 8 | 1.00 ms | 1.17 ms | 2.72 ms |
| 128 | 16.0 ms | 4.7 ms | 22.9 ms |

Soft RVQ + UTTE are not the latency bottleneck.

**DAC encode/decode benchmark** (`cargo run --release -p lightvc-core --example bench_dac`, CPU):

| frames | encode | decode | encode+decode |
|--------|--------|--------|---------------|
| 1 | 18.4 ms | n/a | n/a |
| 2 | 25.0 ms | n/a | n/a |
| 8 | 83.8 ms | 184.2 ms | 266.2 ms |
| 16 | n/a | 322.1 ms | 491.4 ms |
| 32 | 296.4 ms | 677.0 ms | 1000.8 ms |
| 64 | n/a | 1427.2 ms | 2133.7 ms |
| 128 | 1539.6 ms | 3083.0 ms | 4666.8 ms |

The current CPU DAC path is not real-time. Strict sub-50 ms with the frozen
DAC architecture requires GPU acceleration, a faster decoder implementation,
or migration to a causal/lighter codec. Also, direct `decoder.forward()` on a
single latent frame is invalid in the current Candle decoder path; streaming
decode must use a larger decode window with overlap-add or a causal decoder.

**CUDA DAC encode/decode benchmark** (RTX 2080 Ti, 22GB):

| frames | encode | decode | encode+decode |
|--------|--------|--------|---------------|
| 1 | 2.0 ms | 4.4 ms | 6.0 ms |
| 2 | 2.3 ms | 6.9 ms | n/a |
| 4 | 2.9 ms | 9.6 ms | 11.8 ms |
| 8 | 4.1 ms | 13.6 ms | 16.9 ms |
| 32 | 8.8 ms | 42.1 ms | 51.7 ms |
| 128 | 35.4 ms | 180.4 ms | n/a |

CUDA changes the deployment conclusion: frozen DAC is too slow on CPU, but
sub-50 ms is feasible on CUDA.

| component | Strict 1f | Balanced 8f |
|-----------|-----------|-------------|
| DAC encode | 2.0 ms | 4.1 ms |
| Soft RVQ + UTTE adapter | ~0.5 ms | ~2.7 ms |
| DAC decode | 4.4 ms | 13.6 ms |
| VC processing subtotal | ~6.9 ms | ~20.4 ms |
| Resampling | ~6 ms | ~6 ms |
| Audio I/O | ~6 ms | ~6 ms |
| Estimated total | ~18.9 ms | ~32.4 ms |

Both modes fit the 50 ms target on CUDA. The next step is no longer codec
migration; it is CUDA full-pipeline parity and streaming integration.

**Updated implementation order**:

1. Run Soft RVQ + UTTE adapter benchmark on CUDA, not only CPU.
2. Run CUDA full-pipeline parity:
   `DAC encode -> q0/soft RVQ -> UTTE adapter -> DAC decode`.
3. Implement streaming mode with two quality/latency modes:
   - Strict: 1 latent frame hop, minimum latency.
   - Balanced: 8 latent frame window/hop or 8-frame decode window with overlap/crop.
4. Measure end-to-end callback latency with ASIO/CoreAudio.
5. Only revisit causal/lighter codec if CUDA end-to-end latency or quality fails.

**CUDA full-pipeline parity status**:

| component | Rust/Candle | Python/PyTorch parity | CUDA latency |
|-----------|-------------|-----------------------|--------------|
| Soft RVQ | done | MSE = 0 | 0.2-1.0 ms |
| UTTE adapter | done | MSE ~= 0 | 0.3-0.7 ms |
| DAC encode | existing | n/a | 2.0-4.1 ms |
| DAC decode | existing | MSE = 0 | 4.4-13.6 ms |
| Full pipeline | done | pass | 6.9-19.4 ms |

Full CUDA parity passed:

```text
Soft RVQ MSE:  0.00000000
Adapter MSE:   0.00000005
Decode MSE:    0.00000000
```

The frozen DAC + B1 adapter path is now implementation-validated. The next
task is streaming integration, not further offline architecture work.

**Streaming integration plan**:

1. Add a new B1 pipeline path:

```text
encode_step
  -> q0 extraction/cache
  -> soft RVQ residual chain
  -> UTTE adapter
  -> decode_step
```

2. Implement two runtime modes:

- **Strict**: 1 latent-frame hop. Target latency around 19 ms on CUDA.
- **Balanced**: 8 latent-frame window. Target latency around 31 ms on CUDA,
  likely better boundary quality.

3. Handle decoder boundary quality explicitly:

- keep decoder history/window separate from encoder history;
- crop the valid center or newest hop from the decoded window;
- overlap-add with the existing streaming crossfade path.

4. Add benchmarks:

- component timing inside the pipeline;
- callback processing time distribution p50/p95/p99;
- underrun counter;
- strict vs balanced audio artifact check.

**Phase 3b-joint** (only if an adapter is later justified):
1. Initialize from warmup checkpoint
2. Joint fine-tune with lower LR (1e-5)
3. Monitor for adapter quality regression

**Data sufficiency**: 1800 pairs × ~400 frames = 720K frames. With audio-domain
losses (not per-class prediction), this is comparable to other VC systems
that train on ~100K utterances (SynthVC uses 500K synthetic pairs; MeanVC2
uses LibriTTS 585h ≈ ~300K utterances). We may need to expand to all VCTK
same-text pairs (~20K) if convergence is slow.

### 2.5 Content encoder roadmap (Phase 4)

Current generator takes z_s directly. z_s contains both content and speaker.
The generator must learn to separate them implicitly. This works if the
timbre conditioning is strong enough to override source speaker.

**Risk**: source leakage. Phase 1b showed depth 0 has −0.44 SECS contribution.
If the generator doesn't fully replace depth-0-level speaker info, leakage
persists.

**Phase 4 plan (not this experiment)**: train a lightweight causal content
encoder (~5M params) that takes z_s and outputs a speaker-invariant content
representation. Training signal: phoneme classification (MFA labels on VCTK) +
adversarial speaker classifier gradient reversal. This is distillation of the
content disentanglement that Wav2Vec2 provides implicitly, but at <5M params
and fully causal.

This does NOT violate "no Wav2Vec2 at runtime" — the content encoder is a
new, small model trained from scratch.

---

## 3. Singing Mode Design

### 3.1 Why singing ≠ speech + more data

| Dimension | Speech | Singing |
|-----------|--------|---------|
| F0 range | 80-350 Hz | 80-1200+ Hz |
| F0 precision needed | ±5 Hz tolerable | ±1 Hz audible error |
| Vibrato | occasional, subtle | essential, 4-7 Hz oscillation |
| Duration per phoneme | 50-300 ms | 100 ms - 3+ sec (sustained) |
| Dynamic range | ~20 dB | ~50 dB |
| Phonation | modal | modal/breathy/twang/belt |
| Breath noise | natural | stylized, sometimes amplified |
| Training data | VCTK, LibriTTS (abundant) | NUS-48E, M4Singer (scarce) |

A speech-only model applied to singing will fail on:
1. Sustained vowels (training distribution has few long steady states)
2. Pitch jumps (training distribution has slow F0 contours)
3. High pitches (formant structure differs at high F0)
4. Vibrato (F0 modulation pattern unseen in speech)

### 3.2 Mode architecture: shared backbone + mode-specific modules

**Recommendation: unified model with mode token + singing-specific heads.**

Not two separate models. The shared components ensure consistent timbre
identity across modes. The mode-specific components handle the acoustic
differences.

```
Generator:
  shared:
    content_proj(z_s)          # same for speech/singing
    conformer backbone         # same
    timbre conditioning        # same (UTTE cross-attention)

  mode token:
    emb_mode(speech | singing) # learned embedding, added to hidden states

  singing-specific (activated only when mode=singing):
    f0_enhancer:               # deeper F0 processing
      Conv1d(1, 32) + Conv1d(32, 32) + Conv1d(32, hidden_dim)
      Input: detailed F0 contour (including vibrato, not just smoothed)
      Output: additive F0 conditioning to hidden states

    vibrato_tracker:           # detects and preserves vibrato
      autocorrelation-based vibrato frequency/amplitude extraction
      feeds vibrato parameters as conditioning

  output:
    z_t_like = z_s + delta_z   # same output format
```

### 3.3 F0 handling strategy

**Speech mode**:
- Source F0 preserved with minor formant adaptation
- F0 is auxiliary information, not primary control
- Generator can adjust F0 slightly for naturalness

**Singing mode**:
- Source F0 (melody) is PRIMARY — must be preserved exactly
- Pitch deviations > 5 cents (≈3 Hz at 440 Hz) are audible errors
- Vibrato preserved as explicit parameter (frequency, depth, rate)
- Target timbre adapts to source F0 range (formant shift if needed)

**F0 conflict resolution**: when target speaker's natural range doesn't cover
the source melody's range, two options:
1. **Clamp + warn**: clamp F0 to target range, flag in UI
2. **Formant-only mode**: keep source F0, only convert formant/timbre
   (useful for gender-cross conversion with large range differences)

### 3.4 NSF/DDSP auxiliary path

NSF (Neural Source-Filter) and DDSP (Differentiable DSP) generate audio from
explicit F0 + harmonic/noise parameters. They are:
- Lightweight (<5M params)
- Fast (<5ms per chunk)
- Explicitly F0-driven (pitch accuracy guaranteed)
- Compatible with codec-space VC (additive, not replacing)

**Integration for singing mode**:

```
y_codec = DAC_decode(z_q)                     # codec path: timbre + content
y_harm = NSF(f0_source, y_codec_features)     # F0 path: pitch precision
y_final = y_codec + α * (y_harm - y_codec)   # blend, α controlled by mode

  speech: α ≈ 0 (codec only)
  singing: α ≈ 0.3-0.5 (codec + NSF harmonics)
```

**Why this doesn't conflict with codec-space CONCEPT**:
1. NSF operates in audio domain, not codec domain
2. NSF is conditioned on codec output (not independent generation)
3. NSF is optional (α=0 disables it)
4. NSF adds pitch precision that codec quantization may blur

**Alternative**: instead of additive blend, NSF can replace the DAC decoder
for singing mode only. The z_q → NSF(F0, z_q) path generates audio with
explicit pitch control. This is closer to HQ-SVC's approach.

**Rust portability**: NSF is Conv1d + sinusoidal + noise generator. All
trivial in Candle. DDSP is additive synthesis (FFT bins), also portable.

### 3.5 Singing-specific losses

```
# In addition to speech losses:
L_pitch = RMSE(F0(y), F0_source)              # pitch accuracy (cents)
L_vibrato = ||vibrato_params(y) - vibrato_params(source)||²
L_sustain = spectral_stability(y, on_sustained_frames)  # no jitter on long notes
L_breath = breath_detection_loss(y)            # natural breath insertion
```

### 3.6 Training data for singing

| Dataset | Hours | Type | License |
|---------|-------|------|---------|
| NUS-48E | ~2h | Sung + spoken pairs | CC-BY-4.0 |
| M4Singer | ~70h | Singing only | CC-BY-NC |
| Opencpop | ~5h | Singing | CC-BY-NC |
| NHSS | ~10h | Sung + spoken pairs | MIT |

**Minimum viable**: NUS-48E (12 speakers, sung + spoken) for paired training.
NHSS for larger-scale. M4Singer/Opencpop for diversity (NC license means
training-only, not redistribution).

**Curriculum**:
1. Pretrain on speech (VCTK + LibriTTS)
2. Fine-tune on singing with mode token
3. Joint training on mixed batches (speech 60% / singing 40%)

---

## 4. 50ms Latency Budget

### 4.1 Component-by-component estimate

```
Component                           | Strict (ASIO) | Strict (WASAPI) | Balanced
------------------------------------|---------------|-----------------|----------
Capture buffer (cpal)               | 3ms           | 10ms            | 10ms
Resample (device SR → 44100)        | 3ms           | 3ms             | 3ms
DAC encode (1 frame, 512 samples)   | 3ms           | 3ms             | 3ms
Content/F0/energy extraction        | 1ms           | 1ms             | 1ms
Target timbre conditioning          | <1ms          | <1ms            | <1ms
Generator forward (15M, 86fps)      | TBD           | TBD             | TBD
Soft RVQ / adapter                  | <1ms          | <1ms            | <1ms
DAC decode (1 frame)                | 3ms           | 3ms             | 3ms
Resample (44100 → device SR)        | 3ms           | 3ms             | 3ms
Playback buffer (cpal)              | 3ms           | 10ms            | 10ms
------------------------------------|---------------|-----------------|----------
Algorithmic delay (chunk + lookahead)| 12ms         | 12ms            | 93ms
------------------------------------|---------------|-----------------|----------
TOTAL (excluding generator TBD)     | ~34ms         | ~48ms           | ~130ms
```

Notes:
- Strict mode = 1 frame per chunk (512 samples ≈ 11.6ms), 0ms lookahead.
  Algorithmic latency = chunk size = 11.6ms.
- Balanced mode = 4 frames per chunk (2048 samples ≈ 46ms), 4 frames lookahead.
  Algorithmic latency = 93ms.
- Generator time is not measured for the proposed Conformer-lite. Do not claim
  the 35ms/49ms totals as achieved until a Rust/Candle benchmark exists.
- Existing Rust code currently uses the historical converter, not the proposed
  residual generator / adapter.
- DAC encode/decode time is from Rust/Candle benchmarks on CPU (single frame,
  streaming mode, conv-state cached). GPU is ~2x faster.

### 4.2 What dominates

In Strict mode, no single component dominates. The budget is spread across:
- Audio I/O: 6ms (ASIO) or 20ms (WASAPI)
- Algorithmic: 12ms
- Compute: ~10ms + generator TBD (encode + generator + decode)
- Resampling: 6ms

In Balanced mode, the algorithmic delay (93ms) dominates everything.

### 4.3 Sub-50ms feasibility

| Configuration | Total | Sub-50ms? | Quality |
|---------------|-------|-----------|---------|
| Strict + ASIO + GPU | ~35ms | YES | Boundary artifacts from 1-frame chunks |
| Strict + ASIO + CPU | ~37ms | YES | Same caveat |
| Strict + WASAPI + GPU | ~49ms | BORDERLINE | Same caveat |
| Strict + WASAPI + CPU | ~51ms | NO | — |
| Balanced (any) | ~130ms | NO | Good quality |

**Conclusion**: Sub-50ms is plausible only in Strict mode with low-latency
audio backend (ASIO on Windows, CoreAudio on macOS), but it is not yet proven
for the proposed generator/adapter path. The quality cost is boundary artifacts
from single-frame processing.

**Path to high-quality sub-50ms**:
1. Short-term: Strict mode, accept boundary artifacts. The overlap-add in
   `streaming.rs:265` uses 1 hop (512 samples ≈ 11.6ms) crossfade, which
   may be sufficient.
2. Medium-term: implement 2-frame Strict (23ms algorithmic), still sub-50ms
   with ASIO. Better quality at 2 frames per chunk.
3. Long-term: causal codec (Mimi at 24kHz, 12.5Hz, causal). Eliminates
   algorithmic delay from lookahead. Requires accepting 24kHz output.

**The dominant known barrier is DAC's non-causal architecture**. The generator
may still become a compute bottleneck until measured.

### 4.4 Converter context discrepancy

Current Rust code uses 128/192/256 latent frames for converter left-context
(Strict/Balanced/Quality). This may be larger than needed for the historical
converter, but do **not** blindly reduce it: `pipeline.rs` documents a much
larger historical receptive-field estimate. For the new Conformer-lite/adapter
path, compute the exact receptive field from its actual layers and then set
context accordingly. This is a benchmark/verification task, not a design
assumption.

---

## 5. Technology Alignment

### 5.1 Per-system analysis

| System | Key idea | LightVC fit | How to use |
|--------|----------|-------------|------------|
| **X-VC** | codec-space one-step VC with dual conditioning | CONCEPT MATCH | Validates codec-space approach. Their generated-paired-data training is forbidden; their adaptive normalization is worth studying. |
| **MeanVC2** | FRC + UTTE + one-step philosophy | PARTIAL | FRC-style lookahead exists in `streaming.rs`; UTTE/cross-attention exists in the historical converter. The "mean-flow" wording is historical/misleading for current code. Do not treat MeanVC2 as fully integrated into the new generator/adapter path. |
| **StreamVC** | SoundStream codec + causal soft units | PARTIAL | Validates that a lightweight learned content unit encoder is needed. Cannot use their codec (proprietary). Their causal unit concept → Phase 4 content encoder. |
| **RT-VC** | articulatory feature space + causal | PARTIAL | Validates articulatory/phonetic content representation. Their approach is fully end-to-end (no codec). We take the content encoder lesson only. |
| **VChangeCodec** | VC integrated into codec, <1M params, ~40ms | FALLBACK | Migration target if frozen DAC fails. Their 40ms claim validates sub-50ms feasibility for codec-integrated VC. Would require full codec retrain. |
| **YingMusic-SVC** | F0-aware timbre adaptor for singing | SINGING FIT | F0-aware conditioning is directly applicable to singing mode. Their adaptor concept aligns with our NSF auxiliary path. |
| **HQ-SVC** | decoupled codec features + DDSP refinement | SINGING FIT | DDSP refinement after codec decode is compatible with our NSF auxiliary path. Their pitch/volume modeling is useful. |
| **R2-SVC** | NSF + robustness training + singing style | SINGING FIT | NSF modeling validates explicit F0 path. Robustness training (noise/reverb augmentation) is useful for real-world singing. |
| **SVCC 2025** | singing VC evaluation benchmarks | EVALUATION | Use their metrics: pitch RMSE, CER on lyrics, SECS, naturalness MOS. Their test set for benchmarking. |
| **DiFlow-TTS** | discrete flow matching + factorized heads | CONCEPTUAL | Factorized depth-wise prediction is conceptually aligned with our RVQ depth separation, but we are not doing discrete flow matching. |
| **R-VC** | shortcut flow matching (2-step) | CONCEPTUAL | 2-step is still multi-step. LightVC wants one-step/low-step latency, but current code is not a validated mean-flow implementation for the new path. Content token deduplication concept is interesting for Phase 4. |

### 5.2 What LightVC should NOT adopt

| Tempting direction | Why not |
|---------------------|---------|
| Seed-VC style in-context DiT | Too heavy for <50ms. Requires large DiT forward. |
| Multi-step diffusion/flow for VC | Each step adds latency. Conflicts with Strict mode budget. |
| HuBERT content features at runtime | Explicitly forbidden by project rules. |
| FACodec factorized tokens | FACodec is not MIT-licensed (NaturalSpeech is CC-BY-NC). Cannot ship. |
| BigVGAN vocoder | Heavy, not codec-space, conflicts with CONCEPT. |
| ASR-based content loss at runtime | Whisper at runtime is too heavy. Training-only is fine. |

---

## 6. Answers to Key Questions

### Q1: 次の一手は decoder adapter / tolerant decoder でよいか？

**No. The next step is generator retraining with decoded-audio loss.**

T0 and empirical-error diagnostics have changed the answer:

- Frozen DAC/RVQ is tolerant to random noise.
- Soft RVQ is a useful differentiable decode path and small quality boost.
- Existing generator output is structurally wrong and source-like.
- Adapter work is deferred unless audio-domain training fails despite strong
  soft-RVQ oracle performance.

### Q2: その場合、最小実験は何か？

**Phase 3b: soft-RVQ decoded-audio generator training.**

Outputs:
1. Train `TLG` through soft RVQ + DAC decoder
2. Optimize speaker similarity after decode, source anti-leakage, same-text
   STFT-to-oracle, and low-weight latent guidance
3. Evaluate target/source SECS margin and CER on 200 pairs

The minimum smoke run is 50 pairs / 1-3 epochs to check gradient flow and
whether target-source margin becomes positive.

### Q3: 自由会話ゼロショット VC に必要な generator はどの形か？

**Causal Conformer-lite (15M params), residual prediction (z_s + Δz),
audio-domain loss through tolerant decode path.**

The three non-negotiable design choices:

1. **Residual prediction** (z_t_like = z_s + Δz, not from scratch): z_s is
   already near the codec manifold. The generator only modifies the
   speaker-bearing components. This constrains the output space.

2. **Decoded-audio loss** (speaker similarity + anti-leakage + same-text STFT,
   with CER only as evaluation): all Phase 3 failures used latent-domain
   losses. The decoded loss sees the actual failure surface.

3. **Cross-attention timbre** (UTTE-style, not FiLM-only): FiLM conditioning
   was insufficient in WORLD experiments. Cross-attention to 32 timbre tokens
   provides finer-grained speaker control.

### Q4: 歌唱対応は同一モデルか mode分離か？

**Shared backbone + mode token + singing-specific F0/style modules.**

Unified, not separate. Rationale:
- Timbre identity must be consistent across speech and singing (user expects
  "my voice" in both modes). Shared backbone + shared timbre encoder ensures this.
- Singing-specific modules (F0 enhancer, vibrato tracker, NSF path) are
  conditionally activated by mode token. Zero overhead in speech mode.
- Separate models would double storage and risk timbre inconsistency.

The mode token is not just an embedding — it **switches the F0 handling
strategy**: speech mode treats F0 as auxiliary; singing mode treats F0 as
primary and adds NSF/DDSP for pitch precision.

### Q5: 50ms未満はどの設計なら可能性があるか？

**Strict mode (1-frame chunk, 0ms lookahead) + ASIO/CoreAudio + GPU: plausibly sub-50ms after benchmarking.**

The existing Rust infrastructure has the right chunking/audio structure, but
the proposed generator/adapter path has not been implemented or benchmarked.
The quality cost is boundary artifacts from single-frame processing, mitigated
by the 1-hop overlap-add already in `streaming.rs`.

**High-quality sub-50ms is NOT achievable with DAC** due to non-causal
architecture requiring lookahead. Two paths to high-quality sub-50ms:

1. **2-frame Strict** (23ms algorithmic): ~48ms with ASIO. Better quality.
   Needs validation that 2-frame chunks are sufficient for overlap-add.
2. **Causal codec** (Mimi at 24kHz): eliminates lookahead entirely. ~25ms total.
   Requires accepting 24kHz output and retraining all experiments on Mimi.

The generator is not expected to dominate, but this must be verified with a
Rust/Candle benchmark. Do not assume `<3ms` until measured.

### Q6: どの実験が失敗したら、frozen DAC を諦めて VC-aware codec へ移るべきか？

**Specific, measurable triggers:**

1. **Phase 3b decoded-audio generator fails**:
   - target-source SECS margin remains ≤ 0 after smoke training
   - depth d1-d2 empirical cosine remains < 0.4
   - best decoded SECS remains < 0.20

   → Architecture/objective still fails to write target speaker residuals.
   Add stronger timbre cross-attention or target-token conditioning before
   touching the decoder.

2. **Soft RVQ oracle regresses**: best oracle SECS < 0.50 or UTMOS drops > 0.5.
   
   → Re-check soft RVQ implementation and temperature. Adapter is not a fix
   for a collapsed generator.

3. **T2 (adapter), if tried, fails**: cannot achieve SECS ≥ 0.45 AND UTMOS ≥ 3.5
   AND round-trip SECS ≥ 0.95 simultaneously.
   
   → Adapter cannot project to decoder manifold without quality loss.
   Try T3 (partial fine-tune).

4. **T3 (partial decoder fine-tune) fails**: UTMOS on non-VC audio drops > 0.3
   (catastrophic forgetting), OR VC SECS still < 0.45.
   
   → **Frozen DAC is unsuitable. Migrate to T4 (VC-aware codec).**

The migration to T4 means: retrain the DAC decoder (or full codec) with VC
conditioning. This is a 100+ GPU-hour effort and should only start after all
four triggers above fire.

---

## 7. Migration Conditions Summary

```
T0 (diagnostic)
  └─ complete: frozen DAC/RVQ tolerant; soft RVQ useful

Empirical generator-error diagnostic
  └─ complete: structured speaker-direction collapse

Phase 3b decoded-audio generator training ★
  ├─ target-source margin > 0 and SECS rising → continue scaling
  ├─ margin ≤ 0 but oracle remains strong → improve generator/timbre conditioning
  └─ oracle degrades or decoder quality fails → adapter/T3/T4 fallback

★ = proceed to generator training (Section 2) + singing mode (Section 3)
✗ = frozen DAC abandoned, start codec retrain
```

---

## 8. Immediate Action Plan

### Step 1: T0 Noisy Latent Tolerance Sweep

- Write `training/phase3b_tolerance_sweep.py`
- Run on 200 eval pairs, 8 σ levels, 3 paths
- Output: `results/phase3b_tolerance.json` with SECS/UTMOS/CER per path per σ
- **Status**: complete enough to proceed. Frozen DAC/RVQ is tolerant to random
  noise. Use soft RVQ for Phase 3b.

### Step 2: Empirical Generator Error Diagnostic

- Write `training/phase3b_empirical_error.py`
- Re-run existing embedding-MSE model checkpoint with hard and soft RVQ
- **Status**: complete. Existing generator is wrong in speaker residual
  directions and closer to source than target.

### Step 3: Generator retraining with decoded-audio loss (3-5 days)

- Write `training/train_phase3b.py` with the loss design from Section 2.3
- Use soft RVQ as the fixed differentiable decode path
- Train on 1800 pairs for 50 epochs
- Evaluate with bootstrap CI on 200 pairs

### Step 4: Singing mode oracle (after Step 3 reaches Go)

- Prepare NUS-48E singing pairs
- Test same-pair singing oracle with soft RVQ
- Measure pitch RMSE, vibrato preservation, SECS

---

## Appendix A: Soft RVQ Implementation Reference

```python
def soft_rvq_requantize(dac, q0_source, z_t_like, tau=1.0):
    """
    Differentiable soft RVQ that avoids hard argmin cascade.

    q0_source: [B, 1024, T]  — fixed source depth-0 contribution
    z_t_like:  [B, 1024, T]  — predicted target-like latent
    tau:       temperature parameter

    Returns: z_q [B, 1024, T] — soft-quantized latent for decoder
    """
    z_q_sum = q0_source.clone()
    residual = z_t_like - q0_source

    for d in range(1, 9):
        quantizer = dac.quantizer.quantizers[d]

        # Project to codebook space (1024 → 8)
        z_e = quantizer.project_in(residual)       # [B, 8, T]

        # Compute squared distances to all codes
        cb = quantizer.codebook.weight             # [1024, 8]
        # z_e: [B, 8, T] → [B, T, 8]
        z_e_t = z_e.transpose(1, 2)
        dist = torch.cdist(z_e_t, cb.unsqueeze(0)).pow(2) # [B, T, 1024]

        # Soft assignment
        weights = F.softmax(-dist / tau, dim=-1)   # [B, T, 1024]

        # Soft embedding (weighted average of codebook)
        z_q_soft = weights @ cb                     # [B, T, 8]
        z_q_soft = z_q_soft.transpose(1, 2)         # [B, 8, T]

        # Project back to latent space (8 → 1024)
        z_q_1024 = quantizer.project_out(z_q_soft)  # [B, 1024, T]

        residual = residual - z_q_1024
        z_q_sum = z_q_sum + z_q_1024

    return z_q_sum
```

## Appendix B: Latency Budget Constants (from Rust codebase)

| Constant | Value | Source |
|----------|-------|--------|
| DAC_SAMPLE_RATE | 44100 | `lib.rs:20` |
| DAC_HOP_LENGTH | 512 | `lib.rs:23` |
| DAC_FRAME_RATE | 86.13 Hz | `lib.rs:26` |
| DAC_LATENT_DIM | 1024 | `lib.rs:29` |
| ENCODER_OVERLAP | 2048 (4 hops) | `streaming.rs:31` |
| Strict chunk | 512 samples (1 frame) | `streaming.rs:34-49` |
| Balanced chunk | 2048 samples (4 frames) | `streaming.rs:50-64` |
| Quality chunk | 4096 samples (8 frames) | `streaming.rs:65-79` |
| Decode crossfade | 512 samples (1 hop) | `streaming.rs:265` |
| Converter context | 128/192/256 frames (over-sized, should be 32/48/64) | `pipeline.rs:39-41` |
