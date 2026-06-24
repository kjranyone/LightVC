# LightVC Trial Report for Paper Draft

Date: 2026-06-24

This report summarizes the research path that led to the current LightVC design:

```text
source q0 anchor
  + soft residual RVQ
  + ECAPA-token cross-attention adapter
  + frozen DAC decoder
```

The goal is not only to record the winning configuration, but also to preserve
the negative results that shaped the final design.

## 1. Research Question

LightVC started from a lightweight codec-space VC hypothesis:

```text
Can zero-shot voice conversion be done by transforming neural codec trajectories,
while keeping inference small enough for real-time use?
```

The project constraints were:

- inference in Rust/Candle, no Python runtime;
- frozen or mostly frozen codec preferred;
- no VC teacher distillation;
- low latency, with a practical target below 50 ms;
- singing support should remain possible;
- avoid heavy TTS/vocoder pipelines that betray the codec-space concept.

## 2. Final Current Result

The current best system is a frozen-DAC adapter-only model:

```text
source audio
  -> DAC encoder
  -> q0_source anchor
  -> soft RVQ residual path
  -> UTTE ECAPA-token cross-attention adapter
  -> frozen DAC decoder
```

Only the adapter is trained. DAC encoder, quantizer, and decoder stay frozen.

### 2.1 Best 200-Pair Evaluation

| Metric | A0 FiLM | B1 ECAPA-token Cross-Attn | Oracle |
|---|---:|---:|---:|
| target SECS | 0.420 [0.403, 0.436] | **0.508 [0.493, 0.522]** | 0.641 |
| source SECS | 0.311 [0.291, 0.335] | **0.238 [0.220, 0.259]** | 0.108 |
| margin | +0.108 [0.080, 0.136] | **+0.269 [0.242, 0.294]** | +0.533 |
| oracle ratio | 0.657 | **0.792** | 1.000 |

All Strong-Go criteria were satisfied on the full 200-pair evaluation:

| Criterion | Required | B1 CI-bound | Result |
|---|---:|---:|---|
| target SECS | >= 0.48 | 0.493 lower | pass |
| margin | >= +0.18 | +0.242 lower | pass |
| source SECS | <= 0.32 | 0.259 upper | pass |

### 2.2 Runtime Feasibility

The UTTE adapter was exported to safetensors and implemented in Rust/Candle.

| Artifact | Path | Status |
|---|---|---|
| Weight export | `training/export_b1_adapter.py` -> `models/utte_adapter_b1.safetensors` | done |
| Rust adapter | `crates/lightvc-core/src/utte_adapter.rs` | done |
| Rust microbench | `crates/lightvc-core/examples/bench_utte.rs` | done |

Rust/Candle CPU adapter benchmark:

| Frames | Mean |
|---:|---:|
| 1 | 0.53 ms |
| 2 | 1.09 ms |
| 8 | 1.70 ms |
| 32 | 3.04 ms |
| 128 | 5.60 ms |
| 256 | 8.26 ms |

The adapter is not the latency bottleneck. The remaining latency risks are soft
RVQ, DAC encode/decode, resampling, and audio I/O.

Subsequent Rust component benchmarking refined this:

| Frames | Soft RVQ | UTTE Adapter | Soft RVQ + Adapter |
|---:|---:|---:|---:|
| 1 | 0.23 ms | 0.57 ms | 0.90 ms |
| 8 | 1.00 ms | 1.17 ms | 2.72 ms |
| 128 | 16.0 ms | 4.7 ms | 22.9 ms |

Soft RVQ plus the adapter remains lightweight. The bottleneck is DAC
encode/decode on CPU:

| Frames | Encode | Decode | Encode + Decode |
|---:|---:|---:|---:|
| 1 | 18.4 ms | n/a | n/a |
| 2 | 25.0 ms | n/a | n/a |
| 8 | 83.8 ms | 184.2 ms | 266.2 ms |
| 32 | 296.4 ms | 677.0 ms | 1000.8 ms |
| 128 | 1539.6 ms | 3083.0 ms | 4666.8 ms |

Therefore the current method is algorithmically lightweight in the VC adapter
itself, but the frozen DAC CPU decoder is not real-time.

CUDA measurements on an RTX 2080 Ti changed the deployment conclusion:

| Frames | Encode | Decode | Encode + Decode |
|---:|---:|---:|---:|
| 1 | 2.0 ms | 4.4 ms | 6.0 ms |
| 4 | 2.9 ms | 9.6 ms | 11.8 ms |
| 8 | 4.1 ms | 13.6 ms | 16.9 ms |
| 32 | 8.8 ms | 42.1 ms | 51.7 ms |

Estimated end-to-end latency:

| Component | Strict 1f | Balanced 8f |
|---|---:|---:|
| DAC encode | 2.0 ms | 4.1 ms |
| Soft RVQ + UTTE adapter | ~0.5 ms | ~2.7 ms |
| DAC decode | 4.4 ms | 13.6 ms |
| Resampling | ~6 ms | ~6 ms |
| Audio I/O | ~6 ms | ~6 ms |
| Total | ~18.9 ms | ~32.4 ms |

Thus sub-50 ms deployment is feasible on CUDA with the frozen DAC path. The
remaining engineering task is full-pipeline CUDA parity and streaming callback
integration, not immediate migration to another codec.

Full CUDA parity was subsequently confirmed:

| Stage | MSE | Result |
|---|---:|---|
| Soft RVQ | 0.00000000 | pass |
| UTTE adapter | 0.00000005 | pass |
| DAC decode | 0.00000000 | pass |

Component latency for the streaming budget:

| Component | Strict 1f | Balanced 8f |
|---|---:|---:|
| DAC encode | 2.0 ms | 4.1 ms |
| Soft RVQ | ~0.2 ms | ~1.0 ms |
| UTTE adapter | ~0.3 ms | ~0.7 ms |
| DAC decode | 4.4 ms | 13.6 ms |
| VC processing subtotal | 6.9 ms | 19.4 ms |
| Estimated total with resampling + ASIO | 18.9 ms | 31.4 ms |

This establishes that the frozen DAC + B1 adapter path is both numerically
portable to Rust/Candle and compatible with the sub-50 ms latency target on
CUDA. The remaining risk is streaming boundary quality, not model runtime.

## 3. Evaluation Metrics

The main reported metric is SECS:

```text
SECS(output, reference) = cosine(ECAPA(output), ECAPA(reference))
```

Three values are tracked:

- target SECS: similarity to target speaker;
- source SECS: residual similarity to source speaker;
- margin: target SECS - source SECS.

The margin is essential. Several models improved target SECS while retaining
too much source identity. A valid conversion should have a positive margin.

For content preservation, earlier oracle work also used Whisper CER and F0
correlation, but the current Phase 3c adapter results are summarized mainly by
speaker metrics because the same-text oracle data already constrains content.

## 4. Major Negative Results

### 4.1 Continuous DAC Latent Editing Failed

Initial attempts treated DAC latent space as a continuous editable space:

| Approach | SECS Upper Bound |
|---|---:|
| velocity MSE, random pair | 0.14 |
| latent cosine / speaker loss | 0.14 |
| kNN-VC with DAC matching | 0.17 |
| kNN-VC distill with WavLM match | 0.16 |

The kNN target itself had SECS around 0.16, so even a perfect student would be
poor. The root issue was not the optimizer; it was decoder validity:

```text
DAC decoder reconstructs encoder-produced latents,
but not arbitrary modified continuous latents.
```

This rejected:

```text
frozen DAC decoder + arbitrary continuous latent regression
```

but did not reject codec-space VC itself.

### 4.2 WORLD / Source-Filter Route Had a Low Broad Ceiling

WORLD was explored as a mathematically controlled analysis-synthesis path.
Small 20-pair tests initially looked promising, but 200-pair evaluation
corrected the picture.

| Metric | 20-pair optimistic | 200-pair corrected |
|---|---:|---:|
| retrieval | 0.427 | 0.328 |
| oracle rerank | 0.486 | 0.377 |
| DTW oracle | 0.717 | 0.365 |

The corrected WORLD ceiling was about 0.36-0.40 across diverse speaker pairs.
This was below the desired target and below DAC's own reconstruction potential.

### 4.3 Naive RVQ Depth Swap Failed

DAC reconstruction itself had a high ceiling:

| Config | SECS |
|---|---:|
| target all, DAC resynthesis | 0.790 |
| WORLD broad ceiling | ~0.365 |

However, naive RVQ depth mixing failed:

| Config | SECS |
|---|---:|
| target coarse + source rest | 0.327 |
| random half | 0.202 |
| source coarse + target rest | 0.192 |
| source coarse + target mid | 0.171 |

The lesson was:

```text
codebook-valid token != decoder-valid trajectory
```

RVQ depths cannot be pasted independently. The residual chain must be preserved.

## 5. Residual-Chain Breakthrough

The first strong codec-space result came from preserving the RVQ residual chain.

### 5.1 Source q0 + Target Residual Re-Quantization

The winning oracle operation was:

```text
q0_hat = q0_source
q1..8_hat = RVQ_requantize(z_target_like - q0_source)
```

This keeps the source depth-0 anchor for content safety while allowing target
speaker information to enter through residual depths 1-8.

| Config | SECS |
|---|---:|
| src_K1: source d0 + target rest re-quantized | 0.686 |
| target-led K5 | 0.541 |
| WORLD ceiling | 0.365 |

Phase 2a confirmed the tradeoff:

| Config | SECS | CER | F0 corr | Leakage |
|---|---:|---:|---:|---:|
| src_K1 | 0.686 | 0.082 | 0.631 | 0.178 |
| tgt_K5 | 0.541 | 0.086 | 0.550 | 0.202 |

This established that DAC codec-space VC has high potential if the residual
trajectory remains valid.

## 6. Alignment and Retrieval Results

### 6.1 Frame-Independent Retrieval Failed

Attempts to infer the target-like latent from an enrollment bank using
frame-independent retrieval failed:

| Config | SECS | CER |
|---|---:|---:|
| Wav2Vec2 NN | 0.189 | 0.505 |
| PCA NN | 0.200 | 0.585 |
| Wav2Vec2 top-k | 0.201 | 0.441 |
| random | 0.056 | 0.907 |

The failure was temporal discontinuity and mismatched content:

```text
nearest frame in acoustic/codec space != same linguistic unit
```

### 6.2 Same-Text Content-Aware DTW Worked

Wav2Vec2 content-feature DTW restored temporal structure:

| Config | SECS | CER | F0 corr |
|---|---:|---:|---:|
| DAC DTW oracle | 0.686 | 0.082 | 0.631 |
| Wav2Vec2 layer 6 DTW | 0.656 | 0.057 | 0.726 |
| Wav2Vec2 layer 9 DTW | 0.641 | 0.044 | 0.723 |
| Wav2Vec2 layer 12 DTW | 0.635 | 0.073 | 0.706 |

This validated the formula when target-like latents are temporally aligned.

### 6.3 Cross-Text Subsequence DTW Failed

Free-conversation cross-text retrieval did not work:

| Config | SECS | CER |
|---|---:|---:|
| same-text oracle | 0.656 | 0.057 |
| best cross-text | 0.330 | ~0.73 |
| cross full utterance | 0.167 | 0.893 |

The issue was structural:

```text
for different text, a correct monotonic alignment path often does not exist
```

This forced the project away from enrollment-bank retrieval and toward a
trainable generator/adapter.

## 7. Generator Failures and Corrected Diagnosis

### 7.1 Phase 3 Latent-Domain Generators Collapsed

Three generator families failed:

- continuous latent with STE;
- code classification over 1024 classes;
- 8-dimensional codebook embedding MSE.

Empirical diagnostic of the embedding model:

| Depth | Cosine | Code Acc | Oracle-Code Rank |
---|---:|---:|---:|
| d1 | 0.322 | 0.6% | 319 |
| d2 | 0.257 | 0.8% | 327 |
| d3 | 0.161 | 0.1% | 388 |
| d4 | 0.165 | 0.3% | 381 |
| d5 | 0.116 | 0.4% | 422 |
| d6 | 0.134 | 1.2% | 408 |
| d7 | 0.114 | 0.2% | 421 |
| d8 | 0.047 | 0.2% | 472 |

Decoded output:

| Condition | Target SECS | Source SECS | Margin |
|---|---:|---:|---:|
| oracle | 0.664 | 0.191 | +0.473 |
| model hard | 0.041 | 0.126 | -0.086 |
| model soft tau=1 | 0.046 | 0.153 | -0.106 |
| model soft tau=5 | 0.069 | 0.231 | -0.162 |

The generator output was closer to the source than to the target.

### 7.2 T0 Tolerance Sweep Falsified the Wrong Hypothesis

The initial interpretation was "RVQ cascade sensitivity." T0 showed that this
was not the main cause.

Hard RVQ remained stable under random latent noise:

```text
sigma=0.0: hard RVQ SECS ~0.664
sigma=0.5: hard RVQ SECS ~0.659
```

Soft RVQ improved oracle decoding slightly:

```text
hard RVQ:      ~0.664
soft RVQ t=5: ~0.698
```

The corrected diagnosis:

```text
random noise is tolerated,
structured generator error is not.
```

Latent-domain losses learned a speaker-averaged or source-like solution:

```text
argmin_f E || f(x) - y ||^2 = E[y | x]
```

For multi-speaker residual targets, this conditional mean carries weak target
identity.

## 8. Phase 3b: Decoded-Audio Loss Was Not Enough

The next attempt trained through soft RVQ and the frozen DAC decoder using
decoded-audio losses.

Three configurations converged to the same plateau:

| Config | Target | Source | Margin |
|---|---:|---:|---:|
| tau=5, lr=1e-4, latent=0.02 | 0.143 | 0.308 | -0.165 |
| tau=1, lr=3e-4, latent=0.1 | 0.143 | 0.308 | -0.165 |
| tau=5, lr=1e-4, latent=0.5 | 0.124 | 0.283 | -0.159 |

Gradient audit showed:

- speaker loss gradient was not zero;
- the soft-RVQ + decoder path attenuated gradients before `z_pred`;
- anti-leakage was easier to learn than target-specific conversion.

This led to a new design principle:

```text
put target speaker conditioning closer to the decoder
```

## 9. Phase 3c: Timbre-Conditioned Pre-Decoder Adapter

### 9.1 FiLM Adapter Worked but Leaked Source

The first adapter was a small pre-decoder FiLM Conv adapter:

```text
z_q_soft
  -> Conv1d(1024, 256)
  -> FiLM(ECAPA)
  -> GELU
  -> Conv1d(256, 1024)
  -> residual add
  -> frozen DAC decoder
```

With 10K VCTK same-text pairs, adapter-only became valid:

| Metric | 2K AO | 10K AO |
|---|---:|---:|
| target SECS | 0.378 [0.357, 0.399] | 0.420 [0.403, 0.436] |
| source SECS | 0.426 [0.408, 0.442] | 0.311 [0.291, 0.335] |
| margin | -0.048 [-0.078, -0.018] | +0.108 [0.080, 0.136] |

This showed that data scale mattered. The 1.67M-parameter adapter could
generalize with enough same-text pairs.

### 9.2 Capacity Increase Was Not the Solution

Ablations:

| Config | Margin | Target | Source | Interpretation |
|---|---:|---:|---:|---|
| A0 bottleneck 256 | +0.108 | 0.420 | 0.311 | baseline |
| A1 bottleneck 512 | +0.078 | 0.397 | 0.314 | worse |
| A3 margin loss | +0.115 | 0.382 | 0.268 | source lower, target sacrificed |

The bottleneck was not raw adapter capacity. Stronger anti-leakage reduced
both source and target similarity. The target ceiling remained around 0.40.

### 9.3 B1 ECAPA-Token Cross-Attention Breakthrough

The decisive change was replacing FiLM-only conditioning with ECAPA-token
cross-attention:

```text
ECAPA target vector [192]
  -> Linear
  -> 32 timbre tokens [32, 256]

adapter frames [T, 256]
  -> query
timbre tokens
  -> key/value
frame-level cross-attention
```

This used the same ECAPA 1-vector. No target latent tokens were required.

Full 200-pair result:

| Metric | A0 FiLM | B1 ECAPA-token Cross-Attn | Oracle |
|---|---:|---:|---:|
| target SECS | 0.420 [0.403, 0.436] | 0.508 [0.493, 0.522] | 0.641 |
| source SECS | 0.311 [0.291, 0.335] | 0.238 [0.220, 0.259] | 0.108 |
| margin | +0.108 [0.080, 0.136] | +0.269 [0.242, 0.294] | +0.533 |
| oracle ratio | 0.657 | 0.792 | 1.000 |

Conclusion:

```text
the bottleneck was FiLM conditioning,
not ECAPA information capacity.
```

## 10. Current Technical Contribution

The current system suggests the following paper contribution:

> A residual-chain-preserving codec-space VC method that converts speaker
> identity using a frozen DAC decoder and a small ECAPA-token cross-attention
> pre-decoder adapter.

Key design points:

1. preserve DAC residual-chain validity;
2. use source q0 as a content-preserving anchor;
3. use soft residual RVQ for a differentiable decoder input;
4. inject target speaker identity near the decoder;
5. replace FiLM conditioning with ECAPA-token frame-level cross-attention;
6. keep the trained adapter small and runtime-feasible.

## 11. Candidate Paper Framing

### 11.1 Possible Title

**Residual-Chain-Preserving Codec-Space Voice Conversion with Timbre Token
Decoder Adaptation**

### 11.2 Abstract Skeleton

Neural audio codecs provide compact representations for real-time speech
processing, but direct latent editing often leaves the decoder manifold. We
study a sequence of codec-space voice conversion designs under a frozen DAC
decoder. Negative results show that continuous latent regression, naive RVQ
depth swapping, WORLD source-filter conversion, and cross-text retrieval all
fail for different structural reasons. We identify RVQ residual-chain validity
as a necessary condition and use a source depth-0 anchor with residual
re-quantization. To bridge the remaining gap between aligned oracle conversion
and zero-shot inference, we introduce a small timbre-conditioned pre-decoder
adapter. Replacing FiLM conditioning with ECAPA-token cross-attention improves
target speaker similarity from 0.420 to 0.508 and margin from +0.108 to +0.269
on a 200-pair VCTK evaluation, reaching 79% of the aligned oracle while keeping
the DAC decoder frozen. A Rust/Candle implementation of the adapter runs in
0.53 ms for a one-frame chunk on CPU, indicating that the adapter is not the
latency bottleneck.

### 11.3 Main Claims

Claim 1:
Naive codec latent editing and naive RVQ token swapping are insufficient because
decoder-valid trajectories require residual-chain consistency.

Claim 2:
Source q0 anchoring plus residual re-quantization provides a high-quality
same-text oracle and preserves content better than target-led replacement.

Claim 3:
Latent-domain generator objectives collapse to speaker-averaged residuals even
when the decoder and RVQ path are tolerant to random noise.

Claim 4:
Pre-decoder speaker conditioning is more effective than generating the entire
target-like latent trajectory.

Claim 5:
ECAPA-token cross-attention is substantially more effective than FiLM for
injecting target speaker identity into the adapter.

## 12. What Should Be in the Paper vs Appendix

### Main Paper

- Final method diagram.
- Residual-chain validity motivation.
- Main negative-result table.
- src_K1 oracle.
- Phase 3c A0 vs B1 full evaluation.
- Rust/Candle latency table.

### Appendix

- WORLD route details.
- Retrieval/reranking failures.
- T0 tolerance sweep.
- Empirical generator-error table.
- Full ablation logs.
- Implementation details for PyTorch MHA -> Candle q/k/v/o export.

## 13. Remaining Work Before Submission

The current result is strong enough for a workshop-style systems/research
paper draft, but the following items should be completed before a stronger
submission:

1. Streaming decode design:

- larger decode window with overlap-add, or
- Strict 1-frame mode if quality is acceptable.

2. ASIO/CoreAudio end-to-end callback latency measurement.

3. Audio quality metrics beyond SECS:

- UTMOS or DNSMOS;
- Whisper CER/WER;
- F0 correlation;
- human listening samples.

5. Singing-specific validation:

- sustained vowels;
- vibrato preservation;
- high-F0 behavior;
- pitch-controlled conversion.

## 14. Current One-Sentence Summary

LightVC's key finding is that frozen-codec voice conversion becomes viable when
the model stops trying to generate arbitrary codec latents and instead applies
a small timbre-token cross-attention adapter to a residual-chain-valid DAC
trajectory.
