# LightVC Research Report 2026-06-20

## Summary

LightVC の当初 CONCEPT は「重い波形生成器を VC 本体に持たず、codec 空間で低遅延に声質変換する」ことだった。ここまでの実験により、以下が確定した。

1. DAC continuous latent regression は off-manifold 問題で失敗した。
2. WORLD source-filter / mcep retrieval は 200ペア評価で天井が低いことが分かった。
3. DAC token naive depth swap も token trajectory が壊れ、失敗した。
4. ただし、RVQ の残差鎖を保つ re-quantization は明確に有効だった。
5. 現時点の最重要発見は `src_K1`: source depth 0 を保持し、target-like latent に対して depth 1-8 を再量子化する構成である。

最終的な方向は、CONCEPT v1 の「連続 latent を回帰する VC」ではなく、CONCEPT v2 の **codec-valid trajectory translation** へ移行する。

```text
誤: continuous codec latent regression
正: residual-chain-preserving codec token trajectory translation
```

## Project Constraints

- Inference is Rust/Candle only.
- Training is PyTorch in uv environment.
- No conda.
- No VC teacher distillation.
- MIT-compatible dependencies only.
- Target product direction: lightweight, low-latency, preferably below 50 ms.
- CONCEPT should remain codec-space, not a heavy TTS/BigVGAN-style pipeline.

## Phase 0: DAC Continuous Latent Experiments

### Tested Approaches

| Approach | SECS Upper Bound |
|---|---:|
| velocity MSE, random pair | 0.14 |
| latent cosine, distilled speaker embedding | 0.14 |
| waveform cosine after DAC decode | not usable |
| kNN-VC, DAC matching | 0.17 |
| kNN-VC distill, WavLM match | 0.16 |

### Conclusion

The kNN target itself had SECS around 0.16. Even a perfect model could not exceed that target.

The failure was not primarily optimization. The issue was:

```text
z_hat = T(z_source, target)
z_hat not in decoder-valid speech manifold
```

DAC decoder can reconstruct encoder-produced latents, but it does not reliably synthesize natural speech from modified continuous latents.

This invalidated:

```text
frozen DAC decoder + arbitrary continuous latent editing
```

It did not invalidate codec-space VC itself.

## Phase 1: WORLD Source-Filter Route

The next route was analysis-synthesis:

```text
source speech
-> F0 / VUV / energy / WORLD mcep / AP
-> target-like mcep
-> WORLD synthesis
```

The goal was a low-latency source-filter VC distinct from Beatrice-style end-to-end VC.

### Initial Findings

Early small-scale oracle tests suggested:

| Test | SECS |
|---|---:|
| WORLD self resynthesis | 0.865 |
| DTW aligned + F0 shift | 0.350 |
| affine transport | 0.073 |

This showed WORLD itself could reconstruct a speaker, but simple cross-speaker envelope transfer was weak.

### Direct Model Results

| Method | SECS target |
|---|---:|
| direct prediction v1 | 0.261 |
| transport + residual | 0.103 |

The direct model tended to predict a speaker-averaged median mcep. FiLM conditioning was too weak, and L1/MSE regression naturally collapsed toward:

```text
argmin_f E || f(x, ref) - y_target ||^2
=> f(x, ref) approx E[y_target | x]
```

### Timbre Bank Retrieval

A non-learned target timbre bank was tested.

Small 20-pair evaluation initially looked promising:

| Config | SECS |
|---|---:|
| baseline 1-frame | 0.333 |
| + ctx8 | 0.402 |
| + inverse F-ratio weighting | 0.419 |
| + k=3 sharper blend | 0.427 |

However, this 20-pair set was biased toward easy speaker pairs.

## Phase 1 Correction: 200-Pair Evaluation

A larger 200-pair evaluation with bootstrap confidence intervals corrected the earlier optimism.

### Retrieval Results

| Config | Mean | CI low | CI high |
|---|---:|---:|---:|
| ctx8_b5 | 0.324 | 0.310 | 0.339 |
| ctx8_b10 | 0.328 | 0.312 | 0.343 |
| ctx8_b25 | 0.336 | 0.321 | 0.351 |
| ctx8_b50 | 0.343 | 0.328 | 0.358 |
| ctx8_b100 | 0.341 | 0.326 | 0.357 |

### Oracle Rerank Results

| Config | Mean | CI low | CI high |
|---|---:|---:|---:|
| oracle_b5 | 0.366 | 0.351 | 0.381 |
| oracle_b10 | 0.377 | 0.363 | 0.392 |
| oracle_b25 | 0.391 | 0.377 | 0.405 |
| oracle_b50 | 0.395 | 0.379 | 0.410 |
| oracle_b100 | 0.400 | 0.386 | 0.415 |

### Corrected DTW Oracle

The earlier DTW oracle had a slicing bug and was also evaluated on easy pairs. Corrected 200-pair result:

```text
DTW Oracle: 0.365 ± 0.090
```

Breakdown:

| Pair Range | DTW Oracle SECS |
|---|---:|
| 1-20 | 0.683 |
| 81-100 | 0.521 |
| 101-120 | 0.206 |
| 121-140 | -0.006 |
| 181-200 | 0.198 |

### Corrected Interpretation

The previous `0.43` retrieval and `0.71` DTW oracle were overestimates.

Corrected picture:

| Metric | Old 20-pair | Corrected 200-pair |
|---|---:|---:|
| retrieval | 0.427 | 0.328 |
| oracle rerank | 0.486 | 0.377 |
| DTW oracle | 0.717 | 0.365 |

WORLD route conclusion:

```text
WORLD VC ceiling is around 0.36-0.40 for broad 200-pair evaluation.
```

The bottleneck was not only retrieval or ranking. Even the target mcep DTW oracle did not robustly preserve speaker identity across diverse speaker pairs.

## CONCEPT v2

The codec-space idea was retained, but the representation hypothesis changed.

### Rejected Hypothesis

```text
codec latent can be continuously edited and decoded
```

### New Hypothesis

```text
codec-valid token trajectory can be translated if the RVQ residual chain is preserved
```

The key distinction:

```text
codebook-valid token
does not imply
decoder-valid trajectory
```

## Phase 1: RVQ Token Swap Oracle

### Naive Depth Swap

| Config | Mean | CI |
|---|---:|---|
| target_all, DAC resynthesis upper bound | 0.790 | [0.777, 0.802] |
| tgt_coarse + src_rest | 0.327 | [0.305, 0.347] |
| random_half, negative control | 0.202 | [0.182, 0.222] |
| src_coarse + tgt_rest | 0.192 | [0.173, 0.211] |
| src_coarse + tgt_mid | 0.171 | [0.155, 0.188] |
| src_coarse_mid + tgt_fine | 0.143 | [0.128, 0.160] |

### Findings

1. DAC ceiling was high: `0.790`, more than twice the WORLD ceiling.
2. Naive depth swap was close to random mixing.
3. RVQ depths were not independently swappable.
4. The original coarse/mid/fine assumption was wrong.

The failure was the token version of the off-manifold problem:

```text
all tokens are codebook-valid
but mixed-depth token sequence is residual-chain-invalid
```

## RVQ Residual Chain Mathematics

RVQ encoding is sequential:

```text
r_1 = z
q_1 = Q_1(r_1)
r_2 = r_1 - q_1
q_2 = Q_2(r_2)
...
q_d = Q_d(r_d)
r_{d+1} = r_d - q_d
```

Each token `q_d` is conditioned on the residual left by previous depths.

Therefore, replacing `q_d` independently is invalid:

```text
q_d_target may not be valid for r_d_source
```

The right operation is not:

```text
paste target q_1..q_N
```

but:

```text
fix selected prefix
then re-quantize remaining residual
```

## Phase 1b: Residual-Chain-Preserving Re-Quantization

### Results

| Config | Mean | CI |
|---|---:|---|
| target_all, continuous | 0.589 | [0.573, 0.605] |
| source_all | 0.152 | [0.137, 0.168] |
| src_k0, all target quantized | 0.790 | [0.777, 0.804] |
| src_k1, source d0 + target rest | 0.686 | [0.673, 0.698] |
| src_k2, source d0-1 + target rest | 0.416 | [0.396, 0.434] |
| src_k3, source d0-2 + target rest | 0.221 | [0.202, 0.239] |
| src_k5+ | about 0.15 | source_all level |
| tgt_k1 | 0.151 | |
| tgt_k3 | 0.265 | |
| tgt_k5 | 0.541 | [0.523, 0.559] |

### Single Depth Ablation

| Config | SECS |
|---|---:|
| tgt_minus_d0 | 0.352 |
| tgt_minus_d1 | 0.538 |
| tgt_minus_d2 | 0.650 |
| tgt_minus_d3+ | 0.69+ |

### Findings

1. Residual-chain preservation strongly improved token mixing.
2. `src_k1` was extremely strong: source depth 0 plus target residual re-quantization reached `0.686`.
3. Depth 0 was the most important speaker-bearing depth.
4. Depth 1 was second most important.
5. Depth 3+ had small effect on speaker similarity.

This inverted the earlier coarse/mid/fine hypothesis.

Previous hypothesis:

```text
depth 1-3: content
depth 4-6: timbre
depth 7-9: texture
```

Updated hypothesis:

```text
depth 0: speaker strongest + content mixed
depth 1: speaker/timbre still strong
depth 2: speaker auxiliary + phonetic detail
depth 3+: residual texture/detail
```

## Phase 2a: SECS-Content Tradeoff Curve

The critical question was whether high SECS could be achieved without destroying content.

### Go Configs

Criterion:

```text
SECS >= 0.45
CER <= 0.103
```

| Config | SECS | CER | F0 corr | Leakage |
|---|---:|---:|---:|---:|
| src_K1 | 0.686 | 0.082 | 0.631 | 0.178 |
| tgt_K5 | 0.541 | 0.086 | 0.550 | 0.202 |
| tgt_K4 | 0.396 | 0.131 | 0.476 | 0.234 |

### Findings

`src_K1` was the clear winner:

```text
source depth 0 retained
target-like residual depths 1-8 re-quantized
```

It achieved:

```text
SECS = 0.686
CER  = 0.082
```

This is 87% of the DAC quantized upper bound `0.790`, while keeping content degradation modest.

Naive hybrid mixing failed with high CER:

```text
CER = 0.28-0.44
```

The safe path is therefore:

```text
do not paste tokens
retrieve or construct target-like latent
then re-quantize residual conditioned on source depth 0
```

## Current Core Algorithm

The best current oracle operation is:

```text
q_hat_0 = q_source_0
q_hat_1..8 = RVQ_requantize(z_target_like - q_source_0)
y = DAC_decode(q_hat)
```

The essential runtime problem is:

```text
How to obtain z_target_like without DTW alignment?
```

## Phase 2b Direction

The next phase should not retrieve target tokens directly.

It should retrieve target latent frames:

```text
source frame:
  q0_s, z_s, unit_s, f0_s, energy_s, context_s

target enrollment bank:
  z_t_i, unit_t_i, f0_t_i, energy_t_i, context_t_i

retrieve:
  i* = argmin_i d_unit(unit_s, unit_t_i)
              + alpha d_f0(f0_s_shifted, f0_t_i)
              + beta d_energy(energy_s, energy_t_i)
              + gamma d_context(context_s, context_t_i)

convert:
  q0_hat = q0_s
  q1..8_hat = RVQ_requantize(z_t_i* - q0_s)
  y = DAC_decode(q_hat)
```

### Phase 2b Experiments

1. Aligned oracle
   - Existing `src_K1 = 0.686`.
   - Confirms upper bound.

2. Same-phoneme oracle
   - Use text or MFA phoneme labels.
   - Select best target latent frame inside the same phoneme.
   - Measures unit-indexed bank upper bound.

3. Unsupervised unit retrieval
   - Try DAC q0/q1 clusters, low-dimensional z clusters, or lightweight content features.

4. Top-k latent blend plus re-quantization
   - Blend continuous target latents, not tokens.

```text
z_t_like = sum_i w_i z_t_i
q1..8 = RVQ_requantize(z_t_like - q0_s)
```

### Phase 2b Go Conditions

```text
same-phoneme oracle >= 0.55
unit retrieval      >= 0.45
CER                 <= 0.10
target_sim - source_sim > 0
```

## Strategic Conclusions

### What Failed

1. Continuous DAC latent regression.
2. WORLD mcep retrieval as a primary VC route.
3. L1/MSE regression to aligned acoustic targets.
4. FiLM-only speaker conditioning.
5. Naive RVQ depth swapping.
6. Direct target token pasting.

### What Survived

1. Codec-space VC as the main concept.
2. DAC decoder as a high-potential codec decoder.
3. RVQ token operations, if residual chain is preserved.
4. Source depth 0 retention for content safety.
5. Target-like latent residual re-quantization for speaker conversion.

### Updated CONCEPT

LightVC should be described as:

```text
LightVC is not continuous latent editing.
LightVC is residual-chain-preserving codec trajectory translation.
```

Operationally:

```text
preserve source q0 for content safety
retrieve or predict target-like latent z_t_like
re-quantize depths 1..N under the source q0 prefix
decode with DAC
```

## Open Risks

1. Source depth 0 contains speaker information.
   - `src_K1` may leave source leakage.
   - Must track `target_sim - source_sim`, not only SECS.

2. Phase 2a used aligned target latent.
   - Runtime retrieval may fail to approximate `z_target_like`.

3. Unit extraction may require non-lightweight features.
   - Diagnostic use of SSL or MFA is acceptable, but final Rust inference should avoid heavy runtime dependencies.

4. DAC encoder/decoder latency is not yet proven below 50 ms in Rust/Candle.

5. Enrollment protocol may need to be controlled.
   - Arbitrary short target reference may be insufficient.
   - Phoneme-balanced enrollment may be required.

## Recommended Next Steps

1. Implement Phase 2b same-phoneme oracle.
2. Compare:
   - aligned oracle
   - same-phoneme oracle
   - unsupervised unit retrieval
   - top-k latent blend
3. Always report:
   - SECS
   - CER/WER
   - F0 correlation
   - source leakage
   - target_sim - source_sim
4. If same-phoneme oracle is high but retrieval is low, focus on unit/key design.
5. If same-phoneme oracle is low, arbitrary enrollment bank may be insufficient; move to controlled enrollment protocol.

## Current Best One-Line Result

```text
source depth 0 + target-like residual re-quantization achieves SECS 0.686 and CER 0.082,
showing that codec-space VC is viable only when RVQ residual-chain validity is preserved.
```

## Addendum 2026-06-21

Phase 3 tried to replace aligned/retrieved `z_target_like` with a learned generator.

Tested approaches:

| Approach | Result |
|---|---|
| continuous latent + STE | failed |
| code CE over 1024 classes | failed |
| embedding MSE | failed |

The critical failure mode was RVQ-cascade sensitivity:

```text
latent cosine around 0.67
can still decode to SECS around 0.03
```

This means the generator was not merely undertrained. Small residual-latent or code errors are amplified by the sequential RVQ quantization chain and become decoder-invalid trajectories.

Updated conclusion:

```text
target-like residual generation is still the right abstract objective,
but hard RVQ re-quantization is too brittle for approximate generator outputs.
```

Next main experiment:

```text
decoder adapter / tolerant decoder
```

The next question is whether DAC decoding can be made tolerant to approximate target-like residual latents without destroying ordinary DAC round-trip quality. If it cannot, the frozen-DAC route is likely unsuitable for free-conversation zero-shot VC and singing; a VC-aware codec or codec-integrated voice changer becomes the more coherent fallback.
