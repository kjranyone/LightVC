# Plan 11: Causal Source-Filter Voice Conversion

> **Status (2026-06-21): research stopped as main route.**
> Corrected 200-pair evaluation showed WORLD/mcep ceiling around `0.36-0.40`,
> with high speaker-pair variance. Keep this file as diagnostic history only.
> Current route: [plan/12_concept_v2.md](12_concept_v2.md).

## Motivation

DAC latent-space VC 失敗後、2つのアプローチを検討:
1. **mel + BigVGAN** (analysis-resynthesis): vocoderが重い、<50ms レイテンシ不可
2. **causal source-filter VC**: 軽量、物理制約あり、Rust移植容易

→ **source-filter を採用**。Beatrice 型ブラックボックス変換ではなく、
「励振を保ったまま声道フィルタを輸送する」明示的生成モデル。

## 数理モデル

```
音声 = 励振源 × 声道フィルタ + 非周期成分

フレーム k:
  x_k → (f0_k, vuv_k, energy_k, mc_k, codeap_k)

変換:
  mc_t,k = Gθ(mc_s,k, speaker_target_embed)

合成:
  y_k = WORLD_synthesize(f0_shifted_k, mc_t_k, codeap_t_k)
```

### なぜ安定か

- mc (mel-cepstrum) → sp = exp(C(ω)) で正値スペクトル保証
- WORLD synthesis は物理モデル → decoder manifold 問題なし
- 学習対象は 25次元 mel-cepstrum のみ → 軽量

## Oracle 実験結果

| Test | Description | SECS(tgt) | SECS(src) | 判定 |
|------|------------|-----------|-----------|------|
| O4 | WORLD自己再合成 | 0.865 | — | 品質上限 |
| O1 | src F0 + tgt real env | 0.430 | 0.070 | ベースライン |
| O1b | + F0 shift | 0.486 | 0.078 | F0効果+0.06 |
| O1c | + DTW整列 | 0.582 | 0.181 | 整列効果+0.15 |
| O1d | F0 shift + DTW | **0.662** | 0.129 | **✓ 上限OK** |
| O2 | per-register平均env | 0.144 | 0.096 | ✗ 不十分 |
| O3 | affine transport | 0.175 | 0.634 | ✗ src漏洩 |
| O3b | reg置換 + F0 shift | 0.155 | 0.068 | ✗ 不十分 |

### 所見

1. **O1d=0.66**: DTW整列 + F0 shift で 0.66 達成 → source-filter VC は viable
2. **DTW整列 (+0.15) > F0 shift (+0.06)**: phoneme alignment が最重要
3. **O2/O3b=0.15**: register平均では話者性が出ない → frame-level variation が必須
4. **O3=0.175 (src漏洩0.63)**: affine transport は source 変動を保持 → src漏洩
5. **学習対象**: O1d(0.66) - O3b(0.15) = 0.51 の差を neural model で埋める

## アーキテクチャ

### EnvelopeConverter

```
Input:  mc_src [B, 25, T]           (source mel-cepstrum)
        spk_emb [B, D]               (target speaker embedding)
                                       
        ┌─ proj_in: Conv1d(25→192, 1x1)
        ├─ FiLM: spk_emb → γ,β [B, 192]
        ├─ TCN blocks ×6:
        │   ├─ CausalConv1d(192, k=7, dilation=1,2,4,8,1,2)
        │   ├─ GELU
        │   ├─ Conv1d(192→192, 1x1)
        │   └─ Residual
        └─ proj_out: Linear(192→25), zero-init
                                       
Output: mc_hat [B, 25, T]            (predicted target mel-cepstrum)
```

- **Causal**: 未来フレームを見ない（リアルタイム）
- **Zero-init residual**: 学習初期は恒等写像
- **~0.5M params**: 極軽量

### Training Data

- VCTK same-text pairs (244 texts, ~66 speakers/text)
- DTW align source mc to target mc
- Training pair: (mc_src, spk_emb_tgt) → mc_tgt_aligned
- F0 はモデル外で処理（mean shift）

### Inference Pipeline

```
wav_src → WORLD analysis → (f0, mc_src, codeap_src)
                                ↓
         mc_src + spk_emb_tgt → EnvelopeConverter → mc_hat
                                ↓
         f0_shifted = f0_src × (mean_f0_tgt / mean_f0_src)
                                ↓
         (f0_shifted, mc_hat, codeap_tgt_avg) → WORLD synth → wav_out
```

### Latency Budget (<50ms)

| Component | Time |
|-----------|------|
| WORLD analysis (DIO+CheapTrick) | 5-8ms |
| EnvelopeConverter | 1-2ms |
| F0 shift | <1ms |
| WORLD synthesis | 3-5ms |
| Audio buffer | 10-15ms |
| **Total** | **20-31ms** |

## Rust 推論パス

- WORLD analysis: C++ library → Rust FFI, or pure Rust DIO+ CheapTrick
- EnvelopeConverter: Candle (Conv1d, ~0.5M params)
- WORLD synthesis: MLSA filter (pure Rust DSP)
- 依存: すべて MIT/Apache

## Milestones

1. ✅ Oracle tests (O1d=0.66 viable)
2. ⏳ Pre-compute mel-cepstrum cache (21772 files)
3. ⬜ Train EnvelopeConverter (30K steps, DTW-aligned pairs)
4. ⬜ Evaluate: SECS target >0.50, source <0.30
5. ⬜ Add speaker consistency loss (optional)
6. ⬜ Content preservation test (ASR WER)
7. ⬜ Rust inference prototype

## Fallback Plan

If EnvelopeConverter SECS < 0.40:
- Add source speaker embedding (src→tgt mapping)
- Add F0 as model input
- Increase model capacity (hidden=256, n_blocks=8)
- Multi-resolution loss (STFT + mel-cepstrum)

If EnvelopeConverter SECS > 0.50 but content degraded:
- Add content loss (phoneme CTC)
- Constrain residual (||mc_hat - mc_src|| ≤ ε)
