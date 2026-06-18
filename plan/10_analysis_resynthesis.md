# 10 — アーキテクチャ再設計：話者条件付き再合成

> **作成日**: 2026-06-18  
> **ステータス**: 📄 計画中  
> **優先度**: **P0**  
> **前提**: DAC latent-space VC の5つの失敗（plan/09 参照）を踏まえ、アーキテクチャを根本から再設計する。

---

## 1. 設計原則

> **変換ではなく、話者条件付き再合成にする。  
> ただし content から source speaker を落とし、  
> target speaker を使わないと学習できないデータ配置にする。**

### 失敗からの教訓

DAC 方式の失敗は `z_hat = z_src + Δ` で `z_hat` が自然な latent 分布から外れたこと。
分析・再合成方式ではこれを避ける：

```
source x_s → C(x_s): content, P(x_s): prosody
ref r_t    → S(r_t): speaker/timbre
合成: mel_hat = G(C(x_s), P(x_s), S(r_t))
出力: y_hat = Vocoder(mel_hat)
```

### 数理条件（4つの必須条件）

| # | 条件 | 検証方法 |
|---|---|---|
| C1 | I(C(x); speaker) ≈ 0 | speaker adversarial loss / source leakage penalty |
| C2 | I(S(r); speaker) high | SECS(S(r_a), S(r_a')) high for same speaker |
| C3 | G は S(r) を使わないと loss が下がらない | same-text cross-speaker pair / timbre-shift |
| C4 | mel_hat ∈ support(real mel) | mel統計の比較 / vocoder品質確認 |

---

## 2. 壁と対策

### 壁1: Content表現にsource speakerが漏れる

**問題**: 自己再構成だけで学習すると C(x) に speaker 情報が残り、推論で S(ref) が無視される。

**対策**:
- C = WavLM/HuBERT 上位層 (content-dominant)
- C = k-means / discrete units (さらに speaker 情報を削減)
- speaker adversarial loss: speaker classifier(C(x)) が当たらないようにする
- timbre perturbation augmentation

### 壁2: target speaker条件が弱い

**問題**: 単一 speaker embedding では brightness/breathiness/nasality 等の局所 timbre が落ちる。

**対策**: global + local に分ける
- e_global = speaker embedding (WavLM-SV / ECAPA, 256-dim)
- T_timbre = timbre token bank / reference mel tokens (frame-level)
- G(c, p, e_global, T_timbre)

### 壁3: mel/vocoderでもoff-manifoldは起きる

**問題**: mel_hat が vocoder の想定分布から外れると音が壊れる。

**対策**: Q0 で先に vocoder upper bound を測る。G が出す mel_hat の統計を real mel と比較。

### 壁4: 非パラレル学習ではspeaker変換が識別不能

**問題**: 自己再構成だけでは S(ref) を使わなくても loss が下がる。

**対策**: speaker 条件への介入を学習に含める:
- same content, different speaker (same-text parallel pair)
- same content, timbre-shifted speaker
- speaker embedding swapped

---

## 3. 段階的検証パイプライン

### Q0: Vocoder Upper Bound

```
real wav → real mel → Vocoder → wav
```

合格条件:
- WER ≈ real wav (content保持)
- UTMOS > 3.5 (品質)
- SECS(vocoder_output, real_target) 高い (話者保持)

不合格なら synthesizer 以前の問題。

### Q1: Same-Speaker Reconstruction

```
C(x_A) + P(x_A) + S(ref_A) → mel_A → Vocoder → wav
```

合格条件:
- WER < 5%
- UTMOS > 3.5
- SECS(output, A) > 0.7

### Q2: Speaker Swap Identifiability (same-text pair)

```
C(x_A, text u) + P(x_A, text u) + S(ref_B) → mel(x_B, text u)
```

合格条件:
- SECS(output, B) ↑ (target speaker に近づく)
- SECS(output, A) ↓ (source speaker から離れる)
- WER 維持
- UTMOS 維持

### Q3: Nonparallel Generalization

Q2 合格後に任意 source / 任意 target reference へ拡張。

---

## 4. アーキテクチャ

### コンポーネント

| 層 | モデル | 出力 | 推論時 |
|---|---|---|---|
| Content C | WavLM L14-18 → k-means (500 clusters) | discrete units [T] | **必須** |
| Prosody P | pyin/pYIN F0 + energy extractor | F0, energy, V/UV [T] | **必須** |
| Speaker S | WavLM-SV / ECAPA-TDNN (frozen) | embedding [256] | **必須** |
| Generator G | Conformer / UNet mel predictor | mel [128, T_mel] | **必須** |
| Vocoder V | BigVGAN v2 44kHz (frozen) | waveform | **必須** |

### Generator の進化

1. **Baseline**: Conformer / UNet mel predictor (非自己回帰)
2. **Quality**: Conditional Flow Matching on mel (10-32 step ODE)
3. **Real-time**: distill to 1-4 step

### 学習データ構成

```
[自己再構成] x_A → C, P, S(ref_A) → mel_A      (content/prosody/mel学習)
[Same-text pair] x_A(text u) + ref_B → mel_B(text u)  (speaker変換学習)
[Timbre-shift] x_A_shifted + ref_A → mel_A            (汎化)
```

same-text pair が speaker 変換の core training signal。
VCTK は全話者が同じテキストを朗読するため、豊富な same-text pair が存在。

### ライセンス

| コンポーネント | ライセンス | 問題 |
|---|---|---|
| WavLM | MIT | ✅ |
| ECAPA-TDNN | MIT | ✅ |
| BigVGAN v2 | MIT | ✅ |
| VCTK | CC-BY-4.0 | ✅ |
| LightVC 本体 | MIT | ✅ |

---

## 5. CONCEPT.md との整合

CONCEPT.md の「codec-space one-step VC」ビジョンは、DAC latent の話者分離不足と decoder の off-manifold 問題により達成不可能と確定。

新ビジョン:

> **LightVC: 分析・再合成型 real-time VC**  
> WavLM content + F0/prosody + speaker embedding → mel generation → BigVGAN

CONCEPT.md はこの新ビジョンに更新する。

### 維持する要素
- Pure Rust inference (Candle) — BigVGAN, WavLM の Rust 実装が必要
- Real-time streaming — FRC + overlap-add は mel でも適用可能
- Progressive depth control — RVQ depth ではなく、mel 周波数帯域や生成ステップ数で制御
- MIT license

### 現行コードからの継承
| コンポーネント | 状態 |
|---|---|
| Audio I/O (cpal) | ✅ 継続 |
| UI (egui) | ✅ 継続 |
| CLAP plugin | ✅ 継続 |
| Streaming logic (FRC/overlap-add) | ✅ mel でも利用 |
| DAC encoder/decoder | ❌ 廃止（vocoder が BigVGAN に置き換わる）|
| FlowConverter | ❌ 廃止（generator G に置き換わる）|

---

## 6. タスクリスト

### Q0: Vocoder Upper Bound（即時）

| ID | タスク | 所要 |
|---|---|---|
| **10-Q0** | BigVGAN v2 で real mel → wav の upper bound 測定 (WER, UTMOS, SECS) | 半日 |

### Q1: Same-Speaker Reconstruction

| ID | タスク | 所要 |
|---|---|---|
| **10-Q1a** | WavLM content extraction pipeline (L14-18 → k-means) | 1日 |
| **10-Q1b** | F0/energy/VUV extractor (pYIN) | 1日 |
| **10-Q1c** | Speaker embedding precompute (WavLM-SV) | 半日（完了済み） |
| **10-Q1d** | Baseline mel predictor (Conformer/UNet) 学習 | 3日 |
| **10-Q1e** | Q1 評価 (WER, UTMOS, SECS) | 半日 |

### Q2: Speaker Swap (same-text pair)

| ID | タスク | 所要 |
|---|---|---|
| **10-Q2a** | same-text pair データセット構築 (VCTK parallel pairs) | 1日 |
| **10-Q2b** | speaker adversarial loss 実装 | 1日 |
| **10-Q2c** | same-text pair 学習 + 評価 | 2日 |

### Q3: Nonparallel Generalization

| ID | タスク | 所要 |
|---|---|---|
| **10-Q3a** | timbre-shift augmentation 実装 | 1日 |
| **10-Q3b** | 任意 source/target での評価 | 1日 |

### Rust 実装

| ID | タスク | 所要 |
|---|---|---|
| 10-R1 | BigVGAN v2 Rust/Candle 実装 | 1週間 |
| 10-R2 | WavLM Rust/Candle 推論 | 既存 (candle-transformers) |
| 10-R3 | pYIN Rust 実装 | 3日 |
| 10-R4 | streaming mel generation | 2日 |

---

## 更新履歴

- 2026-06-18: 初版作成。DAC latent-space VC の5つの失敗を受けた根本的再設計。
  設計原則: 「変換ではなく、話者条件付き再合成」。
