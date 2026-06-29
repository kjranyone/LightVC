# C2-2: ソース声質リーク抑制 — 設計書

> 2026-06-29. 受肉用途の必須機能。男性声質が変換音声に残る問題を構造的に解決する。

## 問題分析

### 現状のリーク経路

B1 adapterの出力にソース話者（特に男性）の特徴が残る理由:

```
経路1: q0 anchor
  source q0 [depth-0 RVQ] は content anchor として固定される。
  depth surgery実測: d0 = 12% の話者情報を含む。
  → 男性の低F0・広声道・粗さが q0 経由で構造的に保持される。

経路2: 残差チェーンの浅い層
  d1-3 は話者コア（68%）だが、adapter deltaが完全に覆いきれていない。
  source SECS = 0.238 → 約24%のソース話者性が残る。

経路3: 韻律・F0パターン
  男性のF0軌跡（低い中央値、狭いレンジ）が latent に焼き込まれている。
  adapterは音色を変えるが、ピッチパターンは source 由来。
```

### 目標

| 指標 | 現状 | 目標 |
|------|------|------|
| source SECS | 0.238 | **<0.10** |
| target SECS | 0.508 | **≥0.50** |
| CER | 0.47 | **≤0.50** (許容範囲) |
| gender classifier P(male) | ~0.7 (推定) | **<0.15** |

## 設計: 3層防御

### Layer 1: q0 Speaker Scrubbing（構造的）

q0から話者情報を除去するbottleneckモジュール。

```
q0_s [1024, T]
  → Linear(1024 → 64)           # speaker bottleneck (64次元に圧縮)
  → Linear(64 → 1024)           # 復元
  → q0_neutral [1024, T]

訓練:
  L_content: decode(q0_neutral) の音声理解度保持 (CER proxy)
  L_speaker_adv: speaker_classifier(q0_neutral) を GRL で逆伝播
             → bottleneck は話者情報を除去する方向に学習
```

**Speaker classifier**:
- 入力: q0 [1024, T] → mean pool → [1024]
- 出力: speaker logits [N_speakers]
- 訓練: 全訓練ペアの q0 から。q0だけで話者識別できることを先に確認。

**GRL (Gradient Reversal Layer)**:
```
forward: identity
backward: -λ * gradient
```
bottleneck出力 → GRL → speaker_classifier の順で繋ぐ。
classifierは話者を当てようとし、bottleneckは識別を困難にする方向に学習。

### Layer 2: Gender Leakage Penalty（学習時loss）

B1 adapter訓練に gender classifierベースのリークペナルティを追加。

```
output_audio = decode(z_q_adapted)
gender_prob = gender_classifier(ECAPA(output_audio))
L_gender = gender_prob  # P(male) を最小化
```

**Gender classifier**:
- ECAPA埋め込み [192] → Linear(192 → 1) → sigmoid
- 訓練: VCTK男女ラベル（speaker-info.txtから取得）でbinary分類
- 凍結してB1訓練に組み込み

**Loss全体**:
```
L = speaker_weight * L_spk      # target SECS向上
  + leak_weight * L_leak         # source SECS低下（既存）
  + gender_weight * L_gender     # P(male) 低下（新規）
  + stft_weight * L_stft         # 音響品質
  + delta_reg * L_delta          # 安定化
```

`gender_weight` は `leak_weight` と独立して調整可能。
汎用source leakageより、male特性の除去に特化できる。

### Layer 3: F0 Range Normalization（前処理）

エンコード前にソースF0を女性レンジへシフト。

```
source_audio [44.1kHz]
  → pyworld F0抽出 (frame_period=5ms)
  → F0_shift: log(F0) を男性中央値→女性中央値へ平行移動
  → pyworld resynthesis (F0 shifted, spectral envelope維持)
  → shifted_audio [44.1kHz]
  → DAC encode
```

これにより、latentに焼き込まれるF0パターンが女性レンジになる。
韻律・タイミングは保持（F0の輪郭は維持、絶対値のみシフト）。

**リスク**: WORLD再合成アーティファクト。ただしDAC encode/decodeを通すことで平滑化される。

## 実装計画

### Phase 1: Gender Classifier構築（最初にやる）

```python
# train_gender_classifier.py
# ECAPA埋め込み → binary gender分類器
# VCTK speaker-info.txt から男女ラベル取得
# accuracy > 95% を目標
```

これが最も効果的かつ最小変更。B1訓練にlossを1つ追加するだけ。

### Phase 2: q0 Scrubbing Module

```python
# train_q0_scrubber.py
# q0 → bottleneck(64) → q0_neutral
# speaker classifier + GRL で訓練
# standalone で訓練後、B1 pipelineに統合
```

### Phase 3: F0 Normalization

```python
# preprocess_f0_shift.py
# source_audio → F0 shift → shifted_audio
# B1訓練時のsource audioを差し替え
```

### Phase 4: 統合評価

```
構成A: B1 baseline (現状)
構成B: B1 + gender penalty
構成C: B1 + gender penalty + q0 scrubbing
構成D: B1 + gender penalty + q0 scrubbing + F0 norm
```

各構成で:
- source SECS（低下を目指す）
- target SECS（維持/向上を目指す）
- P(male)（低下を目指す）
- CER（許容範囲内）
- ABX: 「男性らしく聞こえるか」の知覚評価

## 判断基準

**Go**:
- source SECS < 0.15
- target SECS ≥ 0.45
- P(male) < 0.20
- CER 変動 ±0.05以内

**No-Go**:
- target SECS が 0.40を下回る
- CER が 0.60を超える
- 音質が主観的に著しく劣化

## ファイル計画

| ファイル | 用途 | Phase |
|---------|------|-------|
| `training/train_gender_classifier.py` | ECAPA→gender binary分類器 | 1 |
| `training/train_q0_scrubber.py` | q0 speaker scrubbing module | 2 |
| `training/preprocess_f0_shift.py` | F0正規化前処理 | 3 |
| `training/eval_source_leakage.py` | source SECS + P(male) + CER統合評価 | 4 |
| `training/train_phase3c_adapter.py` | 拡張: gender_weight loss追加 | 2 |

## リスク分析

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| q0 scrubbing が content を破壊 | 中 | 高 | CER監視 + content loss強化 |
| gender penalty が target品質も下げる | 中 | 中 | gender_weightを小さく開始(0.1) |
| F0 shift が不自然な声を作る | 低 | 中 | shift量を話者ペア毎に最適化 |
| GRL訓練が不安定 | 中 | 中 | λスケジューリング (0→1 ramp) |
