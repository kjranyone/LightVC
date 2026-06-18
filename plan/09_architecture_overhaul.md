# 09 — 学習設計改善計画：Flow Target と Speaker Conditioning の見直し

> **作成日**: 2026-06-18  
> **ステータス**: 📄 計画中（実装未開始）  
> **重要度**: **P0**（出力品質に直結）  
> **前提**: Phase C UTTE 学習完了後の SECS=0.142（目標 >0.70）という評価結果に基づく

---

## 1. エグゼクティブサマリ

Phase C UTTE（108.7M params, 30K step）を学習し、内部指標 spk cos\_sim≈1.0 を達成したが、外部評価 SECS=0.142 に終わった。原因調査の結果、**DAC 潜在空間の話者分離不足**（話者間分散/話者内分散 ≈ 0.85）を観測した。

ただし、「DAC では VC 不可能」という結論は**飛躍**である。より堅い解釈は：

> **非パラレル flow target の設計が破綻している。**  
> `v_target = z_tgt - z_src`（異なる発話・異なる話者）は、全速度の **98.9% が予測不可能な内容差**であり、モデルが学習できる話者成分は 1.1% に過ぎない。いかなる表現空間でも、この target 設定では FM loss は plateau する。

本計画は、**DAC latent pipeline を維持したまま**、学習設計と speaker conditioning を修正する方向で段階的な改善を定義する。Mel + BigVGAN への完全移行は LightVC のブランド要件（軽量・codec-space・Rust native）に反するため、**将来の別案**として凍結する。

---

## 2. 問題分析

### 2.1 観測事実

```
[Phase C UTTE 学習結果]
  内部 spk cos_sim: ≈1.0 (完璧)
  外部 SECS (ECAPA-TDNN): 0.142 (目標 0.70 の 20%)
  WER: 0.000 (内容保持は完璧 = 話者変換が起きていない)
  fm loss: 8.6 → 8.6 (全期間横ばい)
```

### 2.2 観測データ：速度ベクトルの分解

```
v_target = z_tgt - z_src を 3 成分に分解:

  v_full (実測全体):                std = 2.890  abs_mean = 2.261
  v_speaker_shift (話者平均差):      std = 0.309  abs_mean = 0.223
  v_content_residual (内容差の残差): std = 2.878  abs_mean = 2.258

  → 全速度分散の 1.1% のみが話者成分
  → 98.9% は異なる発話間の内容差（予測不可能）
```

### 2.3 根本原因の解釈

**× 悪い解釈**: 「DAC 潜在空間に話者情報が弱い → DAC では VC 不可能」

**✓ 堅い解釈**: 「任意の target 発話 latent を flow target にした学習設計が不適切。異なる内容の発話間の速度ベクトルは、内容差という予測不可能なノイズに支配される。FM loss が plateau (≈ var(v\_target)) するのは、どんな表現空間でも当然。DAC latent の話者分離が弱いことは、**悪化要因**ではあるが**根本原因ではない**。」

### 2.4 補強：なぜ内部 spk metric と外部 SECS が乖離したか

内部 SpeakerEncoder（DAC latent 上の mean+std pooling → 2層MLP）は、モデル自身が最適化対象となるため、velocity 予測の微小な変化で cosine similarity を容易に最大化できる。つまり**メトリック gaming** が起きた。これは speaker encoder の表現力不足と、学習ループ内 evaluator と外部 evaluator の不一致が合わさった結果。

### 2.5 WER = 0.000 の意味

v\_pred の振幅が小さすぎ（abs\_mean ≈ 0.24、v\_target の 8%）て、z\_src がほぼそのまま decode されている。内容保持が「完璧」なのは話者変換が起きていない証拠。

---

## 3. SOTA サーベイからの知見（限定引用）

### 3.1 LightVC に関連する知見のみ抽出

| 知見 | 出典 | LightVC への示唆 |
|---|---|---|
| Target に任意発話を使わず、timbre-shifted source を使う | Seed-VC | flow target を content-matched にすべき |
| k-means 離散化で content/timbre が暗黙分離される | EZ-VC, kNN-VC | DAC RVQ codebook でも同様の分離が可能かもしれない |
| WavLM L6 は content+speaker 混合、上位層は content 中心 | kNN-VC, VEVO | 補助 loss / 診断指標として WavLM が有用 |
| CAM++ / WavLM-SV で speaker conditioning が SECS 0.71 達成 | Takin-VC, Seed-VC | speaker encoder の交換は high-impact / low-risk |
| Self-reconstruction を先に確立し、その後に cross-speaker へ | R-VC, AdaptVC | same-speaker reconstruction ができていない状態で cross-speaker は無理 |

### 3.2 SOTA が全て Mel+BigVGAN であることについて

SOTA システムが Mel+BigVGAN に収束しているのは事実だが、これらは 12K-500K 時間・500K-1.35M step の学習規模であり、LightVC の制約（44h データ、30K step、22GB GPU、軽量 converter）とは桁違い。「SOTA がやっているから」というだけで DAC pipeline を捨てるのは短絡的。まずは現行規模でできる改善を尽くす。

---

## 4. 設計方針

### 4.1 維持するもの

| コンポーネント | 理由 |
|---|---|
| **DAC encoder/decoder** | LightVC の中核。軽量・高品質・MIT・Rust 実装済み |
| **Codec-space pipeline** | ブランド要件。「DAC latent で VC する」ことが USP |
| **CausalResBlock + FiLM** | アーキテクチャ骨格は健全 |
| **1-NFE rectified flow** | 推論 1 ステップは必須。target 設定を直せば機能する |
| **Pure Rust inference** | DESIGN.md のハード制約 |
| **FRC streaming** | 実装済み・稼働中。DAC space でそのまま利用可能 |

### 4.2 見直すもの

| 項目 | 現行 | 問題 | 検討方向 |
|---|---|---|---|
| **Flow target** | 任意発話の z\_tgt | 内容差がノイズ | content-matched target / reconstruction / cycle loss |
| **Speaker encoder** | 独自 (DAC latent上 mean+std) | 表現力不足・gaming | WavLM-SV を teacher とした蒸留（推論時は DAC SpeakerEncoder のみ） |
| **学習内評価** | 独自 SpeakerEncoder cos\_sim | 外部 SECS と乖離 | WavLM feature による補助診断 loss |
| **学習データ** | VCTK 44h | 規模不足 | LibriTTS 100h 拡張（段階的） |

### 4.3 凍結するもの

| 項目 | 理由 |
|---|---|
| **Mel + BigVGAN 完全移行** | LightVC ブランド要件（軽量 codec-space）に反する。将来の別案探索として記録のみ残す |
| **DAC 廃止** | 軽量高品質の中核。廃止の根拠が不十分（target 設計の問題と混同していた） |

---

## 5. 改善計画

### Phase 0: 事前検証（実装前の前提確認）

> **目標**: 後続 Phase の中核仮説を検証し、実装前に Go/No-Go を判断する

Phase B/C の前提となる3つの仮説を、学習前に検証する。いずれかが崩れた場合、該当 Phase の方針を変更する。

| ID | 仮説 | 検証方法 | 所要 |
|---|---|---|---|
| **09-01** | DAC SpeakerEncoder が WavLM-SV embedding を再現できる | 現行 SpeakerEncoder を WavLM-SV teacher で軽量訓練し、**held-out 話者**で評価。loss は **(a) L2-normalize後の cosine loss (b) pairwise contrastive loss** の両方。EER 測定の trial 構成: positive = same speaker different utterance, negative = different speaker。held-out 話者は各5発話以上（不足時は除外）。閾値を cosine sim で 0.0-1.0 を sweep し FAR=FRR となる点を EER とする。train/eval speakers 厳密分離 | 半日 |
| **09-02a** | same-text parallel pair で内容差ノイズが大幅減少する | VCTK same-text pair (同一テキスト・別話者) について、WavLM content feature で DTW 整列後の v\_target を分解。speaker\_shift / content\_residual 比率を再測定。現行（非アラインメント）の 1.1% から何%に改善するか確認 | 1日 |
| **09-03** | LibriTTS に same-text pair が十分存在する | LibriTTS train-clean-100 の normalized text match 件数を counting。VCTK のみで足りる場合は LibriTTS 拡張を後回しにする | 半日 |

**判断基準**:
- 09-01: held-out utterances で `mean cos(pred_embed, teacher_embed)` >0.5 かつ EER <20% → DAC SpeakerEncoder 蒸留は有力 → Phase B へ
- 09-01: 同上 cosine <0.3 → DAC latent から話者表現を抽出するのは困難 → Phase B 破棄、別アプローチ（推論時 WavLM-SV 必須化または γ 凍結再評価）を検討
- 09-02a: selected DTW cost (09-02c で確定) 後の speaker\_shift ratio >10% → C-4 は有力 → Phase C の主軸へ
- 09-02a: 同上 ratio <5% → same-text pair でも不十分 → C-1 (timbre shift) を主軸に
- 09-03: LibriTTS same-text pair >1000件 → LibriTTS 拡張に意味あり → Phase C データ拡張へ
- 09-03: 同上 <100件 → VCTK のみで進行、LibriTTS は augmentation 用

### Phase A: 診断（学習不要・即時実行）

> **目標**: 現行モデルの限界を正確に把握し、後続フェーズの優先順位を確定する

| ID | タスク | 内容 | 所要 |
|---|---|---|---|
| 09-A1 | velocity\_scale sweep | scale=1/3/5/10/20 で SECS を測定。v\_pred の方向に信号があるか診断 | 30 min |
| 09-A2 | timbre\_shift→src overfit test | **tiny overfit test**: z\_0 = timbre\_shift(z\_src), z\_1 = z\_src, reference = z\_src。少数サンプル(10件)で 1000 step overfit させ、FM loss が ≈0 に収束するか確認。「既知の逆変換を overfit できるか」が問い。generalization は測らない | 1h |
| 09-A3 | target/source leakage 測定 | 変換後の SECS(converted, source\_speaker) を測定。元話者が漏れていないか | 30 min |
| 09-A4 | v\_pred 方向分析 | v\_pred と v\_speaker\_shift の cosine similarity を測定。方向が合っているか | 1h |

**判断基準**:
- 09-A1 で SECS が 0.3 以上 → 方向は正しい、振幅不足 → Phase B へ
- 09-A1 で SECS が 0.14 から変化なし → 方向が不正 → Phase C が必須
- 09-A2 で timbre\_shift→src の FM loss が下がる → FM 自体は機能する → target 設定が原因と確定
- 09-A2 で FM loss も横ばい → FM 自体が DAC latent で機能しない → C-5 (RVQ) や γ 再評価を視野

### Phase B: Speaker Conditioning 強化 — WavLM-SV Teacher 蒸留

> **目標**: DAC-latent SpeakerEncoder を WavLM-SV の表現空間に蒸留し、推論時の軽量性を維持したまま speaker conditioning の質を向上

**アプローチ**: WavLM-SV を推論時に使うと zero-shot VC のたびに WavLM forward が必要になり、LightVC の軽量要件に反する。代わりに**学習時に WavLM-SV を teacher として使い、DAC-latent SpeakerEncoder を distill する**。

```
[学習時]
  wav_ref → WavLM-SV (frozen) → teacher_embed [256]   ← ground truth
  z_ref   → DAC SpeakerEncoder (trainable) → pred_embed [256]

  loss_distill = α·(1 - cos(pred_normalized, teacher_normalized))   ← cosine loss
               + β·pairwise_contrastive(pred, batch_speakers)       ← contrastive
  ※ L2 normalize 後の cosine geometry が重要。MSE 単独では embedding の方向情報が無視される。

  ※ FiLM conditioning は pred_embed を使う（推論時と同一経路）

[推論時]
  z_ref → DAC SpeakerEncoder → pred_embed [256] → FiLM
  ※ WavLM-SV 不要。変更なし。
```

| ID | タスク | 内容 | 所要 |
|---|---|---|---|
| 09-B1 | WavLM-SV embedding 事前計算 | 全コーパスの参照音声から WavLM-SV 256-dim embedding を抽出・キャッシュ | 半日 |
| 09-B2 | SpeakerEncoder に蒸留 loss 追加 | converter.py: SpeakerEncoder の forward 出力に対し cosine+contrastive loss を追加。**蒸留 loss は train-only module とし、推論時の SpeakerEncoder アーキテクチャは不変（重み key 構成変更なし）。`export_weights.py` は FlowConverter の推論重みのみを出力し、teacher / loss 用 module は除外。Rust converter.rs 側の更新は不要。** AGENTS.md「converter.py と converter.rs はキー名完全一致」ルール遵守 | 1日 |
| 09-B3 | Phase B (warm-start) 再学習 | 蒸留 loss + reconstruction loss で bottleneck AE を再学習 | ~2h |
| 09-B4 | Phase C 再学習（target 設定は現行維持） | **ablation**: speaker conditioning 改善のみで効果を測定。target 設計は破綻したままなので、効果が出れば儲けもの | ~2h |
| 09-B5 | 評価 | SECS / UTMOS / WER 測定 | 1h |

**位置づけ**: ablation 実験。破綻した target 設定のままなので、speaker conditioning 改善だけで内容差ノイズを越える可能性は低い。効果が出れば Phase C の前提として活用し、出なければ Phase C が主戦場となる。

**期待効果**: SECS 0.14 → 0.2-0.3 程度（控えめ）。蒸留により SpeakerEncoder の表現空間は改善するが、velocity target のノイズ問題は未解決のため。

**リスク**: 低。DAC pipeline は不変、SpeakerEncoder の学習方法のみ変更。推論時のアーキテクチャは完全に同一。

**判断ポイント (09-B5)**:
- SECS > 0.25 → speaker conditioning 改善が有効。Phase C の前提として採用
- SECS < 0.25 → Phase C (target 設計修正) が主戦場。Phase B の蒸留は Phase D に統合して継続

### Phase C: Flow Target 設計の修正（中核改善）

> **目標**: 予測不可能な内容差ノイズを除去し、学習可能な target を構築

この Phase は複数の代替案を並行検討する。実装コストと効果のバランスで採用を決める。

#### 案 C-1: Timbre-Shifted Source Target（Seed-VC 方式）

**idea**: target を別話者の実発話ではなく、**source の timbre-shifted 版**にする。

```
z_src = DAC.encode(src_wav)
z_shifted = DAC.encode(timbre_shift(src_wav))  # pitch/formant shift
v_target = z_shifted - z_src

# speaker condition（蒸留済み DAC SpeakerEncoder への入力）:
#   z_shifted (= timbre_shift 後の DAC latent) を参照 latent として渡す
#   → SpeakerEncoder(z_shifted) で shift 後の timbre を表現
#   → 学習時の teacher は WavLM-SV(shifted_wav)
```

- 内容は同一 → 内容差 ≈ 0 → v\_target は純粋に timbre 変化成分
- timbre shift は信号処理（PSOLA + formant filter）なので teacher 不要
- speaker condition は shift 後音声から抽出（上記）。これにより「与えられた参照話者に近い timbre への変換」を学習する
- Seed-VC と同じアプローチだが、潜在空間が DAC で generator が軽量 conv

**リスク**: 
- timbre shift の変換が狭い（pitch/formant のみ）ため、学習できる変換の幅が限られる。
- **train/inference 分布ズレ**: 学習では z\_shifted (synthetic pitch/formant shifted speaker) を参照するが、推論では実話者の実音声を参照する。PSOLA/formant shift の synthetic speaker embedding が実話者 embedding 空間と連続につながる保証がない。augmentation としては有用だが、zero-shot target speaker 変換の**主学習 objective としては弱い**。C-4 (same-text real pair) と組み合わせることを前提とする。

#### 案 C-2: Denoising FM + Cross-Speaker Fine-tune

**idea**: 2 段階学習。Stage 1 では** denoising FM** で FM が DAC latent で機能することを確証する。identity FM (v=0) は自明なので使わない。

Stage 1 (denoising FM):
```
z_clean = DAC.encode(src_wav)
z_noisy = z_clean + ε   (ε ~ N(0, σ²), σ = 0.1-0.3 程度)
z_0 = z_noisy, z_1 = z_clean
v_target = z_1 - z_0 = -ε
→ 非自明な非零 target。FM がノイズ除去方向を学習できるかを検証
→ vel_proj が非自明な重みに warm-start される
```

Stage 2 (cross-speaker fine-tune):
```
Stage 1 の checkpoint から開始
cross-speaker pair (C-1 の timbre-shifted target または C-4 の same-text pair)
fm loss weight を下げ、spk loss / content loss を中心に
```

**リスク**: Stage 1 (denoising) が cross-speaker 変換と性質が異なるため、transfer する保証はない。C-4/C-1 と組み合わせる。

#### 案 C-3: Cycle Consistency Loss（補助的）

**idea**: target latent を直接予測する代わりに、cycle 一貫性で話者変換を学習する。

```
z_src --convert(spk_tgt)--> z_out1 --convert(spk_src)--> z_out2
loss_cycle = L1(z_out2, z_src)
```

- target として正確な z\_tgt を要求しない
- 話者変換が可逆であることを通じて暗黙的に学習

**単体では identity collapse する**（convert(x, any\_spk)=x が安定解）。以下の**制約をセットで適用**することが前提:

```
必須の併用 loss:
  1. WavLM-SV speaker loss: SECS(converted, target_spk) を最大化
     → identity では target 話者と一致しないため、penalty がかかる
  2. Source speaker margin loss: max(0, margin - |emb(converted) - emb(source)|)
     → source から遠ざかりすぎるのではなく、margin 以上近づかないようにする
     → 単純な距離最小化は「不自然音で source から遠ざかる」を報酬にしてしまうため margin loss が必須
  3. Identity loss (orthogonal): 同一話者変換時は z_out ≈ z_src を要求
     → convert(src, src_spk) = src を保証
```

**リスク**: 学習が 2 倍の forward pass を必要とする。上記 3 loss のみでは collapse を防げない可能性もあり、効果の予測が難しい。

#### 案 C-4: Same-Text Parallel Pair Target（最直接的解決策）

**idea**: VCTK の same-text parallel pair（同一テキスト・別話者朗読）を使い、**内容差を最小化した target** を構築する。

**前提条件（必須）**: same text ≠ same timing。話速・ポーズ・強勢・音素長が異なるため、frame-wise `v_target = z_tgt - z_src` は** timing/prosody residual に支配される**。現行 `train_flow.py` は共通長に crop するのみでアラインメントなし。**DTW / forced alignment / latent alignment が前提条件**。Phase 0 (09-02) で DTW 後の speaker\_shift/content\_residual 比率を確認してから実装する。

**DTW cost の選定（重要）**: DAC latent 同士の Euclidean DTW は、話者差そのものに引っ張られて phoneme alignment にならない。**content-only 特徴量で整列させる必要がある**。優先候補:

| 優先 | cost source | 理由 |
|---|---|---|
| **1** | WavLM 上位層 (L14-18) feature 上の DTW | content-dominant だが、どの層が最適かは Phase 0 (09-02a) で L14〜L18 を比較し、最も residual が下がる層を採用 |
| **2** | Forced alignment (Montreal Forced Aligner) | テキスト→音素タイミングが正確。VCTK はテキスト+音声があるので利用可 |
| **3** | ASR/PPG (HybridFormer, Whisper encoder) feature 上の DTW | content-only。外部モデルが必要 |

Phase 0 (09-02) では、各 cost source で DTW を行い、整列後の v\_target を分解して最も speaker\_shift 比率が高いものを採用する。

```
[VCTK same-text pair pipeline]
  speaker A の朗読 → z_src [1024, T_a]
  speaker B の朗説 → z_tgt [1024, T_b]
  
  Step 1: DTW alignment で T_a ≠ T_b を解決
    → z_tgt_aligned [1024, T_a] (z_src と時間軸一致)
  
  Step 2: v_target = z_tgt_aligned - z_src
    → 内容差は「話者固有の発音癖」程度に縮小
    → v_target の話者成分比率が大幅に上昜
  
  Step 3: 通常の flow matching 学習
```

- VCTK は全話者が同じテキスト群を朗読するため、大量の same-text pair が存在
- **本次問題（内容差ノイズ）に最も直接的に効く**（alignment ありき）
- cross-speaker pair の自然な一般化でもある（同じ内容を別話者が話す = VC の定義）

**リスク**: 
- same-text pair でも発音タイミングのズレにより内容差は完全にはゼロにならない。DTW で位置を揃えた後でも residual が残る。
- DTW の品質が v\_target の質を左右する。粗い DTW（frame-level Euclidean）より、latent space での DTW または forced alignment (Montreal Forced Aligner 等) が望ましい。
- Phase 0 (09-02) で DTW 後の比率を確認するまでは、効果を確定しない。

#### 案 C-5: DAC RVQ Depth Separation

**idea**: DAC の 9 codebook を coarse (content+timbre) / fine (texture) に分け、coarse のみ変換する。

```
z_src → DAC quantizer → 9 codebooks [Nq, T]
  codebooks 1-3 (coarse): converter で変換
  codebooks 4-9 (fine): passthrough
→ 変換すべき成分だけを target にする
```

- すでに Phase 3 (Progressive RVQ-depth FM heads) として ARCHITECTURE に記載済み
- content/timbre 分離の軸として RVQ depth を使う
- 追加の pre-trained model 不要（DAC 自体の機能）

**リスク**: DAC quantizer が continuous latent とどれほど整合するか。encode path の Rust 実装が必要（~100 LOC）。

#### 案の優先順位

| 優先 | 案 | 理由 |
|---|---|---|
| **1** | **C-4 (same-text parallel pair)** | 本次問題（内容差ノイズ）に最も直接的。DTW alignment が前提（Phase 0 で検証） |
| **2** | C-1 (timbre-shifted target) | augmentation として有用。C-4 の補助。単体では train/inference 分布ズレ |
| **3** | C-2 (denoising FM → cross-speaker) | FM 機能確認の前提として有用。C-4/C-1 の warm-start として機能 |
| **4** | C-5 (RVQ depth) | LightVC 独自性が高い。DAC quantizer 実装後に評価 |
| **5** | C-3 (cycle) | 単体では collapse リスク。margin loss + speaker loss + identity loss とセット |

### Phase D: 学習補助としての WavLM 活用（推論コスト増なし・学習コスト大）

> **目標**: 学習時のみ WavLM を使い、推論時の DAC pipeline を維持

**学習コストに関する重要事項**: WavLM は波形域モデルのため、converted latent に WavLM loss を掛けるには **z\_pred → DAC decode → waveform → 16kHz resample → WavLM** の計算経路が必要。これは学習ループ内に**微分可能 DAC decode と 16kHz resample** を組み込むことを意味する。

**torch graph 内に限定**: gradient を converter に返すため、全工程が `torch.autograd` の計算グラフに乗っている必要がある。**librosa は graph 外なので不可**。`torchaudio.functional.resample` または畳み込みベースの resample を使う。DAC decoder は元々 PyTorch モデルなので、そのまま backprop 可能。

**メモリ/速度見積りは楽観視しない**: frozen WavLM でも入力 wav まで gradient を戻すため activations が全層で必要。DAC decoder も秒数・batch・frame length で activation が大きく増える。下記見積りは**暫定値**であり、Phase D 実装前に 1 batch profiling で実測する。

| コンポーネント | メモリ追加見積り(暫定) | 計算追加見積り(暫定) |
|---|---|---|
| DAC decoder activations (backprop 用) | ~500MB-1GB (frame length 依存) | ~10-20ms/step |
| torchaudio.functional.resample (44.1k→16k) | ~50MB | ~1ms/step |
| WavLM-SV activations (backprop 用) | ~500MB-1GB | ~5-10ms/step |
| **合計追加(暫定)** | **~1-2GB** | **~15-30ms/step** |

→ 22GB GPU でも batch\_size や frame length の調整が必要な可能性あり。**Phase D の最初のタスクは profiling** (09-D0) とする。

| ID | タスク | 内容 | 所要 |
|---|---|---|---|
| **09-D0** | **1-batch profiling**: 微分可能 DAC decode + torchaudio resample + WavLM forward/backward のメモリ・速度を実測 | batch=4/8/16, frame=100/200 で sweep。OOM の有無、step time 増加分を記録 | 半日 |
| 09-D1 | WavLM-SV speaker loss 実装 (torch graph 内) | z\_pred → DAC decode → wav → `torchaudio.functional.resample` → WavLM-SV → cosine sim loss | 2日 |
| 09-D2 | WavLM content preservation loss 実装 | WavLM 上位層 (L14-18) で content similarity を監視 | 1日 |
| 09-D3 | Phase C best model に統合して再学習 | target 修正 + WavLM 補助 loss の組み合わせ | ~6-8h (学習時間増含む) |

**重要**: WavLM は学習時のみ使用。推論時には一切不要。`export_weights.py` が FlowConverter の重みのみをエクスポートし、WavLM/DAC decoder は Python 学習環境に残留する。

---

## 6. 凍結：Mel + BigVGAN 移行（別案扱い）

### 6.1 記録

Mel-spectrogram + BigVGAN への完全移行は、SOTA システムの標準構成である。ただし：

1. LightVC のブランド要件（軽量 codec-space VC）に反する
2. BigVGAN v2 (122M params) は DAC decoder (~76M params) より重い
3. Rust/Candle 実装が未存在（~500-800 LOC の新規作成が必要）
4. Phase B/C の改善結果を見る前に判断するのは早急

### 6.2 再評価の条件

以下の**全て**を満たした場合に再評価する：

- Phase B + Phase C の全案を実施しても SECS < 0.50
- DAC latent pipeline の限界が複数の target 設計で一貫して確認された
- BigVGAN Rust 実装のコストを許容できるリソースが確保された

---

## 7. 学習計画サマリ

### 7.1 データ

| Phase | データ | 時間 | 備考 |
|---|---|---|---|
| Phase 0 | VCTK | 44h | 事前検証のみ、学習なし |
| Phase A | VCTK | 44h | 診断のみ、学習なし |
| Phase B | VCTK | 44h | speaker encoder 蒸留 |
| Phase C | VCTK (same-text pair) | 44h | target 設計修正。LibriTTS は 09-03 で pair 件数確認後に判断 |
| Phase D | Phase C と同一 | — | WavLM 補助 loss 追加（学習コスト増は 09-D0 profiling で実測後確定） |

### 7.2 評価プロトコル（held-out speaker split）

全ての SECS/UTMOS/WER 評価は **training speakers と held-out speakers を厳密に分離**する。SpeakerEncoder が speaker ID 記憶に逃げるのを防ぐため。

- **VCTK (109 speakers)**: 90 train / 19 held-out に分割
- **LibriTTS (2456 speakers)**: 2000 train / 456 held-out に分割（拡張時）
- 評価は held-out speakers のみで実施
- 評価 manifest も held-out speakers の pair のみで構成

### 7.3 実行順序と判断ポイント

```
Phase 0 (事前検証)
  ├─ 09-01: DAC SpeakerEncoder held-out 検証
  ├─ 09-02a: WavLM DTW 後 variance 再測定
  ├─ 09-02b: MFA feasibility (02a 結果次第)
  ├─ 09-02c: variance report → DTW cost 確定
  └─ 09-03: LibriTTS text-match counting
        │
        ▼
   [DP-0] 中核仮説の検証結果
        ├─ SpeakerEncoder 再現性 OK + same-text 比率 OK → Phase A/B/C へ
        ├─ SpeakerEncoder 再現性 NG → Phase B 破棄、別案検討
        └─ same-text 比率 NG → C-1 (timbre shift) を主軸に
        │
        ▼
Phase A (診断)
  ├─ A1: velocity_scale sweep
  ├─ A2: timbre_shift→src FM test
  ├─ A3: leakage 測定
  └─ A4: v_pred 方向分析
        │
        ▼
   [DP-A] v_pred に方向信号があるか?
        ├─ Yes → Phase B へ (蒸留で conditioning 強化)
        └─ No  → Phase C を優先 (target 設計修正が必須)
        │
        ▼
Phase B (WavLM-SV teacher 蒸留) ← ablation
   + Phase C (target 設計修正)
   ※ B と C は並行実施可能
        │
        ▼
   [DP-BC] SECS > 0.50 達成? (held-out speakers で評価)
        ├─ Yes → Phase E へ (★研究の核心★)
        └─ No  → Phase C の残案を実施、γ(凍結)を再評価
        │
        ▼
Phase E (★分離性・表現力検証★) ← LightVC の研究価値の核心
  詳細: docs/INNOVATION.md §3 (Q2)
        │
        ▼
Phase D (WavLM 補助 loss) ← E 結果を見て fine-tune判断
```

### Phase E: 分離性・表現力検証（研究の核心）

> **目標**: depth\_strengths で音色・ブレス・質感を独立に制御できるかを実証する。  
> これが確認できれば、DAC latent-space VC の優位性が初めて証明される。  
> 詳細な背景は [docs/INNOVATION.md](../docs/INNOVATION.md) 参照。

| ID | タスク | 内容 | 所要 |
|---|---|---|---|
| **09-E1** | depth\_strengths 分離性テスト | Phase C モデルで depth\_strengths を (1,0,0) / (1,1,0) / (0,1,1) 等に変え、変換後の音声を比較。スペクトル傾斜変化・F0 レンジ変化を定量測定 | 1日 |
| **09-E2** | ProsodyMode 制御テスト | PreserveSource / ImitateTarget / Blend(0.3/0.5/0.7) で抑揚が切り替わるか。F0 コンターチャンジを定量測定 | 1日 |
| **09-E3** | 男性→女性 ABX 評估 | held-out の男性話者→女性参照で変換。「女性らしさ」「ブレス感」「自然さ」を独立評価。各 depth\_strengths / ProsodyMode 設定で 5-10 サンプル | 2日 |
| **09-E4** | 分離性判定 + **判断 (DP-E)** | E1-E3 の結果をまとめ、depth\_strengths による独立制御が成立しているか判定 | 1日 |

**DP-E 判断基準**:

| 結果 | 判断 |
|---|---|
| coarse-only で音色変化、fine-only で質感変化が**独立に**確認 | **SUCCESS**: latent-space VC の分離制御を実証。論文・発表の核。Phase D で更に品質向上。 |
| 変換はできるが depth\_strengths による違いが聴取/測定で判別不能 | **PARTIAL**: 品質はあるが分離性なし。単一の velocity\_scale のみ提供。実用 VC としては成立。 |
| 変換自体が不十分（SECS < 0.50 のまま） | **FAIL**: Phase C の残案または γ 凍結解除 |

### 7.4 評価指標

| 指標 | Phase 0 | Phase A | Phase B | Phase C | Phase E | 最終目標 |
|---|---|---|---|---|---|---|
| SECS (held-out) | 診断 | 診断 | >0.20 | >0.50 | 確認 | **>0.70** |
| UTMOS | — | — | — | >3.0 | 確認 | **>3.5** |
| WER | — | 0% | <5% | <5% | <5% | **<5%** |
| fm loss | — | plateau | — | 下降確認 | — | — |
| SpeakerEncoder cos\_sim | 0.95 ✅ | — | — | — | — | — |
| same-text ratio (DTW後) | 1.5% ✅ | — | — | — | — | — |
| **depth\_strengths 分離性** | — | — | — | — | **coarse/fine 独立変化** | **独立制御** |
| **ProsodyMode 効き** | — | — | — | — | **F0/energy 有意差** | **制御可能** |

---

## 8. リスク評価

### 8.1 技術リスク

| リスク | 確率 | 影響 | 対策 |
|---|---|---|---|
| DAC SpeakerEncoder が WavLM-SV 表現を再現できない | 中 | 大 | Phase 0 (09-01) で先に検証。NG なら Phase B 破棄 |
| C-4 (same-text pair) で DTW 後も内容差ノイズが残る | 中 | 大 | Phase 0 (09-02) で先に測定。forced alignment (MFA) への切替を準備 |
| C-2 (denoising FM) が cross-speaker に transfer しない | 中 | 中 | C-4/C-1 と組み合わせる |
| 蒸留 DAC SpeakerEncoder が WavLM-SV 表現を十分模倣できない | 中 | 中 | 蒸留 loss weight 調整 + SpeakerEncoder の深さ増加 |
| timbre shifter (PSOLA) の品質が粗い | 中 | 中 | 既存実装 (`encode_corpus.py` の timbre shift) を改良 |
| VCTK parallel pair の数が不足 | 低 | 小 | LibriTTS 拡張（09-03 で pair 件数確認後に判断） |
| LibriTTS に same-text pair が少ない | 中 | 小 | VCTK のみで進行。LibriTTS は augmentation 用 |
| RVQ quantizer encode path の Rust 実装バグ | 中 | 中 | C-5 は最優先でないため、時間をかけて検証 |
| C-3 (cycle) が identity collapse する | 高 | 中 | 単体使用禁止。margin loss + speaker loss + identity loss とセット |
| Phase D で微分可能 DAC decode が OOM | 低 | 中 | batch_size 下げる (16→8)。22GB なら対応可能見込み |

### 8.2 ライセンス

| コンポーネント | ライセンス | 問題 |
|---|---|---|
| WavLM (学習時のみ) | MIT | ✅ 推論時に不要なので、配布物に影響しない |
| WavLM-SV (学習時のみ) | MIT | ✅ 同上 |
| DAC (継続使用) | MIT | ✅ |
| LibriTTS | CC-BY-4.0 | ✅ |
| VCTK | CC-BY-4.0 | ✅ |

### 8.3 ブランド整合性

| 要件 | 本計画での扱い |
|---|---|
| 軽量 codec-space VC | ✅ DAC pipeline 維持 |
| Pure Rust inference | ✅ WavLM は学習時のみ、推論は FlowConverter + DAC のみ |
| MIT license | ✅ |
| VC-teacher-free | ✅ 補助モデル表現監督は AGENTS.md で許可済み |
| 1-NFE real-time | ✅ rectified flow 維持 |

---

## 9. 残すコンポーネント・変更しないもの

| コンポーネント | 状態 | 備考 |
|---|---|---|
| DAC encoder/decoder | ✅ 維持 | 中核。変更なし |
| CausalResBlock + FiLM + Snake1d | ✅ 維持 | converter 骨格 |
| FlowConverter architecture | ✅ 維持 | target/conditioning の修正のみ |
| FRC streaming / overlap-add | ✅ 維持 | |
| Audio I/O / UI / CLAP plugin | ✅ 維持 | |
| Phase 3 progressive RVQ (案 C-5) | 📄 この計画に統合 | target 設計の一案として評価 |
| Phase 4 prosody factorization | 📄 保留 | 本計画完了後に再評価 |

---

## 10. タスクリスト

### Phase 0: 事前検証（実装前・必須）

| ID | タスク | 優先度 | 依存 | 所要 |
|---|---|---|---|---|
| **09-01** | DAC SpeakerEncoder held-out 蒸留検証 (cos\_sim / EER) | **P0** | なし | 半日 |
| **09-02a** | WavLM content DTW: WavLM L14-18 feature で DTW 整列後、v\_target の speaker\_shift/content\_residual 比率を測定 | **P0** | なし | 1日 |
| **09-02b** | MFA feasibility: Montreal Forced Aligner の導入コスト(インストール・辞書・実行時間)を評価。WavLM DTW の結果次第で必要性を判断 | P1 | 02a完了 | 半日-1日 |
| 09-02c | variance report: 02a/02b の結果をまとめ、C-4 実装の DTW cost を確定 | P1 | 02a (02b optional) | 半日 |
| 09-03 | LibriTTS text-match pair counting | P1 | なし | 半日 |

### Phase A: 診断（学習不要・即時）

| ID | タスク | 優先度 | 依存 | 所要 |
|---|---|---|---|---|
| **09-A1** | velocity\_scale = 1/3/5/10/20 で SECS 測定 (held-out) | **P0** | DP-0完了 | 30 min |
| **09-A2** | timbre\_shift→src overfit test | **P0** | DP-0完了 | 1h |
| 09-A3 | target/source leakage (SECS to source speaker) 測定 | P1 | DP-0完了 | 30 min |
| 09-A4 | v\_pred と v\_speaker\_shift の cos\_sim 測定 | P1 | DP-0完了 | 1h |

### Phase B: WavLM-SV Teacher 蒸留

| ID | タスク | 優先度 | 依存 | 所要 |
|---|---|---|---|---|
| 09-B1 | WavLM-SV embedding 事前計算・キャッシュ | P0 | 09-01 OK | 半日 |
| 09-B2 | converter.py SpeakerEncoder に蒸留 loss 追加 | P0 | B1 | 1日 |
| 09-B3 | Phase B (warm-start) 蒸留再学習 | P0 | B2 | ~2h |
| 09-B4 | Phase C 再学習 (現行 target, **ablation**) | P1 | B3 | ~2h |
| 09-B5 | SECS/UTMOS/WER 評価 (held-out) + **判断 (DP-B)** | P0 | B4 | 1h |

### Phase C: Flow Target 設計修正

| ID | タスク | 優先度 | 依存 | 所要 |
|---|---|---|---|---|
| **09-C1** | **案 C-4: alignment + same-text pair sampling + manifest schema 更新** — (a) 09-02c で確定した cost (WavLM/PPG DTW または MFA forced alignment) で整列 (b) `build_vctk_manifest.py` に same-text pair 出力追加: `(source, same_text_target, target_ref)` 3要素 schema (c) `encode_corpus.py` に pair encode 追加 | **P0** | 09-02c完了 | 2日 |
| **09-C2** | 案 C-4 で Phase C 学習 + 評価 (held-out) | **P0** | C1 + (B3 if 09-01 OK else baseline SpeakerEncoder) | ~3h |
| 09-C3 | 案 C-1: timbre-shifted target augmentation の実装 | P1 | C2 | 1日 |
| 09-C4 | 案 C-2: denoising FM warm-start の実装 | P1 | C2 | 1日 |
| 09-C5 | 案 C-5: DAC RVQ coarse/fine 分離の評価 | P2 | Rust quantizer実装 | 3日 |
| 09-C6 | 案 C-3: cycle consistency (margin loss + speaker + identity) | P2 | なし | 2日 |
| 09-C7 | 各案の比較評価 + **判断 (DP-C)** | **P0** | C2,C3,C4 | 1日 |

### Phase D: WavLM 補助 Loss（推論コスト増なし・学習コスト大）

| ID | タスク | 優先度 | 依存 | 所要 |
|---|---|---|---|---|
| **09-D0** | **1-batch profiling**: 微分可能 DAC decode + `torchaudio.functional.resample` + WavLM forward/backward のメモリ・速度実測。batch=4/8/16, frame=100/200 で sweep。OOM・step time 増を記録 | **P0** | なし | 半日 |
| 09-D1 | WavLM-SV speaker loss 実装 (torch graph 内) | P1 | D0 OK | 2日 |
| 09-D2 | WavLM content preservation loss 実装 | P1 | D1 | 1日 |
| 09-D3 | Phase C best model に統合して再学習 | P1 | C7, D2 | ~6-8h (学習時間増含む、09-D0 実測値参照) |

### Phase E: 分離性・表現力検証（研究の核心）

> LightVC の研究価値の核心。詳細は [docs/INNOVATION.md](../docs/INNOVATION.md) 参照。

| ID | タスク | 優先度 | 依存 | 所要 |
|---|---|---|---|---|
| **09-E1** | depth\_strengths 分離性テスト: (1,0,0)/(1,1,0)/(0,1,1) でスペクトル傾斜・F0レンジ変化を定量測定 | **P0** | DP-BC OK | 1日 |
| **09-E2** | ProsodyMode 制御テスト: PreserveSource/ImitateTarget/Blend で F0/energy コンターチャンジ測定 | **P0** | DP-BC OK | 1日 |
| **09-E3** | 男性→女性 ABX 評估: 「女性らしさ」「ブレス感」「自然さ」を独立評価 | **P1** | E1, E2 | 2日 |
| **09-E4** | 分離性判定 + **判断 (DP-E)** | **P0** | E1, E2 (E3 optional) | 1日 |

### 凍結

| ID | タスク | 優先度 | 備考 |
|---|---|---|---|
| 09-γ1〜γ7 | Mel + BigVGAN 移行 | **凍結** | §6 参照。再評価条件を満たすまで着手しない |

---

## 11. 意思決定ポイント

### DP-0: 事前検証結果（09-01〜03 完了後）

| 結果 | 判断 |
|---|---|
| `mean cos(pred_embed, teacher_embed)` >0.5 + selected DTW cost 後 speaker\_shift ratio >10% | 全 Phase 進行可 |
| `mean cos(pred_embed, teacher_embed)` <0.3 | Phase B 破棄。推論時 WavLM-SV 必須化または γ 再評価 |
| selected DTW cost 後 speaker\_shift ratio <5% | C-4 ではなく C-1 (timbre shift) を主軸 |

### DP-A: 診断結果判断（09-A1〜A4 完了後）

| 結果 | 判断 |
|---|---|
| velocity\_scale で SECS 上昇 + timbre\_shift FM が下がる | FM は機能する。target 設定と振幅が問題 → Phase B+C へ |
| velocity\_scale で SECS 変化なし | v\_pred 方向が不正 → Phase C を最優先 |
| timbre\_shift FM も横ばい | FM 自体が DAC latent で機能しない → C-5 (RVQ) や γ 再評価を視野 |

### DP-B: Speaker Encoder 蒸留効果判断（09-B5）

| 結果 | 判断 |
|---|---|
| SECS > 0.20 (held-out) | 蒸留が有効。Phase C の speaker conditioning 前提として採用 |
| SECS < 0.20 (held-out) | 蒸留のみでは不十分。Phase C が主戦場。蒸留 loss は Phase D に統合して継続 |

### DP-C: Target 設計の採用判断（09-C7）

各案 (C-1〜C-5) を比較し、最も効果の高いものを採用。複数案の組み合わせも可。

| 結果 | 判断 |
|---|---|
| いずれかの案で SECS > 0.50 (held-out) | その案を採用し Phase D へ |
| 全案で SECS < 0.50 (held-out) | DAC latent の限界が疑われる → γ(凍結) の再評価条件を満たす |

### DP-E: 分離性判定（09-E4）

| 結果 | 判断 |
|---|---|
| coarse/fine で独立した音響変化が確認（スペクトル・F0 で有意差） | **SUCCESS**: latent-space VC の分離制御を実証。Phase D で品質更向上。論文・発表の核。 |
| 変換はできるが depth\_strengths の違いが判別不能 | **PARTIAL**: 単一 velocity\_scale のみ提供。実用 VC としては成立するが、研究新規性は薄い。 |
| 変換自体が不十分 | **FAIL**: Phase C 残案または γ 凍結解除 |

---

## 12. 参考文献（限定）

### 直接関連

| 略称 | arXiv | 採用知見 |
|---|---|---|
| Seed-VC | 2411.09943 | Timbre-shifted target (C-1), CAM++ speaker (B) |
| kNN-VC | 2305.18975 | WavLM L6 での content/speaker 分離分析 |
| R-VC | 2506.01014 | Token dedup, shortcut FM, same-speaker reconstruction first |
| AutoVC | 1907.05842 | Bottleneck disentanglement（現行 FlowConverter の基盤） |
| CycleGAN-VC | 1711.11293 | Cycle consistency loss (C-3) |
| MeanVC2 | 2606.09050 | FRC + UTTE（現行アーキテクチャの基盤） |

### プリトレーンモデル

| モデル | HF ID | 用途 | 推論時 |
|---|---|---|---|
| WavLM-Large | `microsoft/wavlm-large` | 学習時: speaker/content 補助 loss | **不要** |
| WavLM-SV | `microsoft/wavlm-base-plus-sv` | 学習時: teacher embedding (蒸留用) | **不要** (DAC SpeakerEncoder が蒸留済み重みを保持) |
| DAC 44kHz | `descript/dac_44khz` | 推論: encoder + decoder | **必須** (現行維持) |

---

## 更新履歴

- 2026-06-18: 初版作成。Phase C UTTE 評価結果 (SECS=0.142) に基づく改善計画。
  - 当初「DAC 潜在空間の根本見直し」として作成したが、レビューで「DAC 廃止は飛躍。根本問題は非パラレル flow target の設計」と指摘され、方針を修正。DAC pipeline 維持、学習設計・speaker conditioning の改善に焦点を絞った計画に書き直し。
- 2026-06-18: 設計レビュー第2ラウンド対応 (8件修正 + AGENTS.md更新):
  - **[P0]** AGENTS.md の「Teacher蒸留は使わない」を「VC teacher蒸留は禁止 / 補助モデル表現監督は許可」に明確化。WavLM-SV 蒸留が制約違反ではなくなった。
  - **[P0]** Phase 0 (事前検証) を新設: DAC SpeakerEncoder held-out 検証 (09-01)、same-text pair DTW 後 variance 再測定 (09-02)、LibriTTS pair counting (09-03)。
  - **[P0]** Phase A の A2 を identity FM (v=0、自明) → timbre\_shift(src)→src (非自明変換) に修正。
  - **[P0]** C-4 の DTW alignment を「検討」→「前提条件」に格上げ。Phase 0 で検証後に実装。
  - **[P1]** C-1 に train/inference 分布ズレ (synthetic speaker embedding ≠ 実話者 embedding) の警告を追記。
  - **[P1]** Phase D の学習コストを正確に記載: 微分可能 DAC decode + 16k resample + WavLM forward で ~850MB / ~50% 学習時間増。
  - **[P1]** LibriTTS の same-text pair 豊富性を前提とせず、Phase 0 (09-03) で counting する方針に。
  - **[P2]** C-3 の source leakage penalty を margin loss に変更 (不自然音で source から遠ざかるのを防ぐ)。
  - **評価プロトコル**: 全評価で training/held-out speakers を厳密分離 (speaker ID 記憶対策)。
- 2026-06-18: 設計レビュー第3ラウンド対応 (8件修正):
  - **[P1]** C-4 の DTW cost を定義: DAC latent Euclidean は不可。WavLM content feature / forced alignment (MFA) / PPG を優先候補として明記。Phase 0 で各 cost の効果を比較。
  - **[P1]** Phase D の resample を torch graph 内 (`torchaudio.functional.resample`) に限定。librosa は graph 外で gradient が流れないため不可。
  - **[P1]** Phase D に 09-D0 (1-batch profiling) を追加。メモリ/速度見積りを暫定値に格下げ、実測前提に変更。
  - **[P1]** 09-01 の蒸留 loss を MSE 単独 → cosine loss + pairwise contrastive の両方に変更。speaker embedding は cosine geometry が重要。
  - **[P1]** A2 を tiny overfit test として明記: z\_0=timbre\_shift(z\_src), z\_1=z\_src の既知逆変換を少数サンプルで overfit させる。generalization は測らない。
  - **[P1]** C-2 から identity FM を完全削除。denoising FM (z\_noisy→z\_clean) のみに。
  - **[P2]** "Teacher-free" → "VC-teacher-free" に変更（補助モデル表現監督との混同回避）。
  - **[P2]** 09-C1 に manifest/index schema 更新 (`build_vctk_manifest.py` + `encode_corpus.py` の pair 出力対応) を追加。
- 2026-06-18: 設計レビュー第4ラウンド対応 (7件修正 — タスクリスト整合性):
  - **[P1]** 09-D0 (profiling) をタスクリストに反映。Phase D の最初のタスクを D0 に統一。
  - **[P1]** 09-C2 の依存を分岐明記: `C1 + (B3 if 09-01 OK else baseline SpeakerEncoder)`。Phase B 破棄時の C 実行経路を明確化。
  - **[P1]** 優先順位表の C-2 を "identity FM" → "denoising FM" に修正（本文と整合）。
  - **[P1]** 09-01 の EER trial 構成を明記: positive=same speaker diff utterance, negative=different speaker, 各5発話以上, 閾値 sweep。
  - **[P1]** 09-02 を 02a (WavLM DTW) / 02b (MFA feasibility) / 02c (variance report) に分割。
  - **[P2]** "09完了" → "DP-0完了" に修正（依存関係の明確化）。
  - **[P2]** Phase D サマリの "~50%増" を暫定表現に修正（09-D0 実測後確定）。
- 2026-06-18: 設計レビュー第5ラウンド対応 (3件修正 — 最終整合):
  - **[P2]** Phase 0 仮説表の 09-02 → 09-02a に統一（タスクリスト・判断基準と整合）。
  - **[P2]** A3/A4 の依存を「なし」→「DP-0完了」に統一（held-out split 等の Phase 0 成果に依存）。
  - **[P2]** DTW cost 候補の WavLM 層を「L14-18 固定」→「Phase 0 で L14〜L18 を比較し最適層を採用」に変更。
- 2026-06-18: 設計レビュー第6ラウンド対応 (2件修正 — 指標の精密化):
  - **[P2]** DP-0 判定基準の 09-02a を "DTW後" → "selected DTW cost (09-02c で確定) 後" に精密化。
  - **[P2]** 09-01 の cosine similarity を "held-out話者のcosine similarity" → "`mean cos(pred_embed, teacher_embed)` on held-out utterances" に明確化。
- 2026-06-18: 実験プロジェクト枠組みの導入 + Phase E 新設:
  - **Phase E**（分離性・表現力検証）を Phase D の前に新設。depth\_strengths で音色・ブレス・質感を独立に制御できるかが LightVC の研究価値の核心。
  - [docs/INNOVATION.md](../docs/INNOVATION.md) に革新価値・実験意義を整理。
  - Phase 0 実測結果を評価指標表に反映（09-01 cos=0.95 PASS、09-02a ratio=1.5% FAIL → C-1 主軸）。
  - DP-E（分離性判定）を意思決定ポイントに追加。
- 2026-06-18: 設計レビュー第7ラウンド対応 (3件修正):
  - **[P1]** 09-B2: 蒸留 loss を train-only module とし、推論時の SpeakerEncoder アーキテクチャ・重み key 構成を不変にする旨を明記。Rust 側更新不要・AGENTS.md キー名一致ルール遵守を明文化。
  - **[P2]** リスク表の C-2 を "identity FM" → "denoising FM" に修正（本文・タスクリストと整合）。
  - **[P2]** 09-C1 の "WavLM/PPG/MFA で DTW" → "WavLM/PPG DTW または MFA forced alignment" に修正（MFA は DTW ではなく forced alignment）。
