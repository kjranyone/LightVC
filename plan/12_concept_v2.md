# Plan 12: CONCEPT v2 — Codec Token Trajectory Translation

## 概要

LightVC の核心概念は維持しつつ、実験結果で鍛えた改訂版。

### 失敗した仮説（CONCEPT v1）

```
z_s = E(x_s)           # source codec latent
z_hat = Tθ(z_s, target) # 連続回帰で変換
y = D(z_hat)           # decode
```

**問題**: `z_hat ∉ M_codec`。DAC decoder は `E(real speech)` 由来の manifold 上では強いが、VC が作った中間点では壊れる。SECS ceiling ~0.16 で行き詰まり。

### 失敗した仮説（WORLD source-filter retrieval）

```
mc_s → kNN(bank_tgt) → mc_hat → WORLD synth
```

**問題**: 異テキスト bank からのフレーム独立 mcep 距離検索では、正しい対応を見つけられない。

200ペア評価（CI ±0.015）:
- retrieval: 0.34
- oracle rerank: 0.40
- DTW oracle (同テキスト): 0.365 ± 0.090

結論: retrieval と oracle rerank と DTW oracle が同程度。ボトルネックは検索順位や alignment ではなく、WORLD mcep/F0 合成表現そのものの話者多様性限界。

### CONCEPT v2 の主張

> LightVC は、codec latent を自由回帰する VC ではなく、
> **codec-valid な token trajectory を、content/unit/prosody/timbre に分解して
> 低遅延に翻訳する VC** である。

## 数理モデル

### RVQ depth 分離（Phase 1b で実証）

DAC (Descript Audio Codec) は Residual Vector Quantization を使用。
n_q 個の codebook（9）が階層的に符号化する。

**旧仮説（否定された）**: coarse=content, mid=speaker, fine=texture

**実測された depth 別話者寄与度**（200ペア、残差鎖保持再量子化）:

| depth を target→source に置換 | SECS (0.79から減少) | 話者情報量 |
|------|------|------|
| d0 | 0.79 → 0.35 (−0.44) | **最重要** |
| d1 | 0.79 → 0.54 (−0.25) | **強** |
| d2 | 0.79 → 0.65 (−0.14) | 中 |
| d3 | 0.79 → 0.69 (−0.10) | 小 |
| d4+ | 微減 (−0.02〜0.05) | 微 |

```
depth 0:     speaker + coarse acoustics + content（強く混在）
depth 1:     speaker/timbre（まだ強い）
depth 2:     speaker補助 + phonetic detail
depth 3-8:   residual texture/detail（話者情報は薄い）
```

**残差鎖保持の効果**: naive swap (0.19) → re-quant (0.54)。2.8倍改善。

### 変換モデル（改訂）

中心問題: **depth 0-2 の話者情報を、content を壊さずに target 側へどう移すか**

```
source:  x_s → DAC_encode → z_s → quantize → codes_s[9, T]

conversion (残差鎖保持):
  # target-led: target depth 0..K-1 を使用、残りを z_source から再量子化
  q_hat[0:K] = q_tgt[0:K]
  q_hat[K:9] = re_quantize(z_source - Σ q_hat[0:K])

  # source-led: source depth 0..K-1 を保持、残りを z_target から再量子化
  q_hat[0:K] = q_src[0:K]
  q_hat[K:9] = re_quantize(z_target - Σ q_hat[0:K])

output:   z_q = Σ q_hat[d]  →  y = DAC_decode(z_q)
```

Phase 1b oracle 結果（200ペア、DTW-aligned same-text）:

| config | SECS | 備考 |
|--------|------|------|
| src_k0 (全target再量子化) | 0.790 | 量子化上限 |
| src_k1 (src d0 + tgt rest) | 0.686 | content保持で高SECS |
| src_k2 (src d0-1 + tgt rest) | 0.416 | WORLD ceiling超え |
| tgt_k5 (tgt d0-4 + src rest) | 0.541 | 話者注入で高SECS |
| tgt_k3 | 0.265 | depth 0-2不十分 |
| source_all | 0.152 | 下限 |
| target_all (continuous) | 0.589 | DTW alignment有り |
| WORLD ceiling | 0.365 | — |

### なぜ off-manifold 問題が起きないか

1. 全 token は codebook から選ばれる → 常に valid
2. DAC decoder は token 列をそのまま decode → training distribution と一致
3. 連続 latent の補間・外挿を一切行わない

ただし、token 単体が codebook-valid でも、異なる発話・話者から組み替えた depth 間の同時分布や時間軌道が decoder の学習分布上にあるとは限らない。Phase 1 では token validity だけでなく **trajectory validity** を検証する。

### なぜ WORLD より可能性が残るか

1. DAC再合成上限が高い: target_all token decode で SECS ≈ 0.79
2. WORLD mcep/F0 表現の 200ペア天井は ≈ 0.36-0.40
3. source q0 + residual re-quantization の same-text oracle が 0.656-0.686
4. ただし cross-text retrieval は失敗済み。自由会話・歌唱には generator / tolerant decoder が必要。

## アーキテクチャ

### Enrollment（話者登録）

```
target speaker enrollment audio (10-30秒)
  → DAC encode → RVQ tokens
  → target timbre/profile extraction
  → optional diagnostic unit bank
  → runtime target profile / adapter state
```

phoneme/unit bank は診断・補助。自由会話/歌唱の本線は bank retrieval ではなく target-like residual trajectory generation。

### Inference（リアルタイム変換）

```
source wav (streaming)
  → DAC encode (causal, 10ms hop)
  → z_s, q0_s, RVQ tokens
  → Gθ(content/F0/energy/target_profile) → z_t_like
  → q0_s fixed + re-quantize residual depths 1..8
     or tolerant decoder adapter
  → DAC decode (streaming)
  → output wav
```

### レイテンシ予算 (<50ms)

| Component | Time |
|-----------|------|
| DAC encode (causal) | 5-10ms |
| Token lookup | <1ms |
| DAC decode (streaming) | 5-10ms |
| Audio buffer | 10-20ms |
| **Total** | **20-41ms** |

## WORLD実験からの知見（診断として活用）

| 実験 | SECS | 教訓 |
|------|------|------|
| frame kNN (best) | 0.34 | フレーム独立距離では弱い |
| oracle rerank | 0.40 | 候補内に正解はあるが、選択できない |
| DTW oracle | 0.365 ± 0.090 | 正解 target mcep を使っても話者差が大きく、WORLD 表現の上限が低い |
| Viterbi smoothing | 0.39 | 強制連続pathは逆効果 |
| register partition | 0.36 | 検索空間分割は逆効果 |
| L1回帰学習 | collapse | 連続回帰は常に崩壊 |

**結論**: codec token approach では以下を回避する
- フレーム独立距離 → unit-level matching へ
- 連続回帰 → 離散token選択へ
- 強制path制約 → 緩和制約へ
- WORLD mcep/F0 合成の話者依存上限 → neural codec decoder の表現力へ戻す

## 実装計画

### Phase 1: RVQ Analysis（完了）

1. VCTK 音声を DAC encode → RVQ tokens 抽出 ✅
2. depth別 token の話者識別力を測定 ✅
   - F-ratio on token indices ≈ 0（離散トークンには不適切）
   - 残差鎖保持再量子化による depth ablation で測定 ✅
3. **coarse/mid/fine 分離仮説は否定** → depth 0-2 が話者情報の核 ✅
4. depth swap oracle（naive + residual-chain-preserving）✅
5. trajectory validity 測定 ✅
6. **結果**:
   - naive swap: 0.19 ≈ random_mix → residual chain 崩壊
   - residual-chain-preserving re-quant: 0.54 (tgt_k5) → 有効
   - depth 0 が最重要（置換で −0.44 SECS）、depth 1 が次（−0.25）
   - 量子化上限: 0.790 (src_k0)、DAC decoder は強力

### Phase 2a: Multi-metric Oracle Sweep（same-text, aligned）

中心問題: speaker を握る depth 0-2 を content を壊さず target へ移す方法

3系統の depth assignment で K=1..5 sweep:

1. **Target-led**: target depth 0..K-1 + source rest re-quant（話者優先）
2. **Source-led**: source depth 0..K-1 + target rest re-quant（内容優先）
3. **Hybrid**: depth-level mixing（d0=tgt, d1=src, etc.）

多指標評価（SECS だけでは不十分）:
- **SECS**: ECAPA cosine (output vs target)
- **Content CER**: Whisper ASR → edit distance vs source text
- **F0 correlation**: log-F0 Pearson (output vs source)
- **Source leakage**: ECAPA cosine (output vs source)

**Go条件**:
- SECS >= 0.45 かつ content CER が source_all の CER + 0.05 以内
- SECS-content tradeoff curve で Pareto-optimal な config が存在

### Phase 2b: Cross-text Frame Retrieval（失敗）

same-text oracle (0.656) vs cross-text DTW (0.33)。フレーム独立検索・subseq DTW共に、異テキストenrollmentからは有効なz_t_likeを構築できない。

| method | same-text | cross-text |
|--------|-----------|------------|
| DTW oracle (DAC latent) | 0.686 | — |
| Wav2Vec2-DTW (layer 6) | 0.656 | 0.33 |
| frame NN (Wav2Vec2 hidden) | 0.576 | ~0.20 |
| unit-indexed retrieval | — | ~0.33 |

**結論**: retrieval路線は診断・補助。自由発話・歌唱の組合せ空間をbank検索でカバーできない。

### Phase 3: Target-like Latent Trajectory Generator（失敗、原因特定）

retrieval ではなく **生成** で z_t_like を作る。

```
source audio → DAC encode → z_s → q0_s (content anchor)
                                  ↓
content_s + f0_s + energy_s + target_timbre → Gθ → z_t_like
                                  ↓
q0_s 固定 + RVQ_requantize(z_t_like - q0_s) → z_q
                                  ↓
DAC decode → output audio
```

**試した生成アプローチ**:
- continuous latent + STE
- code CE (1024 classes)
- embedding MSE

**結果**: すべて失敗。latent cosine が 0.67 程度まで近づいても、decode 後 SECS は 0.03 程度まで崩れるケースがある。

**原因**: RVQ quantization cascade の感受性。oracle が強いのは exact / aligned target latent を使うためで、近似予測の微小誤差は残差鎖の後段で増幅され、全く違う音になる。

**結論**: 生成器をさらに大きくする前に、decoder 側が近似 residual latent を受けられるかを検証する。Phase 3 の本線は generator scaling ではなく decoder tolerance へ移る。

### Phase 3b: Decoder Adapter / Tolerant Decoder（現在の本線）

目的: 近似 `z_t_like` や soft residual embedding を、hard RVQ cascade で壊さず decode できる経路を作る。

実験:
1. **Noisy latent tolerance**: `z_t + σ ε` を decode し、SECS/CER/UTMOS の σ sweep を取る。
2. **Soft residual decode**: `q0_source_embedding + predicted residual_embedding` を hard token 化せず decoder に渡す。
3. **Adapter only**: `z_in -> Adapter -> frozen DAC decoder`。decoder 本体は凍結。
4. **Partial fine-tune**: adapter で足りない場合のみ decoder 後段を解凍。

Go条件:
- exact/noisy continuous latent decode >= 0.60
- generator latent decode >= 0.45
- CER <= 0.10
- UTMOS/DNSMOS が target_all から大きく劣化しない

撤退条件:
- 小さな latent noise で SECS/CER が急崩壊する
- adapter が round-trip 音質を壊す
- decoder fine-tune が catastrophic forgetting を起こす

### Phase 3c: Singing Mode

- F0 stream 強化（vibrato/rhythm 保持）
- speech/singing mode token 追加
- 声区転換・息・長母音対応

### Phase 4: Lightweight Content Encoder

Wav2Vec2依存を削除:
- 軽量content unit encoder（蒸留 or 新規学習）
- precomputed target timbre index
- Rust/Candle streaming推論

### Phase 5: Rust 推論

1. DAC encoder/decoder → Candle実装
2. Gθ → Candle実装
3. Streaming overlap-add
4. **Go条件**: RTF < 0.5, latency < 50ms

## 失敗条件と Fallback

### Phase 2 で cross-text retrieval が全滅（実測）

same-text (0.656) vs cross-text (0.33)。異テキスト間に単調pathが存在しない。
→ Phase 3（生成アプローチ）へ移行。retrieval は診断として保持。

### Phase 3 で cross-text generator < 0.40

→ 異テキスト enrollment からの target-like latent 生成自体が弱い。
same-text mode（ダビング/台本/定型文）に特化するか、
DAC decoder fine-tune を検討。

## CONCEPT 整合性

### 維持する要素

- ✅ codec-space one-step VC
- ✅ 軽量・低遅延
- ✅ Rust 推論
- ✅ VC 本体に重い波形生成をさせない
- ✅ MIT/Apache ライセンス

### 破棄する要素

- ❌ continuous latent regression
- ❌ frozen decoder + 任意連続点の decode
- ❌ WORLD mcep bank retrieval（診断としては有用だが本線ではない）

### 新規要素

- 🆕 RVQ depth 分離による content/timbre 分解（Phase 1b実証）
- 🆕 source q0 固定 + residual re-quantization（Phase 2a実証）
- 🆕 content-aware temporal alignment（Phase 2c実証、same-text有効）
- 🆕 target-like latent trajectory generator（Phase 3本線）
- 🆕 multi-objective anti-collapse training

## 関連ファイル

- `plan/10_analysis_resynthesis.md` — BigVGAN approach（破棄）
- `plan/11_source_filter_vc.md` — WORLD approach（研究停止、診断として参照）
- `training/converter.py` — DAC converter（v1 連続回帰、参照用）
- `training/dac_model.py` — DAC encoder/decoder
