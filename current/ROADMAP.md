# LightVC 鳥瞰ロードマップ（データ × モデル × 評価 × 製品）

> status: **生きた計画**。最終更新 2026-07-19。
> 反省起点: E0–E2 を深掘りする間、全体計画が無く「低レベル原則」「whisper 欠落」「augmentation 不在」を全て事後発見した。本書はその再発防止＝各段の依存物（データ・指標・判定者）を着手前に確定させる。

## 0. 製品要件（不変、README より）

ASMR・官能バ美肉向けリアルタイム VC。E2E <50ms / Rust(Candle) / MIT / zero-shot / **息・囁き・小声＝商品**（低レベル音量の一級市民化, vocoder.md §3.5）。

## 1. システム分解と現在地

```text
[content encoder(因果蒸留)] → [retrieval/timbre] → [prosody policy] → [★vocoder] → [runtime]
      未着手(G-enc)             部分資産(M2)          部分資産(render_m2)  耳ゲート通過+Rust移植DONE  app配線DONE(offline)/streaming path A実装中
```

**現在地（2026-07-19）**: vocoder = **耳ゲート通過**（freebig ≈ BigVGAN, 2026-07-16/17, **ADOPTED**）＋ **Rust/Candle 移植 DONE**（freebig 非causal SNR 106.6dB / freeC causal SNR 88.5dB, `candle_vocoder_port.md` §0.5）＋ **リアルタイム達成**（チャンク streaming K=4 で RTF 0.94, +8.7ms）。config **C = causal 5.8ms**。
**★2026-07-19 訂正（app 配線 streaming 検証）**: 「5.8ms」は**合成窓側のみ**。freeC 訓練 mel は centered（BigVGAN mel_spectrogram、win/2 先読み内包）→ 真causal 左寄せ mel では streaming が SNR 1.26dB に崩壊（**mel 起因・vocoder 無罪**）。matching 品質 streaming の**真遅延 ≒ 合成 5.8 + mel 解析 ~23 + buffer ≒ 30ms**（<50ms・Beatrice/paravo 級）。**path A（centered streaming mel）採用**（実装中）・**path B（causal-mel 再訓練で ~5.8ms）は将来**（`candle_vocoder_port.md` §0.5 / `vocoder.md` §3.7）。
判明した結論: 従来の品質ギャップの主因は **"未訓練/データ被覆"** だった（arch・損失・位相・source-filter のいずれでもない）。手作り DSP／source-filter／位相ヘッド系（§4 反証済み系譜, `RESEARCH` の A/S 天井）は **negative result** として整合し、波形ニューラルボコーダ（freebig）が勝った。

**クリティカルパス**: 「vocoder が耳ゲートを通らない限り上流に投資しない」制約は **解除**（通過済）。
→ **次フェーズ = 上流**: content encoder（G-enc 因果蒸留）＋ **多参照 Factor コンポーザ前段**（`current/zeroshot_vc.md` の Z0→Z6、うち **Z4 disentanglement gate を最優先**）＋ vocoder の **app 配線**（`lightvc-app` inference_loop → Candle `FreeVocoder`, K=4〜8）＝**offline は DONE（62dB, commit 82462a5）、streaming は path A（centered mel）実装中**。
残る vocoder 側の課題は (i) **streaming path A（centered streaming mel, 真遅延 ~30ms）実装**、(ii) K=2/256 サンプル厳密低レイテンシ（単スレ未達 → マルチスレ/高速 GEMM、`candle_vocoder_port.md` §0.5）。

## 2. データ戦略の一枚絵

### 2.1 コーパス役割マトリクス

| コーパス | vocoder recon | timbre/style 条件 | content/prosody | 評価 | 備考 |
|---|---|---|---|---|---|
| female-dataset（実、2775話者・60h+級） | **主力** | 主力 | — | 主力 | スタイルラベル無し→**proxy マイニング必須** |
| Irodori TTS v3（合成） | 補助のみ | caption 条件の面展開 | ✗ | ✗（GT にしない） | §2.2 |
| male/VCTK | 低F0 robustness 補助 | — | source側主力 | same-text 検証 | |
| golden mini set | — | — | — | 固定AB | **whisper カテゴリ欠落（要キュレーション）** |

### 2.2 Irodori TTS の位置づけ（再確認と限界）

README 既定「主データにしない」は維持。追加で明確化:
- **TTS 音声を vocoder の GT/評価に使うと「TTS の合成癖」を正解として学習・採点する**（RESEARCH の過去教訓「評価 GT が TTS でないか確認」）。E2 以降の recon GT は実音声のみ。
- TTS の正しい用途は caption_key→style 条件の**ラベル付き面展開**（実データにはラベルが無い）に限る。
- TTS には自然な息継ぎ・マイク近接感・周期ゆらぎが乏しい＝低レベル一級市民化と正面衝突。**TTS 比率はスタイル条件学習時でも従属的に**。

### 2.3 Augmentation 設計（従来欠落していた章）

原則: **target 側（vocoder が描く声）は汚さない。入力・条件側と、不変性を教えるための摂動のみ。**

| Augmentation | 適用先 | 目的 | 段階 |
|---|---|---|---|
| **ランダムゲイン −30〜0 dB**（セグメント単位） | vocoder 学習入力+GT 両方に同係数 | レベル不変性＝低レベル一級市民化の実装。loss の逆ラウドネス重みと対 | **E2 Phase B から即** |
| 静区間オーバーサンプリング（proxy: RMS/CPP/HNR） | サンプラ | 息・小声の学習密度確保 | E2 多話者化から |
| F0/formant/SR 摂動 | content encoder 蒸留の入力 | speaker 情報を落とす（README 既定） | G-enc |
| 軽リバーブ/マイク EQ/背景ノイズ | content encoder・robustness 系のみ | 実運用入力の頑健性。**vocoder GT には適用禁止** | M3 前 |
| ピッチシフト data（srcshift 系） | cross-gender 学習ペア | 男→萌えの F0 レジスタ | E5/G-cross |
| noise seed/jitter 再サンプル | vocoder 学習（実装済: 毎 step 新 noise） | 確率成分の分布学習 | 済 |

**やらないこと**: target 音声への SpecAugment/time-mask 系（P1 回帰 muffle と同種の平滑化圧力を生む）、TTS で実音声を「増やした事にする」水増し。

### 2.4 量とスタイル分布の目標

| 段階 | 量 | スタイル要件 |
|---|---|---|
| E2（現在） | 単一話者 14分 | 不問（ゲート目的） |
| E2 多話者化 | 5–10話者 × 計 3–5h | **quiet（RMS 下位3割）比率 ≥ 30%**、breathy/囁き proxy 混入確認 |
| Core（10–50h） | README ラダー準拠 | 必須発話型リスト（README）+ whisper 実データのキュレーション完了が前提 |

## 3. 段階ラダー（各段の「着手前に揃える物」を明記）

| 段 | 内容 | 事前に揃える物（データ/指標/判定） | 状態 |
|---|---|---|---|
| E0 | オラクル物理 | golden set / gate v1–v3 / 耳 | **済**（v1.5） |
| E1 | overfit 学習性 | floor 較正 | **済** |
| E2-A | 単一話者 recon (no-GAN) | oracle キャッシュ / E0 バッテリー | **済**（freebig 耳ゲート通過, ADOPTED） |
| E2-B | +GAN texture | MPD/MRD 流用確認 / **ゲイン augmentation** / 耳AB(gt vs net vs nsf3_gan) | **済**（freebig ≈ BigVGAN で耳ゲート合格） |
| E2-C | 多話者+timbre 条件 | **quiet マイニング** / 話者選定 / timbre encoder 結線 | 次（上流フェーズと並走） |
| G-enc | content encoder 蒸留 | HuBERT teacher / 摂動 augmentation / leakage 指標 | 次（Z4 gate 最優先） |
| E5 | B4 結線 (VC) | CIPT / srcshift ペア / G-cross ゲート | 未 |
| E4/M3 | realtime/Rust | causality CI / パリティ / RTF | **Rust 移植 DONE**（parity SNR 88–106dB / K=4 RTF 0.94）・app 配線 offline DONE(62dB)・**streaming path A(centered mel, 真遅延~30ms) 実装中** |

**判定者の分担（固定）**: 機械=アーティファクト検出（gate 指標、耳ラベルで較正済）／人間=官能評価のみ（Smoothness/Tenderness/…、README の HITL）。診断 AB を人間に投げない。

## 4. 既知の穴（優先順）

1. **whisper/breathy の実データ評価セットが無い**（golden 未収載）— proxy マイニング+人間確認で解消。E2-C 前必須。
2. **ゲイン augmentation 未実装** — E2-B と同時（数行）。
3. nsf3_gan との同一データ AB 未実施（content 特徴の抽出待ち）。
4. 枝B（BigVGAN 蒸留）の並走未着手 — E2-B が耳で崩れた場合の保険なので、E2-B 失敗時に起動。
5. ~~Candle 側パリティ（E4）は torch 実装凍結後。~~ **解消**（freebig/freeC 移植 DONE, parity SNR 88–106dB, `candle_vocoder_port.md` §0.5）。app 配線 offline DONE(62dB)。残 = **streaming path A（centered mel, 真遅延~30ms）実装**（freeC 訓練 mel が centered ゆえ左寄せ mel は SNR 1.26dB 崩壊＝mel 起因・vocoder 無罪）と K=2 厳密低レイテンシ最適化。
