# Y-S1 Causal Continuous Codec

status: PROPOSED rev2 整理版（2026-09-08。設計・ABIを保持し、過去の承認と未完了事項を反映）

女声再構成には耳PASS、Rust decoderにはS1-4完了の履歴がある。製品VCの採用ではない。
decoderのp95上限は2026-08-28のオーナー承認により≤3.5ms。生成可能性S1-LGの
第3条件は保留のため、総合PASSと扱わない。現在地・証拠の留保は [RESEARCH.md](RESEARCH.md)、
上流生成器は [cfm_ys1.md](cfm_ys1.md)。以下のラダーは契約であり、全工程の再実行指示ではない。

## 0. 役割と判定

Y-S1 は、48kHz の実音声を低遅延で連続 latent に圧縮・再構成する自作 codec である。
製品推論では decoder のみを使い、上流の CFM が生成した latent を波形へ変換する。

```text
学習: real wav -> causal encoder -> z @ 100fps -> causal decoder -> wav
製品: E/P/S -> CFM -> z @ 100fps -> causal decoder -> wav
```

Y-S0 で凍結 DACVAE decoder は 25fps、透明品質に約320msの未来文脈を必要とする
ことが確定した。従って DACVAE は製品 runtime に入れず、offline 品質アンカーとして
のみ残す。証拠は `results/ys0/`。

Y-S1 はまだ採用ではない。採用条件は以下を全て満たすこと。

1. 厳密因果・stateful streaming と full-sequence の一致
2. GT 対 causal reconstruction の人間の耳ゲート合格
3. Rust/Candle parity
4. codec decoder の計算予算と full-graph 静的台帳が p95 30ms 枠内

proxy、PESQ、mel-L1、RTF 単独では昇格しない。

## 1. 固定 ABI

| 項目 | rev1 |
|---|---|
| sample rate | 48,000 Hz |
| channels | mono |
| hop | **480 samples = 10ms** |
| latent rate | **100fps** |
| latent dimension | **32 continuous values/frame** |
| encoder timestamp | `z[t]` は `x[0:(t+1)*480]` のみを参照 |
| decoder emission | `z[t]` 受理後に `y[t*480:(t+1)*480]` を確定 |
| algorithmic future context | **0 samples** |
| runtime latent | deterministic encoder output `z`。VAE samplingなし |
| waveform scale | float32, nominal `[-1, 1]`。発話単位正規化なし |

hop 960（20ms）は計算量・品質の診断 arm に限る。フレーム待ちの p95 が約19msに
達し、E/P/G/codec/I/O の残予算を失うため、製品候補へ自動昇格させない。

latent は量子化しない。CFM は連続分布を直接生成でき、RVQ は fine texture と breath
を落とす既知リスクがある。学習後に full training split から channel-wise `mean/std`
を一括計算し、CFM ABI の固定 affine とする。発話統計、先頭N件、実行中の逐次統計を
使わない。

## 2. rev1 ネットワーク

### 2.1 Encoder

```text
wav [B,1,T]
 -> causal Conv1d k7, 1 -> 32
 -> stage r=8: ResUnit(d=1,3,9) -> causal strided Conv1d k=16, 32 -> 64
 -> stage r=5: ResUnit(d=1,3,9) -> causal strided Conv1d k=10, 64 -> 128
 -> stage r=4: ResUnit(d=1,3,9) -> causal strided Conv1d k=8, 128 -> 256
 -> stage r=3: ResUnit(d=1,3,9) -> causal strided Conv1d k=6, 256 -> 512
 -> causal Conv1d k3, 512 -> 32
 -> z [B,32,T/480]
```

各 ResUnit は `SnakeBeta -> Conv1d(k7,dilation=d,C->C/2) -> SnakeBeta ->
Conv1d(k1,C/2->C) + identity`。全 Conv1d は `groups=1`。padding は左側だけに置く。
通常の causal Conv1d は左pad `(k-1)*d`。stride `r`、kernel `2r` の downsample は
左padを **`k-r=r`** とし、出力frame `t` の最終参照点を入力 `(t+1)r-1` に固定する。
右padと末尾の未来補完は行わない。hop整列した長さ `F*480` から厳密にF frameを得る。

### 2.2 Decoder

```text
z [B,32,F]
 -> causal Conv1d k7, 32 -> 512
 -> stage r=3: streaming ConvTranspose1d k=6, 512 -> 256 -> ResUnit(d=1,3,9)
 -> stage r=4: streaming ConvTranspose1d k=8, 256 -> 128 -> ResUnit(d=1,3,9)
 -> stage r=5: streaming ConvTranspose1d k=10, 128 -> 64 -> ResUnit(d=1,3,9)
 -> stage r=8: streaming ConvTranspose1d k=16, 64 -> 32 -> ResUnit(d=1,3,9)
 -> SnakeBeta -> causal Conv1d k7, 32 -> 1 -> tanh
 -> wav [B,1,F*480]
```

ConvTranspose は対称 padding で切り出さない。各 latent frame が作る `2r` サンプルの
後半を stage state に保持し、次 frame の前半と overlap-add する。確定済みの先頭
`r` サンプルだけを下流へ渡す。この規則を4段とも適用し、未来 latent を待たない。
full-sequence 実装では末尾に生じる未確定tail `r` を各stageで捨て、streaming実装と
同じ長さ・同じstartup zero-stateにする。

### 2.3 実装制約

- standard Conv1d / ConvTranspose1d / elementwise activation / add のみ
- depthwise/grouped conv、reflection padding、発話全体 normalization を禁止
- anti-alias FIR を使う場合も左寄せ因果 FIR のみ。rev1 には入れない
- weight normalization は学習時のみ。export 前に除去する
- discriminator は学習専用なので非因果でよい
- 基準構成の上限は 35M parameters。parameter 数ではなく streaming MAC と実測を採点する
- PyTorch と Rust の key は §8 の名前に完全一致させる

初期c48構成（8.94M/4.04 GMAC/s）は p95 3.63ms・RTF 0.346で不合格。c40bも
50-runでp95 3.32msとなり不合格。ABIを変えず幅だけを下げた **c32** をrev1既定にした。
構造値は **4.59M parameters / decoder 1.83 GMAC/s**。

**rev2 訂正**: PyTorch eager の batch 計測（8 frame 処理時間÷8）は本 probe の
protocol 違反であり、c48/c40b の除外根拠として弱い。**c32 は暫定本線**とし、
**c40 を品質診断 arm** として S1-1/S1-2 で同一発話・同一 step で並走させる。
c32 が耳で同等と確認された時点で初めて幅を固定する。オーナー実測
（i5-12400F・1-frame call 2,000 回）: p50 2.362ms / p95 2.566ms / p99 2.756ms /
max 3.918ms / mean RTF 0.239。c32 は暫定的に hard gate 内。正式 probe は
§6 protocol で再取得する。

## 3. なぜ直接学習を本線にするか

DACVAE teacher latent は 25fps かつ約320msの未来情報を含む。これを100fpsの厳密因果
encoderへ frame L1 回帰すると、student が観測できない未来を平均化して予測することに
なり、muffle/transient loss を構造的に招く。frame rate も一致しない。

従って rev1 の教師は実音声そのものとする。

- **本線 S1-D**: real wav -> 自作 causal codec。再構成 loss + waveform GAN
- **offline anchor**: GT / DACVAE offline / Y-S1 を同じ発話で聴く
- **禁止**: DACVAE latent の補間後 L1、DACVAE decoder feature の無条件模倣、
  DACVAE再構成音を唯一の target にすること

DACVAE 蒸留は S1-D が耳ゲートを通らない場合の自動救済策にしない。S1-D が既に
transient と発音を保ち、bad tag が texture に局在した場合だけ、学習時限定の単独 arm
S1-T として検討する。その場合も real wav loss を主とし、teacher 項の有無だけを比較する。

これは VC teacher 蒸留ではないが、外部 codec の非因果性を製品へ持ち込まないための
境界である。

## 4. 学習レシピ

### 4.1 データ

- 主 GT は実音声のみ。女声targetを本対象とする。男声は診断・頑健性の別枠で、
  S1-3の男声再構成FAILを女声再構成PASSで隠さない。男性入力VCの成立は上流を含む別ゲート。
- TTS corpus は codec の主 GT と held-out 合否判定に使わない
- 48kHz mono。resample はキャッシュ生成時に1回だけ
- **左文脈付き crop（rev2）**: codec の境界影響は encoder 過去受容野 18,436
  samples ≈384ms ＋ decoder 影響持続 20,354 samples ≈424ms ≈ **計 808ms**。
  crop ごとの zero-state 学習は先頭 0.81s を人工状態にする（1.28s の 63%）ため
  禁止する。学習 crop は:
  - 実発話から **左文脈 ≥1.0s**（808ms + 余裕）を付け、
    **loss/discriminator の対象はその後続 61,440 samples = 128 frame = 1.28s のみ**
  - 左文脈は state 収束用で loss 掛けなし。発話の実際の先頭（左文脈が取れない
    場合）だけ zero-state を許し、その crop では先頭から loss を掛ける
  - hop 境界に揃える
- 発話単位 loudness normalization はしない
- gain augmentation `[-30, 0] dB`（rev2 固定仕様）: **適用確率 0.5/crop**、
  適用時は一様分布。無適用 50% を原音声レベルのanchorとして残す。
  `[-30,0]` 一様＋quiet 30% oversampling の重ね合わせは低レベルへ偏るため、
  quiet oversampling は候補プールからの抽出確率のみ（レベル操作は gain aug 側と
  二重にしない）
- quiet/breath/whisper 候補を30%以上サンプルする。通常発話だけで batch を埋めない
- train/held-out は話者と発話の両方を分離する。**held-out は無加工（gain aug なし）
  の arm も同時に出す**

### 4.2 Phase R: 再構成の立ち上げ

Generator loss の初期値:

```text
L_R = 15 * L_logmel + 2 * L_mrstft + 1 * L_wave_l1
```

**rev2 で固定する loss 仕様（再現可能とする）**:

- `L_logmel`（multi-resolution log-mel L1）:
  - FFT/window/hop: `(512,512,160), (1024,1024,256), (2048,2048,480)`
  - 48k・mel bins **FFT 別 `(64,96,128)`**（rev2a: 128×n_fft512 は zero
    filterbank を生じる実測バグ）・`fmin=0, fmax=24000`・`norm=slaney,
    mel_scale=slaney`
  - `log(clamp(mag, min=1e-5))`、reduction=mean、発話統計正規化なし
- `L_mrstft`（spectral convergence + log magnitude L1）:
  - FFT `(512,1024,2048)`・window=hann 同長・hop `(160,256,480)`
  - SC=`‖S_y−S_ŷ‖_F/‖S_y‖_F`、logmag=`|log(clamp(S,1e-5))差|` の mean、両者の和
- `L_wave_l1`: loss 区間（左文脈を除く 1.28s）の waveform L1 のみ
- 上記すべて loss 区間のみに適用。左文脈は state 収束専用
- KL、RVQ、latent L1、crest/sharpness/env-stab 等の手作り品質 loss は入れない
- 1発話 overfit で sample alignment、振幅、勾配、stateful decode を先に検査する

### 4.3 Phase G: texture

Phase R の信号経路が健全と確認できた後だけ GAN を追加する。

```text
L_G = L_R + 1 * L_adv + 2 * L_feature_matching
```

discriminator は既存の MPD + MRD + MS-SB-CQT を48kHz化して使用する。
**rev2 固定値**: MPD periods `(2,3,5,7,11)`（48k で低域十分）。MRD は
`Conv1d(k=3..15 odd, stride=2, MF=8)` の系列を 48k で 6 段（k=3,5,7,9,11,13）。
MS-SB-CQT は 48k・bands 6・hop 480 相当。CQT は training-only。
discriminator 入力は **loss 区間のみ**（左文脈を与えない）。
判別器追加と generator architecture 変更を同じ run で行わない。

- AdamW、generator `lr=2e-4`, discriminator `lr=2e-4`, betas `(0.8, 0.99)` を起点
- fp32 基準を先に通し、その後 AMP。XPU backward は groups=1 のみ
- EMA `0.999`。評価は raw/EMA の両方を同じ固定発話で出す
- discriminator が飽和した場合も手作り loss を足さず、更新比・lr・disc構成を診断する
- GAN は codec texture の学習であり、失敗した content/CFM を救済する用途にしない

loss weight は smoke 後に固定し、full run 中に耳ラベルを見て連続調整しない。変更は
1 run 1仮説で別 tag にする。

## 5. Streaming contract

state は layer ごとに明示する。

- causal Conv1d: `(kernel-1)*dilation` 個の過去 activation
- strided Conv1d: 過去 activation + stride phase
- ConvTranspose1d: 未確定 overlap tail
- SnakeBeta/tanh/add: stateなし

`decode_step(z_t)` は必ず480サンプルを返す。startup はゼロ state、flush は製品の
連続運転では使わない。offline evaluation の末尾だけ state をゼロ入力で flush し、
その tail を品質採点から除外せず別欄に記録する。

必須テスト:

1. future mutation: frame `t+1...` を変更しても、frame `t` までの出力が完全一致
2. full vs step: 同一 z の full decode と1-frame step decode の最大誤差 `<=1e-5`
3. arbitrary chunk: `{1,2,3,5,8,17}` frames/chunk が同一出力
4. reset: state reset 後に別発話の履歴が漏れない
5. length: F framesから厳密に `F*480` samples
6. PyTorch/Rust parity: fp32 max abs `<=1e-4`, SNR `>=80dB`

## 6. レイテンシ・RTF予算

codec の algorithmic lookahead は0。10ms frame の待ちは上流の frame scheduling と
共有し、二重計上しない。静的な full-graph 内部目標は27msとし、30msまで3ms残す。

| 項目 | p95 上限 | 備考 |
|---|---:|---|
| input block + frame aggregation | 10.0ms | hop480。I/Oと重なる分は実測JSONで説明 |
| causal E + prosody | 4.0ms | 将来の実測枠 |
| 1-step CFM G | 6.0ms | 将来の実測枠 |
| **Y-S1 decoder** | **3.5ms** | 2026-08-28承認。1 frame/480 samples、state update込み |
| output queue/copy | 1.0ms | host I/O自体との二重計上禁止 |
| jitter reserve | 2.5ms | decoderへ0.5ms充当、合計27ms |

Y-S1 decoder の単体目標:

- Core i5-12400F、1 thread、release相当: 10ms step の p95 `≤3.5ms`
- 平均 RTF `<0.20` を理想、`<0.30` を hard gate
- **probe protocol（rev2 修正）**: `decode_step` を **1 frame/step で呼び**、
  warm-up 後 **10,000 step** の p50/p95/p99/max・mean RTF を保存。
  batch 処理時間の均し割りは禁止（8 frame 処理÷8 は p95 of average であり
  protocol 違反）。JSON に `runs, warmup_steps, state_bytes, working_set_bytes,
  weight_bytes, cpu_name, thread_count, schema` を必ず含める
- XPU は小チャンク kernel launch が律速になり得るためCPUと別に測る。IPEX CPUは使わない
- parameter数、MAC/s、activation working set、weight bytesを0-GPU probeで先に保存

単体 gate を通っても製品合格ではない。最終判定は実オーディオ callback を含む
full graph p95 `<30ms`。

## 7. ラダーと停止条件

| gate | 内容 | 合格条件 |
|---|---|---|
| **S1-0** | shape/MAC/state probe | hop480、未来不変、full/step parity、decoder予算見込み |
| **S1-1** | 1発話 overfit（**c32 本線＋c40 診断 arm 同一条件**） | alignment/振幅正常。耳でclick・buzz・muffleなし |
| **S1-LG** | **latent 生成可能性 gate（rev2 新設）** | 下記3條件すべて |
| **S1-2** | 5–10話者、3–5h Phase R | held発話で破綻なし。breath/quietを保持 |
| **S1-3** | full real corpus Phase G | 女声GT対causal decodeの人間の耳ゲート合格。確認カテゴリを明記 |
| **S1-4** | Rust/Candle | parity、causality、10ms step p95 `≤3.5ms` |
| **S1-5** | CFM接続前 gate | latent global ABI固定、1-sample CFM overfit、静的27ms台帳 |

**S1-0 PyTorch subgate: 暫定 PASS（2026-08-27 rev2 格下げ）**。
`training/test_causal_codec.py` は 6/6 PASS。full/step max abs 2.68e-7 は有効。
一方 `results/ys1/s1_0_probe.json` の p95 2.51ms は **8-frame batch を8で割った
protocol 違反の値**であり、正式な frame p95 ではない。オーナー実測
（i5-12400F・1-frame call 2,000 回）: **p50 2.362 / p95 2.566 / p99 2.756 /
max 3.918ms / mean RTF 0.239** ＝ c32 は暫定的に hard gate 内。正式値は
§6 protocol（warm-up 後 10,000 step・1 frame/step・完全な JSON）で再取得し、
Rust gate（S1-4）で確定する。

**S1-LG: latent 生成可能性 gate（設計上はS1-2前。履歴では後置され、第3条件が保留）**。
「再構成可能」と「CFM が生成可能」は別物である（無量子化・無正規化の
連続 latent は再構成のために位相/微細 noise を高エントロピー成分へ逃がせる。
oracle latent が良くても生成 latent が off-manifold になり得る）。合格条件:
1. **latent 監査**: channel-wise scale/分布・channel 間相関・時間差分の
   統計・latent スペクトルの帯域偏りを report 化。異常（特定 channel の
   発散・相関の塊・超高周波 dominance）がなく
2. **摂動耐性**: latent に SNR 40dB 程度の微小摂動を加えても decode 波形が
   聴取可能範囲を維持する（劣化が線形であること）
3. **tiny CFM**: 5–10 発話で学習した小 CFM が学習発話の latent を再現でき、
   かつ**短い held 区間の latent も条件から再現できる**（off-manifold でない）
最初から KL/RVQ を入れる必要はない。本 gate で反証された場合のみ
正則化（KL/VQ）を別 arm で検討する。

停止条件:

- S1-0 で p95予算の見込みがない構成は学習しない
- S1-1 の固定1発話がfitしない場合、データやGANを足さず信号経路を直す
- **S1-LGの未完了を解消せず、本番CFMの追加学習へ進まない**。
  過去のS1-5実施は未完了を解消した証拠ではない。固定発話の診断で実装・生成可能性を再確認する。
- S1-3 の causal reconstruction が耳で落ちたら CFMへ進まない
- proxy 改善だけで full run を延長しない
- S1-D failure を DACVAE teacher やGAN強化で無条件に救済しない。
  研究側が比較を設計して劣化を局在化する。人間のbad tag提出を前提にしない。
- **c32 と c40 の耳比較で c32 が劣る場合、幅固定を白紙に戻す（rev2）**

## 8. Weight ABI

実装開始時から次の prefix を固定する。

```text
encoder.pre.{weight,bias}
encoder.stages.{i}.res.{j}.conv1.{weight,bias}
encoder.stages.{i}.res.{j}.conv2.{weight,bias}
encoder.stages.{i}.res.{j}.act1.{alpha,beta}
encoder.stages.{i}.res.{j}.act2.{alpha,beta}
encoder.stages.{i}.down.{weight,bias}
encoder.out.{weight,bias}
decoder.pre.{weight,bias}
decoder.stages.{i}.up.{weight,bias}
decoder.stages.{i}.res.{j}.conv1.{weight,bias}
decoder.stages.{i}.res.{j}.conv2.{weight,bias}
decoder.stages.{i}.res.{j}.act1.{alpha,beta}
decoder.stages.{i}.res.{j}.act2.{alpha,beta}
decoder.post_act.{alpha,beta}
decoder.post.{weight,bias}
```

`i` は実行順で0始まり、`j` は dilation `{1,3,9}` の順。weight normalization の
`weight_g/weight_v` は export ABI に残さない。

**ABI 注意（rev2）**: SnakeBeta の `alpha/beta` は実装上 **log-domain の値**で
保存される。Rust 側では `exp(alpha)` を適用してから使用する。この件は
safetensors metadata の schema 注記に明記する。

safetensors metadata に `sample_rate`, `hop_length`, `latent_dim`, `strides`,
`channels`, `causal=true`, `snake_log_domain=true`, schema versionを入れる。

## 9. 実装順

1. PyTorch model と stateful reference、causality/parity test
2. 0-GPU parameter/MAC/state/RTF probe
3. S1-1 overfit trainer と固定AB renderer
4. S1-2/3 data loader・GAN trainer
5. export と Rust decoder、同じ parity fixture
6. S1-3/S1-4 合格後にだけ Y-S1 latent 用 CFM を実装

Y-1 offline研究armと `data/latent48/` は廃止済みであり、復元・再利用しない。
Y-S1 cache は別 root と schema を使い、実音声と split manifest から再生成する。
