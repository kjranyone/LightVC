# Astrapeを超えるSOTA寄りのLightVCアイデア

求めるなら、**Astrapeの「連続latent + CFM + causal encoder/decoder」を超える方向は、codec空間の1-step変換 + factorized token制御 + teacher distillation** だと思う。

Astrapeは思想として良いけど、まだ「F³ encoder/decoderを自前で持ち、CFMを4〜8 stepで回す」発想に見える。ここを超えるなら、**VC専用の大きい生成パイプラインを作らず、既存codec表現を“変換可能な中間言語”として使う**ほうがSOTA寄り。

## 本命案：X-VC + SynthVC + MeanVC2 の合体

設計はこれ。

```text
Mic input
→ pretrained neural codec encoder
→ source codec latent / token
→ one-step codec-space converter
   - source latent
   - target frame-level acoustic condition
   - target utterance-level speaker embedding
   - prosody/rhythm control token
→ pretrained codec decoder
→ output waveform
```

ポイントは、**波形生成をVCモデルにやらせない**こと。  
VCモデルは codec latent を変換するだけにする。

2026年4月の **X-VC** はまさにこの方向で、pretrained neural codec の latent space で **one-step conversion** を行い、source codec latents と target reference speech 由来の frame-level acoustic conditions を jointly model し、utterance-level speaker information を adaptive normalization で注入します。さらに generated paired data と role-assignment strategy で学習時・推論時のミスマッチを減らしています。

これが Astrape より強いのは、**CFM ODEを複数stepで解くより、codec latentを1回で変換する設計に寄せられる**ところ。

## SOTAネタ1：codec-space one-step converter

Astrape超えのコアはこれ。

```text
x_codec: source codec latent
c_tgt_frame: target reference から得た局所的音響条件
s_tgt: target speaker embedding

y_codec = Converter(x_codec, c_tgt_frame, s_tgt)
```

モデルは大きいDiTではなく、

```text
causal Conformer-lite
or
Mamba/SSM-lite
or
depthwise Conv + gated attention
```

でよい。

「continuous latent + flowで生成」ではなく、**codec latentを条件付き写像する**。  
これなら decoder は既存codecに任せられるので、VC本体の負担が減る。

X-VCが示唆しているのは、**VCを waveform generation ではなく codec-space translation と見なす**こと。ここがかなり重要。

## SOTAネタ2：synthetic parallel distillation

次に効くのはこれ。

**SynthVC** は、pre-trained zero-shot VC model が生成した synthetic parallel data を使って、streaming end-to-end VCを学習します。明示的な content-speaker separation や recognition module を不要にし、neural audio codec architecture上で低遅延streaming推論を行い、論文では end-to-end latency 77.1ms とされています。

これはLightVCにめちゃくちゃ向いてる。

つまり学習時は、

```text
source speech
↓ teacher VC
teacher converted speech
```

を大量生成して、

```text
source codec latent → target codec latent
```

を student に覚えさせる。

このとき teacher は重くていい。RVCでもSeed-VCでもMeanVCでもいい。  
本番は student だけ。

LightVCの実装としては、

```text
Teacher: 高品質zero-shot VC
Student: codec-space one-step converter
Loss:
- codec latent L1 / L2
- multi-scale STFT
- speaker similarity
- content consistency
- adversarial optional
```

で進めるのが現実的。

## SOTAネタ3：MeanVC2の bounded future context

完全causalを神格化しすぎると音が悪くなる。  
ここは **MeanVC2** の考え方が強い。

MeanVC2は、MeanVCの弱点として「小さいchunkで品質が落ちる」「chunk-wise AR denoisingで学習系列長が実質2倍になる」「reference mel品質に敏感」を挙げ、**future-receptive chunking** と **universal timbre token encoder** を導入しています。40ms chunkで安定変換し、latencyを211msから110msへ下げたと報告されています。

LightVCに落とすなら、

```text
strict causal mode: 0ms lookahead
low latency mode: 40〜80ms lookahead
quality mode: 120ms lookahead
```

を切り替える。

Astrapeが「fully causal」を強みにしているなら、超えるには **“完全因果だけでなく、bounded lookaheadを設計変数にする”** ほうが現実的。配信・通話では 80〜120ms くらいなら許容される場面が多いし、品質差が大きい。

## SOTAネタ4：universal timbre token encoder

Astrapeの弱点になりそうなのは、target speaker conditioning の設計です。  
単純な speaker embedding だけだと、声質の細部や録音品質に左右される。

MeanVC2は、global speaker embedding と cross-attentionによる fine-grained timbre cue retrieval を組み合わせる **universal timbre token encoder** を入れています。これにより低品質referenceへのロバスト性とzero-shot speaker similarityを改善する狙いです。

LightVCではこうする。

```text
target reference audio
→ speaker embedding
→ timbre token bank
→ cross-attentionで現在chunkに必要な声質cueを取得
```

UIにも落とせる。

```text
Timbre strength
Breathiness
Brightness
Nasality
Age/weight感
Source leakage suppression
```

全部を明示ラベルで学習しなくても、token bank + adapterで制御量を作れる。

## SOTAネタ5：prosody/rhythmを別トラックにする

単に「声色だけ変える」と、元話者の癖が漏れる。  
ここを超えるには、**content / timbre / prosody / rhythm をfactorize** したほうが良い。

**Discl-VC** は、SSL表現から content と prosody の discrete token を分離し、flow matching transformer + in-context learningでzero-shot VCを行う設計です。特に prosody token を non-autoregressive に予測して、韻律制御を強めています。

**R-VC** は rhythm-controllable zero-shot VCで、Mask Generative Transformerによる duration modeling と shortcut flow matching を使い、target speakerの rhythm/style transfer を狙います。shortcut flow matchingにより2 stepでも高いtimbre similarityと品質を狙う設計です。

LightVCに入れるなら、

```text
content stream
timbre stream
prosody stream
rhythm/duration stream
```

を分ける。

ただしRVCみたいにHuBERTやF0を必須にするのではなく、**codec latentから軽量tokenizerで分ける**。

最終的にはこう。

```text
y_codec = Converter(
  source_content_codec,
  target_timbre_tokens,
  prosody_mode,
  rhythm_mode
)
```

prosody_mode はUIで：

```text
Preserve source prosody
Blend
Imitate target prosody
Flatten for privacy
```

みたいにできる。

これは「単に似せるVC」よりプロダクト価値が高い。

## SOTAネタ6：discrete flow matching / factorized heads

VCそのものではないけど、TTS側のネタも使える。

**DiFlow-TTS** は discrete codec representations を連続空間に埋め込んでflowするのではなく、**purely Discrete Flow Matching** を探索し、prosody と acoustic detail に分けた factorized flow prediction heads を使います。低遅延で、既存baselinesより最大25.8倍高速生成と報告されています。

VCに持ち込むなら、

```text
codec tokenを連続latentに戻して回帰する
```

だけでなく、

```text
RVQ codebook depthごとに予測する
coarse token: content/timbre
fine token: texture/detail
```

にする。

これは 2026年の streaming TTS でも似た考え方が出ていて、Mimi codec の32層RVQ codeを progressive depth-wise にdecodeすることで、time-to-first-byte 48.99msを狙う研究があります。

LightVCなら、

```text
coarse codec layers: 低遅延で即出す
fine codec layers: 後続chunkで補正
```

が面白い。

つまり **progressive codec-depth conversion**。

```text
t = now:
  RVQ layer 1-4 を即時変換
t + small delay:
  RVQ layer 5-12 を補完
```

これ、低遅延と音質の両立にかなり効く可能性がある。

# Astrapeを超えるLightVC案

名前をつけるなら、

## LightVC-X: Codec-Space One-Step Streaming VC

```text
Input waveform 44.1kHz/24kHz
↓
Pretrained codec encoder
↓
Codec latent/token
↓
Factorized streaming converter
  - content path
  - timbre token cross-attention
  - prosody/rhythm controller
  - AdaLN/FiLM speaker injection
↓
Progressive codec-depth decoder
↓
Output waveform
```

## モデル構成

```text
Codec:
- Mimi / EnCodec / DAC / SoundStream系
- encoder/decoderは基本frozen

Converter:
- 5M〜30M parameters
- causal Conv + SSM/Mamba-lite + local attention
- one-step latent/token conversion

Conditioner:
- universal timbre token encoder
- target reference cache
- prosody/rhythm token predictor

Training:
- synthetic parallel distillation
- role assignment: standard / reconstruction / reversed
- codec latent loss
- STFT loss
- speaker similarity loss
- content preservation loss
```

X-VCの generated paired data と role-assignment strategy は、LightVCでもかなり使える。

## 推論

```text
Frame: 20ms
Chunk: 40ms
Lookahead:
- 0ms strict
- 40ms balanced
- 80ms quality

Converter:
- one forward per chunk
- no ODE loop
- no retrieval
- no external F0 extractor
```

ここがAstrape超えポイント。

```text
Astrape:
  continuous latent + CFM 4〜8 step + full custom encoder/decoder

LightVC-X:
  pretrained codec + one-step codec-space converter + factorized token control
```

## さらに攻めるなら：dual-path converter

声質変換は全部同じモデルでやらず、2パスに分ける。

```text
Fast path:
  low-latency coarse conversion
  content/timbreの大枠だけ

Refine path:
  bounded lookaheadでtexture補正
  breath/noise/high-frequency/detail
```

出力は常にfast pathで出しつつ、refine pathは次chunkに混ぜる。  
これはストリーミングTTSの block-wise / depth-wise codec decoding の考え方に近い。

## 研究としての一番強い新規性

論文ネタにするなら、この組み合わせが強い。

> **Progressive RVQ-depth voice conversion in codec space with synthetic teacher distillation and universal timbre token conditioning.**

日本語にすると、

> codecの時間方向だけでなく、RVQ depth方向にも段階的に変換するリアルタイムVC。

これ、Astrapeの「時間chunkをcausalに処理する」より一段上の設計です。

普通のstreaming VCは時間方向のlatencyばかり見ます。  
でも neural codec は RVQ depth を持っているので、

```text
時間方向 latency
×
codebook depth方向 fidelity
```

の2軸で設計できる。

つまり、

```text
低遅延モード:
  coarse codeのみ変換

高品質モード:
  fine codeまで変換

privacy mode:
  timbre-bearing layersを強く変換

natural mode:
  lower layers preserve, upper layers convert
```

みたいな制御ができる。

これはアプリとしても研究としても強い。

# 実装優先順位

まずはこれ。

## Phase 1: codec latent distillation

```text
pretrained codecを選ぶ
source/teacher converted pairを作る
codec latent同士をone-step converterで学習
```

ここで VC として鳴るか確認。

## Phase 2: target timbre token

```text
target referenceを複数chunkに分解
speaker embedding + local timbre tokensを作る
cross-attentionでconverterに入れる
```

## Phase 3: factorized RVQ depth

```text
coarse/fine codeを分ける
coarseは低遅延
fineは品質補正
```

## Phase 4: prosody/rhythm control

```text
source preserve
target imitate
blend
privacy flatten
```

Discl-VC / R-VC系のネタをここで入れる。

# 結論

Astrapeを超えるなら、**CFMで波形/latentを生成するVC**ではなく、**codec-space translation engine**にするのが本命。

一言で言うと：

> **RVCでもAstrapeでもなく、codec tokenをリアルタイムに翻訳するVC。**

最強案はこれ。

```text
Pretrained codec
+ one-step codec-space converter
+ synthetic parallel distillation      ← 削除 (2026-06 改訂)
+ universal timbre token encoder
+ progressive RVQ-depth conversion
+ bounded future context
```

これなら、Astrapeの思想を継承しつつ、より軽く、よりSOTAっぽく、プロダクトにも落としやすいです。

---

## 設計改訂メモ (2026-06)

**synthetic parallel distillation を削除。** 理由：

1. **Seed-VC（想定teacher）自体がteacherなしで学習されている** — "timbre shifter"は信号処理拡張であり、neural teacherではない。SOTAゼロショットVC 16モデル中14がteacher-free。
2. **蒸留は品質要件ではなく速度圧縮** — multi-step拡張を1-stepに圧縮するだけ。Mean-flow / shortcut flow matchingなら最初から1-stepで学習可能。
3. **Teacherの品質が上限になる** — Astrape超えのCONCEPTと矛盾。
4. **ライセンス汚染リスク** — Seed-VCはGPL-3.0かつアーカイブ済み。
5. **Phase Aが数日→数時間に短縮** — teacherでfake pair生成する代わりに、実音声をDACエンコードするだけ。

**代替：mean-flow / shortcut flow matching の直接学習（Paradigm 6）+ bottleneck warm-start（Paradigm 2）**

```text
Pretrained codec (DAC)
+ one-step mean-flow converter (1-NFE, no ODE loop)
+ bottleneck autoencoder warm-start (speaker disentanglement)
+ timbre-shifter augmentation (signal processing, NOT a teacher)
+ universal timbre token encoder
+ progressive RVQ-depth factorized FM heads  ← 新規性の核心
+ bounded future context
```

target latent = 実際のtarget話者のDAC latent（real recording）。teacher不在。
