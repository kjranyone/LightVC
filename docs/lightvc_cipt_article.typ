#import "@preview/fletcher:0.5.5" as fletcher: diagram, node, edge

#set document(title: "LightVC — CIPT: 出力側同一性監督による話者天井の突破")
#set page(paper: "a4", margin: (x: 2.0cm, y: 2.0cm), numbering: "1 / 1")
#set text(font: ("Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"), size: 10pt, lang: "ja")
#set par(justify: true, leading: 0.7em)
#show heading.where(level: 1): it => block(above: 1.2em, below: 0.6em)[
  #set text(size: 14pt, weight: "bold", fill: rgb("#1a4d6d")); #it.body]
#show heading.where(level: 2): it => block(above: 0.9em, below: 0.45em)[
  #set text(size: 11pt, weight: "bold", fill: rgb("#2a6f8f")); #it.body]

#let accent = rgb("#1a4d6d")
#let box2(body, fill: rgb("#eef4f8")) = box(fill: fill, inset: 7pt, radius: 5pt, stroke: 0.6pt + accent, width: 100%)[#body]
#let cin = rgb("#fbe4c8")
#let cmod = rgb("#dbe9f4")
#let cgen = rgb("#d7ecd7")
#let cout = rgb("#f6d5d5")
#let csrc = rgb("#f4e8f7")
#let cdis = rgb("#e7dcf3")
#let closs = rgb("#f6d5d5")
#let caux = rgb("#eee")
#let figcap(n, t) = align(center)[#text(size: 8.5pt, fill: rgb("#555"))[*図 #n.* #t]]
#let hd(c) = text(fill: white, weight: "bold")[#c]
#let thead(row) = if row == 0 { rgb("#1a4d6d") } else if calc.rem(row, 2) == 0 { rgb("#f2f6f9") }

// ===== 図: 推論システム =====
#let dia_infer = diagram(
  node-stroke: 0.7pt + rgb("#334"), node-corner-radius: 4pt, spacing: (9mm, 6.5mm),
  node((0,1), [Source 波形\ (男性)], fill: cin, name: <src>),
  node((0,3.5), [Target 参照\ (女性,数秒)], fill: cin, name: <tgt>),
  node((2,0), [ContentVec\ #text(7pt)[768ch]], fill: cmod, name: <cv>),
  node((2,1), [F0 抽出\ + srcshift], fill: cmod, name: <f0>),
  node((2,2), [Energy], fill: cmod, name: <eng>),
  node((2,3.5), [TimbreEncoder], fill: cmod, name: <tb>),
  node((4,0), [ContentScrub\ #text(7pt)[GRL]], fill: cmod, name: <sc>),
  node((5.3,1), [⊕], fill: white, shape: fletcher.shapes.circle, name: <cat>),
  node((7,1.6), align(center)[*NSF-HiFiGAN*\ *Generator*], fill: cgen, width: 26mm, height: 15mm, name: <g>),
  node((9.3,1.6), [変換波形\ 44.1kHz], fill: cout, name: <out>),
  edge(<src>, <cv>, "-|>"), edge(<src>, <f0>, "-|>"), edge(<src>, <eng>, "-|>"),
  edge(<tgt>, <tb>, "-|>"), edge(<cv>, <sc>, "-|>"),
  edge(<sc>, <cat>, "-|>", [$c'$]), edge(<f0>, <cat>, "-|>", [logF0]), edge(<eng>, <cat>, "-|>", [logE]),
  edge(<cat>, <g>, "-|>", [cond 770]),
  edge(<f0>, <g>, "-|>", [F0 (励振)], label-side: right),
  edge(<tb>, <g>, "-|>", [$s$ (AdaIN)], label-side: right),
  edge(<g>, <out>, "-|>"),
)

// ===== 図: CIPT 学習 =====
#let dia_cipt = diagram(
  node-stroke: 0.7pt + rgb("#334"), node-corner-radius: 3pt, spacing: (11mm, 5.2mm),
  node((0,0), [女性 content\ /F0/energy], fill: cin, name: <fc>),
  node((0,1.1), [女性 参照→$s$], fill: cin, name: <fr>),
  node((0,3.0), [*男性* content], fill: cin, name: <mc>),
  node((0,3.9), [srcshift F0\ /energy], fill: cin, name: <mf>),
  node((0,4.9), [*女性 target*\ 参照→$s_"tgt"$], fill: cin, name: <mt>),
  node((2.7,2.4), align(center)[*Generator*\ #text(7pt)[NSF-HiFiGAN\ (fine-tune)]], fill: cgen, width: 26mm, height: 16mm, name: <g>),
  node((4.6,0.7), [$hat(y)$\ #text(7pt)[(self)]], fill: cgen, name: <yh>),
  node((4.6,4.1), [$hat(y)_x$\ #text(7pt)[(cross)]], fill: cgen, name: <xy>),
  node((6.6,0.0), [$L_"mel"$+$L_"mrstft"$\ #text(7pt)[($hat(y)$ vs 実 $y$)]], fill: closs, name: <lr>),
  node((6.4,1.6), align(center)[*Discriminator2*\ #text(7pt)[real=$y$ / fake=$hat(y),hat(y)_x$]], fill: cdis, width: 30mm, name: <d>),
  node((8.6,1.6), [$L_"GAN"$+$L_"FM"$], fill: closs, name: <lg>),
  node((6.7,3.2), [ECAPA →\ #text(7pt)[$L_"idout"$ ($hat(y)_x$→$e_"tgt"$)]], fill: caux, stroke: (dash: "dashed"), name: <ec>),
  node((6.7,4.6), [ContentVec →\ #text(7pt)[$L_"構音"$ ($hat(y)_x$ vs 男性c)]], fill: caux, stroke: (dash: "dashed"), name: <cvn>),
  edge(<fc>, <g>, "-|>"), edge(<fr>, <g>, "-|>"),
  edge(<mc>, <g>, "-|>"), edge(<mf>, <g>, "-|>"), edge(<mt>, <g>, "-|>", [$s_"tgt"$], label-side: right),
  edge(<g>, <yh>, "-|>"), edge(<g>, <xy>, "-|>"),
  edge(<yh>, <lr>, "-|>"), edge(<yh>, <d>, "-|>"), edge(<xy>, <d>, "-|>"), edge(<d>, <lg>, "-|>"),
  edge(<xy>, <ec>, "-|>"), edge(<xy>, <cvn>, "-|>"),
)

// ===== 図: 生成器 =====
#let dia_stack = diagram(
  node-stroke: 0.7pt + rgb("#334"), node-corner-radius: 3pt, spacing: (7mm, 4.0mm),
  node((0,0), [cond [770, T]], fill: cin, name: <cond>),
  node((0,1), [conv_pre 770→512], fill: cmod, width: 40mm, name: <pre>),
  node((0,2), [UpBlock 0  ↑8 · 512→256], fill: cgen, width: 40mm, name: <u0>),
  node((0,3), [UpBlock 1  ↑4 · 256→128], fill: cgen, width: 40mm, name: <u1>),
  node((0,4), [UpBlock 2  ↑2 · 128→64], fill: cgen, width: 40mm, name: <u2>),
  node((0,5), [UpBlock 3  ↑2 · 64→32], fill: cgen, width: 40mm, name: <u3>),
  node((0,6), [UpBlock 4  ↑2 · 32→16], fill: cgen, width: 40mm, name: <u4>),
  node((0,7), [UpBlock 5  ↑2 · 16→8], fill: cgen, width: 40mm, name: <u5>),
  node((0,8), [conv_post + tanh  8→1], fill: cmod, width: 40mm, name: <post>),
  node((0,9), [波形 [1, 512·T]], fill: cout, name: <wav>),
  node((-1.85,3), align(center)[Source module\ #text(7pt)[SineGen 9調波\ →Linear→tanh]], fill: csrc, name: <sm>),
  node((-1.85,4.5), [F0], fill: cin, name: <f0b>),
  node((2.0,4.5), align(center)[$s$ (192)], fill: cmod, name: <sb>),
  edge(<cond>, <pre>, "-|>"), edge(<pre>, <u0>, "-|>"), edge(<u0>, <u1>, "-|>"), edge(<u1>, <u2>, "-|>"),
  edge(<u2>, <u3>, "-|>"), edge(<u3>, <u4>, "-|>"), edge(<u4>, <u5>, "-|>"),
  edge(<u5>, <post>, "-|>"), edge(<post>, <wav>, "-|>"), edge(<f0b>, <sm>, "-|>"),
  edge(<sm>, <u2>, "..|>", [励振], label-side: left), edge(<sm>, <u4>, "..|>"),
  edge(<sb>, <u1>, "..|>", [AdaIN], label-side: right), edge(<sb>, <u4>, "..|>"),
)
#let dia_block = diagram(
  node-stroke: 0.7pt + rgb("#334"), node-corner-radius: 3pt, spacing: (6mm, 4.3mm),
  node((0,0), [入力 $x$], fill: white, name: <i>),
  node((0,1), [ConvTranspose1d\ #text(7pt)[↑rate, ch/2]], fill: cgen, width: 32mm, name: <ct>),
  node((0,2), [$plus$ noise_conv(source)], fill: csrc, width: 32mm, name: <add>),
  node((0,3), [AdaIN\ #text(7pt)[IN(x)(1+γ(s))+β(s)]], fill: cmod, width: 32mm, name: <ada>),
  node((0,4.1), align(center)[MRF\ #text(7pt)[ResBlock k3+k7+k11]], fill: cgen, width: 32mm, name: <mrf>),
  node((0,5.1), [出力 $x'$], fill: white, name: <o>),
  edge(<i>,<ct>,"-|>"), edge(<ct>,<add>,"-|>"), edge(<add>,<ada>,"-|>"), edge(<ada>,<mrf>,"-|>"), edge(<mrf>,<o>,"-|>"),
)

// ===== 図: ストリーミング推論 =====
#let ok = text(fill: rgb("#1a7a1a"))[✓因果]
#let bd = text(fill: rgb("#b58900"))[△有界先読み]
#let no = text(fill: rgb("#b00"))[✗要causal化]
#let dia_rt = diagram(
  node-stroke: 0.7pt + rgb("#334"), node-corner-radius: 4pt, spacing: (8mm, 6mm),
  node((0,0), [Target 参照\ (女性)], fill: caux, name: <tref>),
  node((2.3,0), [TimbreEncoder], fill: caux, name: <te>),
  node((4.4,0), [$s$ (192)\ #text(7pt)[1回・cache]], fill: caux, name: <s>),
  node((0,2), [入力フレーム\ #text(7pt)[hop 11.6ms]], fill: cin, name: <in>),
  node((2.3,2), [Causal\ Content Enc.\ #text(6.5pt)[#no #text(fill:rgb("#555"))[(ContentVec蒸留)]]], fill: cmod, width: 30mm, name: <ce>),
  node((4.4,2), [ContentScrub\ #text(6.5pt)[#ok]], fill: cmod, name: <sc>),
  node((2.3,3.15), [Causal F0\ #text(6.5pt)[#bd]], fill: cmod, name: <f0r>),
  node((4.4,3.15), [Energy\ #text(6.5pt)[#ok]], fill: cmod, name: <en>),
  node((6.1,2.4), [⊕], fill: white, shape: fletcher.shapes.circle, name: <catr>),
  node((7.9,2.4), align(center)[Causal\ *NSF-HiFiGAN*\ #text(6.5pt)[#bd → causal化で #ok]], fill: cgen, width: 30mm, name: <gr>),
  node((10.1,2.4), [出力フレーム], fill: cout, name: <outr>),
  edge(<tref>,<te>,"-|>"), edge(<te>,<s>,"-|>"),
  edge(<s>,<gr>,"-|>", [AdaIN], label-side: left),
  edge(<in>,<ce>,"-|>"), edge(<ce>,<sc>,"-|>"), edge(<in>,<f0r>,"-|>"),
  edge(<sc>,<catr>,"-|>"), edge(<f0r>,<catr>,"-|>"), edge(<en>,<catr>,"-|>"),
  edge(<f0r>,<gr>,"-|>"), edge(<catr>,<gr>,"-|>"), edge(<gr>,<outr>,"-|>"),
)

// ================= 本文 =================
#align(center)[
  #text(size: 18pt, weight: "bold", fill: accent)[LightVC — CIPT で話者天井を破る]
  #v(-0.3em)
  #text(size: 10.5pt, fill: rgb("#555"))[出力側同一性の微分監督による 男性→ターゲット女性 ゼロショット音声変換]
  #v(-0.2em)
  #text(size: 8.5pt, fill: rgb("#888"))[第2版 / 2026-07-08]
]
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))

#box2(fill: rgb("#eef4f8"))[
  *要旨* — 基盤モデル m3b は自然な女性声を出せるが、出力の話者類似(SECS)が *0.5 で頭打ち*になる「本人らしさの天井」を抱えていた。本稿はその原因を「男性 content の同一性漏れを、学習が出力側で一度も是正していない」ことと特定し、*CIPT (Cross-Identity Perceptual Training)* — 生成音声の話者埋め込みをターゲットへ微分監督する自前機構 — を提案する。CIPT により held-out 男性→女性の SECS を *0.5 → 0.63* まで押し上げ、聴感でも本人性が段階的に向上した。外部VC手法の移植には依らない(Frontier)。
]

= 1. 基盤モデル m3b と推論経路

目標は ASMR・官能バ美肉向けリアルタイム VC(推論 Rust/Candle・学習 PyTorch・MIT・E2E 50ms 未満)。`m3b` は source の content(何を)× target の identity(誰の声)× F0(高さ)を分離する source-filter 型 VC。推論経路を図1 に示す。

#v(0.2em)
#align(center)[#scale(72%, reflow: true)[#dia_infer]]
#figcap[1][推論。ContentVec(話者分離content)+srcshift F0+energy を、ContentScrub(GRL)後に NSF-HiFiGAN 生成器へ。話者は参照 mel→TimbreEncoder(ECAPA蒸留)の埋込 $s$ を AdaIN 注入。補助モデルは学習のみ、推論は自前経路。]

#box2[
  *主要諸元*: 44.1kHz / hop512 / 86Hz フレーム / mel128。*content* = ContentVec(768, HuBERT系)。*話者* $s$(192, ECAPA-TDNN へ cosine 蒸留)。*F0* = source 実コントゥアを target 中央値へ線形移動(srcshift)。学習型F0は warble で不採用。
]

= 2. 問題: 本人らしさの天井

決定的な切り分け: *女性発話の自己再構成は人物同一性 二重丸(SECS≈0.9)*だが、*男性→女性変換は SECS≈0.5 で頭打ち*で、参照埋込を生 ECAPA に差し替えても超えない。

#box2(fill: rgb("#fdecec"))[
  *診断*: decoder は本来 identity を出せる(recon が証拠)。天井の原因は *男性 content が source 同一性を漏らし*、decoder がそれを忠実にレンダリングすること。かつ m3b の学習は id loss を *埋め込み*(TimbreEncoder ≈ ECAPA)にのみ課し、mel/GAN は全て *自己再構成*。→ *「他人(男性) content に target 同一性を上書きする」出力を一度も監督していない*。これが天井の根。
]

反復自己精製(出力を再変換)で SECS が上がる事実は原理を裏付けたが、多重ループは再 vocode で構音を破壊するため不可。→ *壊さず1パスで*出力側を監督する機構が必要。

= 3. CIPT: 出力側同一性の微分監督

*原理*: cross-speaker 変換(content=話者X, target=話者Z)を生成し、生成音声の ECAPA 埋込を $e_Z$ へ微分逆伝播する。ECAPA は loss のみ(推論非依存, 合法)。

#box2[
  *予備実験 A1(失敗・重要な負)*: decoder を凍結し content 精製器のみ学習 → held-out SECS は +0.03 の微小ピーク後に減衰。*凍結 decoder + content 微調整では identity を頑健に押せない* → decoder 自体に容量を与える必要。
]

*本手法 A2*: 図2 の二ストリームで decoder(+TimbreEncoder+ContentScrub)を fine-tune。

#v(0.2em)
#align(center)[#scale(74%, reflow: true)[#dia_cipt]]
#figcap[2][CIPT 学習。共有 Generator に2ストリーム。*self-recon*(女性, GTあり): mel+MR-STFT+GAN+id で品質・構音を担保。*cross*(男性→女性, GTなし): 生成 $hat(y)_x$ の ECAPA を $e_"tgt"$ へ($L_"idout"$)、ContentVec を男性 content に一致($L_"構音"$)、さらに $hat(y)_x$ を Discriminator の fake に入れ実音声多様体へ拘束。]

#box2[
  *設計の要点*(いずれも成立に必須): (1) *decoder fine-tune*(凍結不可)。 (2) *male→female 実分布*で cross 学習(女性crossのみでは汎化不足)。 (3) *cross 出力を GAN の fake* に — ECAPA を騙すだけのノイズ化を実音声多様体で抑止。 (4) *ContentVec 一致*で構音(何を喋るか)を固定。 (5) *女性 self-recon* で品質担保。 \
  *cross 損失*: $L_"cross" = 3{-}5 dot L_"idout" + 6 dot L_"構音" + 1{-}2 dot L_"xGAN"$。self は m3b と同一(45 mel + 2 mrstft + 3 id + GAN + FM)。
]

= 4. 結果: 天井の突破

held-out 男性→女性2ペアの SECS(ECAPA cosine, 高いほど本人)。source→target は元来 0.0〜0.1。

#set text(size: 9pt)
#table(columns: (2.2fr, 1fr, 1fr, 1fr, 1fr), inset: 5pt, align: (left, center, center, center, center),
  stroke: 0.4pt + rgb("#bbd"), fill: (_, row) => thead(row),
  table.header([#hd[段階]], [#hd[pair0]], [#hd[pair1]], [#hd[flat 2-8k↓]], [#hd[聴感]]),
  [source→target (元)], [+0.10], [+0.00], [—], [別人],
  [m3b (天井)], [+0.446], [+0.508], [0.51/0.53], [女性だが本人不足],
  [CIPT w3 (初成功)], [+0.478], [+0.570], [0.47/0.67], [本人ぽい],
  [*CIPT w5 (押し込み)*], [*+0.508*], [*+0.632*], [0.56/0.74], [*本人・やや荒い*],
  [target (参照上限)], [—], [—], [0.45/0.59], [—],
)
#set text(size: 10pt)

#box2(fill: rgb("#eaf5ea"))[
  *pair1 で +0.508 → +0.632*(m3b比 +0.124)と 0.5 天井を明確に突破。学習は *単調上昇・無劣化で完走*し、cross-GAN が ECAPA 敵対的騙し(ノイズ化)を防いだ。聴感でも m3b→w3→w5 と本人性が段階的に向上(耳判定で確認)。
]

*知見*: SECS は知覚を過小評価する指標(kNN-VC は高 SECS でも音質不良で失格した前例)。従って最終判定は常に *耳*。CIPT の SECS 上昇は聴感一致で裏付けられた。

= 5. 音質の詰め (Phase B, 進行中)

w5 は本人性で合格だが「透明感・ツヤ不足／やや荒い」との kansei 指摘。スペクトル診断で原因を客観化した(表: *spectral flatness 2-8k* = 倍音間ノイズ=荒さ指標)。

#box2[
  target 0.45–0.59 / *w5 0.56–0.74(全手法中最悪)* / w3 0.47–0.67 / m3b 0.51–0.53。 → *強い identity 押し込み(w_idout=5)がスペクトルにノイズを乗せた*のが「荒さ」の正体。高域エネルギー自体は過剰だがノイジーで、透明感/ツヤを masking している。
]

*対策(anneal)*: identity は既に重みに焼き付いた(banked)ため、w5 から *w_idout を下げ(→2)・cross-GAN を強め(→2)* て再学習。GAN が実女性音声の clean な多様体へ引き戻し、flatness を target 水準へ下げる。荒さ除去で透明感・ツヤが顕在化する狙い(耳で確定予定)。

= 6. リアルタイム/ストリーミング推論 (課題)

RVC は ContentVec が双方向 self-attention のため、音声をチャンクに区切って buffer + crossfade する。これがチャンク遅延(~100–300ms)の主因。本アーキは *フレーム同期のストリーミングが可能な設計*だが、現行 checkpoint は同じく双方向 ContentVec に依存するため *そのままでは非ストリーミング*。以下に道筋を示す。

#v(0.2em)
#align(center)[#scale(70%, reflow: true)[#dia_rt]]
#figcap[3][ストリーミング推論経路。話者 $s$ は target 参照から1回計算しキャッシュ(灰=オフライン)。以降は入力フレーム毎に因果処理。因果性: #ok / #bd / #no。]

#set text(size: 9pt)
#table(columns: (2fr, 1.1fr, 3fr), inset: 5pt, align: (left, center, left), stroke: 0.4pt + rgb("#bbd"),
  fill: (_, row) => thead(row),
  table.header([#hd[段]], [#hd[因果性]], [#hd[備考]]),
  [ContentVec (content)], [✗ 双方向], [*唯一の本質的ボトルネック*。未来フレームを見る],
  [TimbreEncoder ($s$)], [✓ オフライン], [target 参照から1回・キャッシュ。毎フレーム不要],
  [ContentScrub], [✓ 完全因果], [kernel=1 pointwise conv],
  [Energy], [✓ 完全因果], [フレーム RMS],
  [F0 (srcshift)], [△ 小先読み], [因果 pitch tracker / neural F0 で ~0–20ms],
  [NSF-HiFiGAN 生成器], [△ 有界先読み], [受容野ぶん(数フレーム)。causal conv 化でゼロ可],
  [CIPT (identity)], [✓ 無影響], [学習手法。推論経路を変えない],
)
#set text(size: 10pt)

#box2[
  *障壁と解*: ボトルネックは ContentVec の双方向性(RVC と同根)。ただし本研究には *「純 causal で双方向と同質・lookahead 不要」*の負の結果(生成経路で複数検証)があり、*未来を見なくても品質は落ちない*。よって解は *ContentVec を causal content encoder に蒸留*(オンラインで content 相当を因果生成) — decoder はそのまま流用可。話者 $s$・F0 が content から分離・前計算/因果な点が RVC と異なり、*チャンク不要のフレーム同期*を可能にする。
]

#box2(fill: rgb("#eef4f8"))[
  *レイテンシ試算*: frame hop = 512/44100 = 11.6ms。causal content enc.(0) + 低先読み decoder + causal F0 → *アルゴリズム遅延 ≈ 1–3 フレーム(12–35ms) + 演算時間*。目標 *E2E 50ms 未満は射程内*で、RVC のチャンク遅延を大きく下回れる。演算: NSF-HiFiGAN ~14M は GPU/XPU で余裕 RT、CPU(Rust/Candle)でも最適化で RT 可。 \
  *残工程*: (1) causal content encoder の蒸留、(2) 生成器の causal conv 化(or 有界先読み許容)、(3) causal F0。最大の不確実性(causal で品質劣化?)は既に否定済み。
]

= 7. 生成器詳細(参考)

#grid(columns: (auto, auto), column-gutter: 10mm, align: top,
  [#align(center)[#text(9pt, weight: "bold")[(a) 生成器スタック]] #scale(66%, reflow: true)[#dia_stack]],
  [#align(center)[#text(9pt, weight: "bold")[(b) UpBlock]] #scale(66%, reflow: true)[#dia_block]
   #v(1mm)
   #box2[#text(8pt)[6 段で ch 半減・長さ×512(=hop)。F0 由来 source を各段励振、$s$ を AdaIN(零初期化 film)。生成器 約14M。推論依存 = 生成器+TimbreEncoder+ContentScrub+ContentVec のみ。]]],
)

= 8. 知見と今後

#box2[
  *検証して不採用にした経路(負の知見)*: 学習F0(warble「婆さん」) / 構音clone s_art(自己再構成では勾配出ず, 2回) / cross-style parallel(DTW artifact) / few-shot(男性content漏れ不変) / kNN-VC(音質不良・SECS劣位, 外部手法) / CIPT-A1(凍結decoderでは押せず)。 \
  *成立した本線*: *CIPT-A2* — 出力側同一性の微分監督 + decoder fine-tune + cross-GAN + ContentVec 構音固定。
]

#box2(fill: rgb("#eef4f8"))[
  *今後*: (Phase B) 音質 anneal で透明感/ツヤ。(Phase C) 萌え/官能の kansei 軸 — 出力側勾配が通る今、self-recon では動かなかった構音/スタイル軸を出力監督で再挑戦(human-in-the-loop)。並行して リアルタイム/streaming(E2E 50ms 未満)と Rust/Candle パリティ。
]

#v(0.4em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#align(center)[#text(size: 8pt, fill: rgb("#888"))[LightVC / CIPT — 推論 Rust(Candle) 目標・学習 PyTorch・MIT · 補助モデル(ECAPA/ContentVec)は loss/監督のみ]]
