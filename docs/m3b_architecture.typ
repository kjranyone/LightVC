#import "@preview/fletcher:0.5.5" as fletcher: diagram, node, edge

#set document(title: "LightVC — m3b 基盤VC アーキテクチャ")
#set page(paper: "a4", margin: (x: 2.0cm, y: 2.0cm), numbering: "1 / 1")
#set text(font: ("Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"), size: 10pt, lang: "ja")
#set par(justify: true, leading: 0.7em)
#show heading.where(level: 1): it => block(above: 1.3em, below: 0.7em)[
  #set text(size: 15pt, weight: "bold", fill: rgb("#1a4d6d")); #it.body]
#show heading.where(level: 2): it => block(above: 1.0em, below: 0.5em)[
  #set text(size: 11.5pt, weight: "bold", fill: rgb("#2a6f8f")); #it.body]

#let accent = rgb("#1a4d6d")
#let box2(body, fill: rgb("#eef4f8")) = box(
  fill: fill, inset: 7pt, radius: 5pt, stroke: 0.6pt + accent, width: 100%)[#body]
#let cin = rgb("#fbe4c8")
#let cmod = rgb("#dbe9f4")
#let cgen = rgb("#d7ecd7")
#let cout = rgb("#f6d5d5")
#let csrc = rgb("#f4e8f7")
#let cdis = rgb("#e7dcf3")
#let caux = rgb("#eee")
#let figcap(n, t) = align(center)[#text(size: 9pt, fill: rgb("#555"))[*図 #n.* #t]]

// ---------- 図1: 推論システム ----------
#let dia_infer = diagram(
  node-stroke: 0.7pt + rgb("#334"), node-corner-radius: 4pt, spacing: (9mm, 6.5mm),
  node((0,1), [Source 波形\ (男性)], fill: cin, name: <src>),
  node((0,3.5), [Target 参照\ (女性,数秒)], fill: cin, name: <tgt>),
  node((2,0), [ContentVec\ #text(7pt)[768ch]], fill: cmod, name: <cv>),
  node((2,1), [F0 抽出\ + srcshift], fill: cmod, name: <f0>),
  node((2,2), [Energy], fill: cmod, name: <eng>),
  node((2,3.5), [TimbreEncoder], fill: cmod, name: <tb>),
  node((2,4.7), [ECAPA 教師\ #text(7pt)[(蒸留・学習時のみ)]], fill: caux, stroke: (dash: "dashed"), name: <ec>),
  node((4,0), [ContentScrub\ #text(7pt)[GRL]], fill: cmod, name: <sc>),
  node((5.3,1), [⊕], fill: white, shape: fletcher.shapes.circle, name: <cat>),
  node((7,1.6), align(center)[*NSF-HiFiGAN*\ *Generator*], fill: cgen, width: 26mm, height: 15mm, name: <g>),
  node((9.3,1.6), [変換波形\ 44.1kHz], fill: cout, name: <out>),
  edge(<src>, <cv>, "-|>"), edge(<src>, <f0>, "-|>"), edge(<src>, <eng>, "-|>"),
  edge(<tgt>, <tb>, "-|>"), edge(<ec>, <tb>, "..|>", [distill], label-side: left),
  edge(<cv>, <sc>, "-|>"),
  edge(<sc>, <cat>, "-|>", [$c'$ 768]), edge(<f0>, <cat>, "-|>", [logF0]), edge(<eng>, <cat>, "-|>", [logE]),
  edge(<cat>, <g>, "-|>", [cond 770]),
  edge(<f0>, <g>, "-|>", [F0 (励振)], label-side: right),
  edge(<tb>, <g>, "-|>", [$s$ (AdaIN)], label-side: right),
  edge(<g>, <out>, "-|>"),
)

// ---------- 図2: 生成器 ----------
#let dia_stack = diagram(
  node-stroke: 0.7pt + rgb("#334"), node-corner-radius: 3pt, spacing: (7mm, 4.0mm),
  node((0,0), [cond [770, T]], fill: cin, name: <cond>),
  node((0,1), [conv_pre  7×1 · 770→512], fill: cmod, width: 42mm, name: <pre>),
  node((0,2), [UpBlock 0  ↑8 · 512→256], fill: cgen, width: 42mm, name: <u0>),
  node((0,3), [UpBlock 1  ↑4 · 256→128], fill: cgen, width: 42mm, name: <u1>),
  node((0,4), [UpBlock 2  ↑2 · 128→64], fill: cgen, width: 42mm, name: <u2>),
  node((0,5), [UpBlock 3  ↑2 · 64→32], fill: cgen, width: 42mm, name: <u3>),
  node((0,6), [UpBlock 4  ↑2 · 32→16], fill: cgen, width: 42mm, name: <u4>),
  node((0,7), [UpBlock 5  ↑2 · 16→8], fill: cgen, width: 42mm, name: <u5>),
  node((0,8), [conv_post + tanh  8→1], fill: cmod, width: 42mm, name: <post>),
  node((0,9), [波形 [1, 512·T]], fill: cout, name: <wav>),
  node((-1.9,4.5), align(center)[F0], fill: cin, name: <f0>),
  node((-1.9,3), align(center)[Source module\ #text(7pt)[SineGen 9調波\ →Linear→tanh]], fill: csrc, name: <sm>),
  node((2.0,4.5), align(center)[$s$ (192)\ #text(7pt)[話者埋込]], fill: cmod, name: <s>),
  edge(<cond>, <pre>, "-|>"), edge(<pre>, <u0>, "-|>"), edge(<u0>, <u1>, "-|>"), edge(<u1>, <u2>, "-|>"),
  edge(<u2>, <u3>, "-|>"), edge(<u3>, <u4>, "-|>"), edge(<u4>, <u5>, "-|>"),
  edge(<u5>, <post>, "-|>"), edge(<post>, <wav>, "-|>"), edge(<f0>, <sm>, "-|>"),
  edge(<sm>, <u2>, "..|>", [励振], label-side: left), edge(<sm>, <u4>, "..|>"),
  edge(<s>, <u1>, "..|>", [AdaIN], label-side: right), edge(<s>, <u4>, "..|>"),
)
#let dia_block = diagram(
  node-stroke: 0.7pt + rgb("#334"), node-corner-radius: 3pt, spacing: (6mm, 4.3mm),
  node((0,0), [入力 $x$], fill: white, name: <i>),
  node((0,1), [ConvTranspose1d\ #text(7pt)[↑rate, ch/2]], fill: cgen, width: 33mm, name: <ct>),
  node((0,2), [$plus$ noise_conv(source)], fill: csrc, width: 33mm, name: <add>),
  node((0,3), [AdaIN\ #text(7pt)[IN(x)·(1+γ(s))+β(s)]], fill: cmod, width: 33mm, name: <ada>),
  node((0,4.1), align(center)[MRF\ #text(7pt)[ResBlock k3+k7+k11\ (dilations 1,3,5) → 平均]], fill: cgen, width: 33mm, name: <mrf>),
  node((0,5.2), [出力 $x'$], fill: white, name: <o>),
  edge(<i>,<ct>,"-|>"), edge(<ct>,<add>,"-|>"), edge(<add>,<ada>,"-|>"), edge(<ada>,<mrf>,"-|>"), edge(<mrf>,<o>,"-|>"),
)

// ---------- 図3: 学習 ----------
#let dia_train = diagram(
  node-stroke: 0.7pt + rgb("#334"), node-corner-radius: 3pt, spacing: (10mm, 5.2mm),
  node((0,0), [content\ #text(7pt)[(摂動)]], fill: cin, name: <c>),
  node((0,1.2), [F0], fill: cin, name: <f0>),
  node((0,2.1), [energy], fill: cin, name: <e>),
  node((0,3.3), [参照波形 rw], fill: cin, name: <rw>),
  node((0,4.6), [実波形 $y$], fill: cin, name: <y>),
  node((0,6), [ECAPA 教師 $e$\ #text(7pt)[(事前計算)]], fill: caux, stroke: (dash: "dashed"), name: <te>),
  node((2,0), [ContentScrub\ #text(7pt)[GRL]], fill: cmod, name: <sc>),
  node((2,3.3), [TimbreEncoder], fill: cmod, name: <tb>),
  node((2,5.5), [cemb_pred\ #text(7pt)[(GRL 敵対器)]], fill: cmod, name: <cp>),
  node((3.3,0.9), [⊕], fill: white, shape: fletcher.shapes.circle, name: <cat>),
  node((5,1.7), align(center)[*Generator*\ #text(7pt)[NSF-HiFiGAN]], fill: cgen, width: 24mm, height: 12mm, name: <g>),
  node((6.7,1.7), [$hat(y)$], fill: cgen, name: <yh>),
  node((8.4,3), align(center)[*Discriminator2*\ #text(7pt)[MPD5+MSD3+MRD3]], fill: cdis, width: 30mm, name: <d>),
  node((5,4.4), [$L_"mel"$×45 + $L_"mrstft"$×2], fill: cout, name: <lr>),
  node((10.6,3), [$L_"gadv"$×1 + $L_"fm"$×2], fill: cout, name: <la>),
  node((4,5.7), [$L_"id"$×3  #text(7pt)[$1-cos(s,e)$]], fill: cout, name: <li>),
  node((6.8,6.3), [$L_"cadv"$×0.1], fill: cout, name: <lc>),
  edge(<c>, <sc>, "-|>"), edge(<sc>, <cat>, "-|>", [$c'$]), edge(<f0>, <cat>, "-|>"), edge(<e>, <cat>, "-|>"),
  edge(<cat>, <g>, "-|>", [cond]), edge(<f0>, <g>, "-|>", [F0], label-side: right),
  edge(<rw>, <tb>, "-|>"), edge(<tb>, <g>, "-|>", [$s$]), edge(<g>, <yh>, "-|>"),
  edge(<yh>, <d>, "-|>"), edge(<y>, <d>, "-|>", [real], label-side: right), edge(<d>, <la>, "-|>"),
  edge(<yh>, <lr>, "-|>"), edge(<y>, <lr>, "-|>"),
  edge(<tb>, <li>, "-|>"), edge(<te>, <li>, "..|>"),
  edge(<sc>, <cp>, "-|>", [GRL], label-side: right), edge(<te>, <cp>, "..|>"), edge(<cp>, <lc>, "-|>"), edge(<te>, <lc>, "..|>"),
)

// ================= 本文 =================
#align(center)[
  #text(size: 20pt, weight: "bold", fill: accent)[LightVC — m3b 基盤VC]
  #v(-0.3em)
  #text(size: 11pt, fill: rgb("#555"))[男性(中性)→ターゲット女性 ゼロショット音声変換 / アーキテクチャと学習]
  #v(-0.2em)
  #text(size: 9pt, fill: rgb("#888"))[2026-07-08 時点スナップショット]
]
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))

= 1. 概要と目標

ASMR・官能バ美肉向けのリアルタイム音声変換。*推論は全て Rust (Candle)、学習は PyTorch*、MIT ライセンス、E2E 50ms 未満、XPU 対応を目標とする。`m3b` は「エフェクト音声」から脱し「ターゲット女性が喋っている」自然な声を成立させた *基盤 (foundation)* モデルであり、以降の本人性・萌え表現はこの上に積む。

*基本方針*: source の content(何を) × target の identity(誰の声) × F0(高さ) を分離し、推論時 GUI でリアルタイム補正可能に保つ。補助モデル(ContentVec / ECAPA)は loss・表現供給に使うが、*最終推論の本体アルゴリズムは自前*で構成する(Frontier 研究)。

= 2. 信号諸元

#box2[
  #grid(columns: (1fr, 1fr, 1fr, 1fr), row-gutter: 4pt,
    [*サンプル率*], [44,100 Hz], [*STFT / 窓*], [N_FFT 2048 / 2048],
    [*ホップ長*], [512 (11.6ms)], [*フレーム率*], [86.13 Hz],
    [*メル帯域*], [128], [*F0/energy*], [86Hz フレーム])
]

= 3. 推論パイプライン (システム全体)

#v(0.3em)
#align(center)[#scale(74%, reflow: true)[#dia_infer]]
#figcap[1][推論データフロー。橙=入出力、青=学習モジュール(推論で使用)、緑=生成器、破線=学習時のみ。source から content/F0/energy、target 参照から話者埋込 $s$ を得て生成。]

= 4. モジュール詳細

== 4.1 NSF-HiFiGAN デコーダ (`nsf_hn.py`)

Source-filter 型ニューラルボコーダ。HiFiGAN 生成器に *NSF(Neural Source-Filter)* 励振を統合。図2 に全体スタックと UpBlock 内部を示す。

#v(0.3em)
#align(center)[#scale(74%, reflow: true)[#grid(columns: (auto, auto), column-gutter: 12mm, align: top,
  [#align(center)[#text(9pt, weight: "bold")[(a) 生成器スタック]] #v(1.5mm) #dia_stack],
  [#align(center)[#text(9pt, weight: "bold")[(b) UpBlock 詳細]] #v(1.5mm) #dia_block])]]
#figcap[2][生成器。(a) cond→conv_pre→6段 UpBlock(chを半減しつつ長さを×512)→conv_post→波形。F0 由来の source を各段へ励振注入、$s$ を AdaIN 注入。(b) 各 UpBlock の内部。]

== 4.2 TimbreEncoder (話者埋め込み, ECAPA蒸留)

参照 mel(128) → 4層 Conv(256ch, stride で1/4) → *統計プーリング*(時間方向 mean+std) → 線形 → 192次元 $s$。
学習中に *ECAPA-TDNN 埋め込み(speechbrain spkrec-ecapa-voxceleb, 192次元)へ cosine 蒸留* し、強いゼロショット話者性を獲得。推論時は本エンコーダのみ使用(SV 非依存、Rust 移植可)。

== 4.3 ContentScrub (話者漏れ抑制, GRL)

Content(768) → Conv 768→512→768 の残差(最終層零初期化)。*勾配反転層(GRL)* 経由で話者分類器(回帰)を敵対学習し、content から残留話者を除去 → 話者性を埋め込み側に集約。

== 4.4 F0: srcshift

学習型 F0 予測器は *warble(揺れ)で「婆さん声」化するため不採用*。代わりに source の *実 F0 コントゥア*(自然な微変動)を保持しつつ、レジスタ(中央値)のみ target 参照へ線形移動: $F 0' = F 0_"src" dot ("med"_"tgt" \/ "med"_"src")$。source の抑揚を残しつつ女性音域へ。

= 5. 学習プロセス

学習時グラフ(推論と同一の生成器 + 学習時のみの弁別器・GRL 敵対器・ECAPA 教師)を次の横向きページの図3 に示す。

#page(flipped: true, margin: 1.6cm)[
  #v(1fr)
  #align(center)[#scale(84%, reflow: true)[#dia_train]]
  #v(0.6em)
  #figcap[3][学習グラフ。赤=損失。生成器(緑)は図1・2 と同一。ECAPA 教師埋込(事前計算,破線)へ TimbreEncoder を cosine 蒸留、ContentScrub は GRL で話者除去(cemb_pred 敵対器)。実波形 $y$/生成 $hat(y)$ を Discriminator2 で GAN+FM。弁別器・GRL 敵対器・ECAPA 教師は全て推論では破棄。]
  #v(1fr)
]

== 5.1 Warm-start チェーン

#box2[
  #grid(columns: (auto, 1fr), row-gutter: 5pt, column-gutter: 8pt,
    [*m1_v2*], [NSF-HiFiGAN ボコーダ (content→波形) を単体で試聴可能水準まで学習],
    [→ *m2_vc6*], [TimbreEncoder(AdaIN 音色) + ContentScrub + 多解像度弁別器を追加],
    [→ *m3*], [TimbreEncoder を *ECAPA へ cosine 蒸留*、content-scrub(GRL)、F0 は srcshift へ],
    [→ *m3b*], [*content 摂動(content_pert)* 学習でクロスジェンダーの content 汎化を強化])
]

== 5.2 損失 (生成器) と学習設定

#grid(columns: (1.15fr, 0.85fr), column-gutter: 8pt,
  box2[
    #grid(columns: (auto, auto, 1fr), row-gutter: 3.5pt, column-gutter: 5pt,
      [*Mel L1*], [×45], [mel L1(再構成中核)],
      [*MR-STFT*], [×2], [複数解像度スペクトル],
      [*Identity*], [×3], [$1 - cos(s, e)$ 蒸留],
      [*Content-adv*], [×0.1], [GRL 話者除去],
      [*GAN adv*], [×1], [LSGAN],
      [*Feature match*], [×2], [弁別器中間特徴])
  ],
  box2[
    #grid(columns: (auto, 1fr), row-gutter: 3.5pt, column-gutter: 5pt,
      [*データ*], [女性 2,760発話],
      [*セグメント*], [32フレーム],
      [*最適化*], [AdamW lr 2e-4],
      [*バッチ*], [16],
      [*GRL α*], [10k step で0.5],
      [*摂動*], [WORLD formant/pitch])
  ])

= 6. 推論ネットワーク詳細 (層仕様)

$B$=バッチ, $T$=フレーム数(86Hz)。出力は $T dot 512$ サンプル。

#set text(size: 8.5pt)
#table(columns: (auto, auto, auto, auto, auto), inset: 4pt, align: (left, left, center, center, left),
  stroke: 0.4pt + rgb("#bbd"),
  fill: (_, row) => if row == 0 { rgb("#1a4d6d") } else if calc.rem(row, 2) == 0 { rgb("#f2f6f9") },
  table.header([#text(fill: white)[*層*]], [#text(fill: white)[*種別*]], [#text(fill: white)[*ch in→out*]], [#text(fill: white)[*k / stride*]], [#text(fill: white)[*出力 [ch, len]*]]),
  [入力 cond / F0 / s], [—], [—], [—], [[770,T] / [T] / [192]],
  table.cell(colspan: 5, fill: rgb("#eaf0f4"))[#text(weight: "bold")[Source module (NSF 励振)]],
  [F0 upsample], [linear interp ×512], [—], [—], [[1, 512T]],
  [SineGen], [正弦調波(9本)+noise], [—], [—], [[512T, 9]],
  [l_linear + tanh], [Linear], [9→1], [—], [[1, 512T] = source],
  table.cell(colspan: 5, fill: rgb("#eaf0f4"))[#text(weight: "bold")[本体 (HiFiGAN 生成器 + AdaIN)]],
  [conv_pre], [Conv1d], [770→512], [7 / 1], [[512, T]],
  [up0 (+noise/AdaIN/MRF×3)], [ConvTranspose1d], [512→256], [16 / 8], [[256, 8T]],
  [up1 (+noise/AdaIN/MRF)], [ConvTranspose1d], [256→128], [8 / 4], [[128, 32T]],
  [up2 (+noise/AdaIN/MRF)], [ConvTranspose1d], [128→64], [4 / 2], [[64, 64T]],
  [up3 (+noise/AdaIN/MRF)], [ConvTranspose1d], [64→32], [4 / 2], [[32, 128T]],
  [up4 (+noise/AdaIN/MRF)], [ConvTranspose1d], [32→16], [4 / 2], [[16, 256T]],
  [up5 (+noise/AdaIN/MRF)], [ConvTranspose1d], [16→8], [4 / 2], [[8, 512T]],
  [conv_post + tanh], [Conv1d], [8→1], [7 / 1], [[1, 512T] = 波形],
)
#set text(size: 10pt)
#v(0.3em)
#box2[*AdaIN film*: 各段 $i$ で Linear(192→$2 dot "ch"_i$) が $(gamma_i,beta_i)$ を出力し $x <- "IN"(x)(1+gamma_i)+beta_i$(零初期化)。*生成器 約14M params*。推論依存=生成器+TimbreEncoder+ContentScrub+ContentVec のみ。]

== 6.1 Discriminator2 (学習のみ, 3系統11弁別器)

#set text(size: 8.5pt)
#table(columns: (auto, auto, 1fr), inset: 4pt, align: (left, center, left), stroke: 0.4pt + rgb("#bbd"),
  fill: (_, row) => if row == 0 { rgb("#1a4d6d") } else if calc.rem(row, 2) == 0 { rgb("#f2f6f9") },
  table.header([#text(fill: white)[*系統*]], [#text(fill: white)[*数*]], [#text(fill: white)[*構造*]]),
  [MPD (周期)], [5], [period 2,3,5,7,11。Conv2d 1→32→128→512→1024→1024→1、(5,1)stride(3,1)],
  [MSD (スケール)], [3], [生/×2/×4 pool。Conv1d 1→128→…→1024→1、grouped、kernel 41],
  [MRD (スペクトル)], [3], [STFT(512/1024/2048)、Conv2d on log-mag、32ch×4 → 1],
)
#set text(size: 10pt)
#v(0.2em)
#box2[*生成器総損失*: $L_G = 45 L_"mel" + 2 L_"mrstft" + 3 L_"id" + 0.1 L_"cadv" + L_"gadv" + 2 L_"fm"$。 *弁別器*: $L_D = sum_k (D_k (y)-1)^2 + D_k (hat(y))^2$ (LSGAN)。]

= 7. 到達状況と知見

#box2(fill: rgb("#eaf5ea"))[
  *成立*: ・デコーダ無罪(女性自己再構成=人物同一性 二重丸、mel-L1 0.5は聴感無関係)。 ・srcshift で F0 自然化。 ・ECAPA 蒸留+content摂動で SECS を source 0.02〜0.09 → *0.50〜0.58*(「コア声質は似てきた」)。
]
#v(0.3em)
#box2(fill: rgb("#fdecec"))[
  *課題*: 出力 SECS が生 ECAPA(天井)でも *0.5〜0.58 で頭打ち* = 男性 content が source 同一性を漏らし出力を引き戻す(女性 content 自己再構成が二重丸なのが傍証)。単一 ECAPA 埋込+AdaIN では固有の細部を押し切れない。
]
#v(0.3em)
#box2[
  *不採用(負の知見)*: 学習F0(warble「婆さん」) / 構音clone s_art(自己再構成では勾配出ず無効果, 2回) / cross-style parallel(DTW artifact+学習推論ミスマッチ) / few-shot(男性content漏れ不変) / kNN-VC(音質不良・SECS劣位, 外部手法ゆえ Frontier 方針で不採用)。
]

#v(0.5em)
#line(length: 100%, stroke: 0.5pt + rgb("#ccc"))
#align(center)[#text(size: 8.5pt, fill: rgb("#888"))[LightVC / m3b — 推論 Rust(Candle) 目標・学習 PyTorch・MIT]]
