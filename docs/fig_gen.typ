#import "@preview/fletcher:0.5.5" as fletcher: diagram, node, edge
#set page(width: auto, height: auto, margin: 10pt)
#set text(font: ("Noto Sans CJK JP", "DejaVu Sans"), size: 8.5pt)

#let cin = rgb("#fbe4c8")
#let cmod = rgb("#dbe9f4")
#let cgen = rgb("#d7ecd7")
#let cout = rgb("#f6d5d5")
#let csrc = rgb("#f4e8f7")

#let stack = diagram(
  node-stroke: 0.7pt + rgb("#334"),
  node-corner-radius: 3pt,
  spacing: (7mm, 4.2mm),
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

  edge(<cond>, <pre>, "-|>"),
  edge(<pre>, <u0>, "-|>"), edge(<u0>, <u1>, "-|>"), edge(<u1>, <u2>, "-|>"),
  edge(<u2>, <u3>, "-|>"), edge(<u3>, <u4>, "-|>"), edge(<u4>, <u5>, "-|>"),
  edge(<u5>, <post>, "-|>"), edge(<post>, <wav>, "-|>"),
  edge(<f0>, <sm>, "-|>"),
  edge(<sm>, <u2>, "..|>", [励振], label-side: left),
  edge(<sm>, <u4>, "..|>"),
  edge(<s>, <u1>, "..|>", [AdaIN], label-side: right),
  edge(<s>, <u4>, "..|>"),
)

#let block = diagram(
  node-stroke: 0.7pt + rgb("#334"),
  node-corner-radius: 3pt,
  spacing: (6mm, 4.5mm),
  node((0,0), [入力 $x$], fill: white, name: <i>),
  node((0,1), [ConvTranspose1d\ #text(7pt)[↑rate, ch/2]], fill: cgen, width: 33mm, name: <ct>),
  node((0,2), [$plus$ noise_conv(source)], fill: csrc, width: 33mm, name: <add>),
  node((0,3), [AdaIN\ #text(7pt)[IN(x)·(1+γ(s))+β(s)]], fill: cmod, width: 33mm, name: <ada>),
  node((0,4.1), align(center)[MRF\ #text(7pt)[ResBlock k3 + k7 + k11\ (dilations 1,3,5) → 平均]], fill: cgen, width: 33mm, name: <mrf>),
  node((0,5.2), [出力 $x'$], fill: white, name: <o>),
  edge(<i>,<ct>,"-|>"), edge(<ct>,<add>,"-|>"), edge(<add>,<ada>,"-|>"),
  edge(<ada>,<mrf>,"-|>"), edge(<mrf>,<o>,"-|>"),
)

#grid(columns: (auto, auto), column-gutter: 14mm, align: top,
  [#align(center)[*(a) 生成器スタック*] #v(2mm) #stack],
  [#align(center)[*(b) UpBlock 詳細*] #v(2mm) #block],
)
