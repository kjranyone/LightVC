#import "@preview/fletcher:0.5.5" as fletcher: diagram, node, edge
#set page(width: auto, height: auto, margin: 10pt)
#set text(font: ("Noto Sans CJK JP", "DejaVu Sans"), size: 9pt)

#let cin = rgb("#fbe4c8")
#let cmod = rgb("#dbe9f4")
#let cgen = rgb("#d7ecd7")
#let cout = rgb("#f6d5d5")
#let caux = rgb("#eee")

#diagram(
  node-stroke: 0.7pt + rgb("#334"),
  node-corner-radius: 4pt,
  spacing: (10mm, 7mm),
  node((0,1), [Source 波形\ (男性)], fill: cin, name: <src>),
  node((0,3.5), [Target 参照\ (女性, 数秒)], fill: cin, name: <tgt>),

  node((2,0), [ContentVec\ #text(7pt)[768ch]], fill: cmod, name: <cv>),
  node((2,1), [F0 抽出\ + srcshift], fill: cmod, name: <f0>),
  node((2,2), [Energy], fill: cmod, name: <eng>),
  node((2,3.5), [TimbreEncoder], fill: cmod, name: <tb>),
  node((2,4.7), [ECAPA 教師\ #text(7pt)[(蒸留・学習時のみ)]], fill: caux, stroke: (dash: "dashed"), name: <ec>),

  node((4,0), [ContentScrub\ #text(7pt)[GRL]], fill: cmod, name: <sc>),
  node((5.3,1), [⊕], fill: white, shape: fletcher.shapes.circle, name: <cat>),
  node((7,1.6), align(center)[*NSF-HiFiGAN*\ *Generator*], fill: cgen, width: 26mm, height: 15mm, name: <g>),
  node((9.3,1.6), [変換波形\ 44.1kHz], fill: cout, name: <out>),

  edge(<src>, <cv>, "-|>"),
  edge(<src>, <f0>, "-|>"),
  edge(<src>, <eng>, "-|>"),
  edge(<tgt>, <tb>, "-|>"),
  edge(<ec>, <tb>, "..|>", [distill], label-side: left),
  edge(<cv>, <sc>, "-|>"),
  edge(<sc>, <cat>, "-|>", [$c'$ 768]),
  edge(<f0>, <cat>, "-|>", [logF0]),
  edge(<eng>, <cat>, "-|>", [logE]),
  edge(<cat>, <g>, "-|>", [cond 770]),
  edge(<f0>, <g>, "-|>", [F0 (励振)], label-side: right),
  edge(<tb>, <g>, "-|>", [$s$ (AdaIN)], label-side: right),
  edge(<g>, <out>, "-|>"),
)
