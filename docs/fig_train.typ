#import "@preview/fletcher:0.5.5" as fletcher: diagram, node, edge
#set page(width: auto, height: auto, margin: 10pt)
#set text(font: ("Noto Sans CJK JP", "DejaVu Sans"), size: 8.5pt)

#let cin = rgb("#fbe4c8")
#let cmod = rgb("#dbe9f4")
#let cgen = rgb("#d7ecd7")
#let cdis = rgb("#e7dcf3")
#let closs = rgb("#f6d5d5")
#let caux = rgb("#eee")

#diagram(
  node-stroke: 0.7pt + rgb("#334"),
  node-corner-radius: 3pt,
  spacing: (11mm, 5.5mm),
  // inputs
  node((0,0), [content\ #text(7pt)[(摂動)]], fill: cin, name: <c>),
  node((0,1.2), [F0], fill: cin, name: <f0>),
  node((0,2.1), [energy], fill: cin, name: <e>),
  node((0,3.3), [参照波形 rw], fill: cin, name: <rw>),
  node((0,4.6), [実波形 $y$], fill: cin, name: <y>),
  node((0,6), [ECAPA 教師 $e$\ #text(7pt)[(事前計算)]], fill: caux, stroke: (dash: "dashed"), name: <te>),
  // modules
  node((2,0), [ContentScrub\ #text(7pt)[GRL]], fill: cmod, name: <sc>),
  node((2,3.3), [TimbreEncoder], fill: cmod, name: <tb>),
  node((2,5.5), [cemb_pred\ #text(7pt)[(GRL 敵対器)]], fill: cmod, name: <cp>),
  node((3.3,0.9), [⊕], fill: white, shape: fletcher.shapes.circle, name: <cat>),
  // generator
  node((5,1.7), align(center)[*Generator*\ #text(7pt)[NSF-HiFiGAN]], fill: cgen, width: 24mm, height: 12mm, name: <g>),
  node((6.7,1.7), [$hat(y)$], fill: cgen, name: <yh>),
  // disc
  node((8.4,3), align(center)[*Discriminator2*\ #text(7pt)[MPD5+MSD3+MRD3]], fill: cdis, width: 30mm, name: <d>),
  // losses
  node((5,4.4), [$L_"mel"$×45 + $L_"mrstft"$×2], fill: closs, name: <lr>),
  node((10.6,3), [$L_"gadv"$×1 + $L_"fm"$×2], fill: closs, name: <la>),
  node((4,5.6), [$L_"id"$×3\ #text(7pt)[$1-cos(s,e)$]], fill: closs, name: <li>),
  node((6.6,6.2), [$L_"cadv"$×0.1], fill: closs, name: <lc>),

  edge(<c>, <sc>, "-|>"),
  edge(<sc>, <cat>, "-|>", [$c'$]),
  edge(<f0>, <cat>, "-|>"),
  edge(<e>, <cat>, "-|>"),
  edge(<cat>, <g>, "-|>", [cond]),
  edge(<f0>, <g>, "-|>", [F0], label-side: right),
  edge(<rw>, <tb>, "-|>"),
  edge(<tb>, <g>, "-|>", [$s$]),
  edge(<g>, <yh>, "-|>"),
  edge(<yh>, <d>, "-|>"),
  edge(<y>, <d>, "-|>", [real], label-side: right),
  edge(<d>, <la>, "-|>"),
  edge(<yh>, <lr>, "-|>"),
  edge(<y>, <lr>, "-|>"),
  edge(<tb>, <li>, "-|>"),
  edge(<te>, <li>, "..|>"),
  edge(<sc>, <cp>, "-|>", [GRL], label-side: right),
  edge(<te>, <cp>, "..|>"),
  edge(<cp>, <lc>, "-|>"),
  edge(<te>, <lc>, "..|>"),
)
