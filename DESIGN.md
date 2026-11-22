# LightVC デザイン・システム(M3 Expressive 準拠・2026-11-22)

status: 適用済み。実装は `crates/lightvc-app/src/ui.rs`。根拠は Material 3 Expressive の 6 戦術
(use a variety of shapes / rich and nuanced colors / guide attention with typography /
contain content for emphasis / fluid and natural motion / component flexibility)。

## 0. 前提と制約

- 筐体: GPUI 0.2.2(Windows)。**暗黙の色遷移(hover の fade)は無い** = hover は即時。動きは
  `with_animation` + `bounce(ease_in_out)`(M3E の spatial spring「オーバーシュートして跳ねる」に相当)
  で明示的に作る。
- ブランド: きゃわわ Future Bass(ラベンダー×ピンク)を維持し、M3 の**色のロール(役割)**に載せ替える。
- 文章は日本語。フォントはシステム(DirectWrite)のフォールバック。
- 「要素を並べた」状態からの脱却が目的: **画面ごとに主役を 1 つ**決め、強調の階層で並べる。

## 1. 色(M3 のトーン・ロール)

seed = きゃわわピンク/ラベンダー。surface はラベンダーの淡いトーン列(明るいほど重要)。

| ロール | 値 | 用途 |
|---|---|---|
| primary | `#E05A8A` | 主役の行動(Start・Load・Convert)・選択済み・メーター針 |
| on-primary | `#FFFFFF` | primary の上の文字 |
| primary-container | `#FBD9E5` | 選択済み chip・強調の器 |
| on-primary-container | `#4A1B31` | 同上の文字 |
| secondary | `#7C68BE` | バッジ(BYPASS)・アクセント |
| secondary-container | `#E7DFF8` | tonal ボタン・Host 行 |
| on-secondary-container | `#2A2050` | 同上の文字 |
| tertiary | `#2EB88C` | CONVERTING/GATE(正常系)・Out メーター |
| tertiary-container | `#CCF1E2` | 正常の器 |
| on-tertiary-container | `#063828` | 同上の文字 |
| error | `#C03B5E` | Stop・over/underrun・REC |
| error-container | `#F9D6DF` | 警告の器 |
| surface | `#F7F1F9` | 画面背景 |
| surface-container-low | `#F2EBF5` | 設定・登録のカード |
| surface-container-high | `#E6DCEB` | **主役のカード**(運用) |
| surface-container-lowest | `#FFFFFF` | 診断(白・最も明るい) |
| on-surface | `#2B2233` | 基本の文字 |
| on-surface-variant | `#5D5266` | 補助の文字 |
| outline | `#86748F` | outlined の縁 |
| outline-variant | `#D9CDDF` | 区切り・メーターの溝 |

戦術「**明るさで階層を作る**」: 高 importance ほど明るい surface に載せる(M3E: brightest surface
mapping)。運用は container-high、登録は low、診断は白。

## 2. タイポグラフィ(M3 type scale + emphasized)

| スタイル | サイズ/重み | 用途 |
|---|---|---|
| display | 22 / bold | ワードマーク「✦ LightVC」 |
| **headline-emphasis** | 24 / bold | **主役の数値**(E2E 遅延・compute p99)。editorial moment |
| title-medium | 15 / semibold | カード見出し |
| body-medium | 13 / regular | 本文・パス |
| label-large | 12 / medium | ボタン・chip |
| label-small | 10.5 / regular(dim) | ヒント・単位・host |

戦術「**タイポで注意を導く**」: 画面で最も見せたい数値(遅延・p99)だけ headline-emphasis にする。
数値+単位は baseline を揃え、単位は label-small で薄く(M3 の units)。

## 3. シェイプ(shape scale + tension)

| トークン | radius(px) | 用途 |
|---|---|---|
| shape-xs | 4 | **tension 要素(診断カード)** |
| shape-md | 10 | カード(基調・角丸) |
| shape-lg | 14 | 主役カード |
| shape-full | 999 | ボタン・chip・connected button group・メーター・バッジ |

戦術「**round と square を混ぜて緊張を作る**」: 全体を角丸(full/md)で統一し、**診断カードだけ
四角(xs)**にして「ここは計測値」という注意を引く。M3E: "Break from the surrounding shape style
to draw attention"。

## 4. コンポーネント(M3E の部品の写像)

| M3E | LightVC 実装(ui.rs) | 備考 |
|---|---|---|
| Button (filled) | `filled_btn` | 高強調 = その画面で 1 つだけ(Realtime=Start・Offline=Convert) |
| Button (tonal) | `tonal_btn` | 中強調(Load・Record・Browse・Play) |
| Button (outlined) | `outlined_btn` | 低強調の枠付き(管理…・設定) |
| Text button | `text_btn` | 最低強調(Clear・Cancel・✕) |
| **Connected button group** | `btn_group` | Pitch(+0/+5/+7/+12)・Buffer・threads。**選択肢だけ膨らむ**(幅・full 角・primary 塗り)—— M3E の "selected button changes shape and width" の静的表現 |
| Filter chip | `chip` | Target/Voices の選択。選択で primary-container +「✓」 |
| **Loading indicator** | `loading(text)` | 3 点が `bounce` で順に脈動(GPUI に shape morph は無いため、M3E の "capture attention" を大きさの脈動で代用) |
| Linear progress | `meter` | 溝(outline-variant)・針(primary/tertiary)・**末端の停止ドット**(M3 の stop indicator) |
| Status badge | `badge` | CONVERTING=tertiary / BYPASS=secondary / STOPPED=on-surface-variant の小さな full 角 |
| REC indicator | 点滅する error の点(`bounce`・無限) | 進行中の合図に motion を使う |

## 5. レイアウト(containment と主役)

**Realtime タブ**(主役 = 今まさに変換がどう動いているか):
1. **運用(hero・surface-container-high・shape-lg)**: 状態バッジ + 対象の声 + GATE + 遅延/RTF(headline)
   + compute ms(p99 を headline)+ over/underrun + 輸送バー(Bypass/Mute/Pitch group + **Start(filled) 1 個** /
   Stop = error filled)。登録と診断より大きく・明るく。
2. **登録(surface-container-low・shape-md)**: Model(exports・threads group・操作者: Add/Record/Clear=低強調・
   Load=tonal)と Target の chip 列。
3. **診断(surface-container-lowest=白・shape-xs=tension)**: Input/Out メーター・host・Buffer group・
   E2E の帯(帯の分割は線形・M3E の wavy は GPUI で描けないため不採用)。

Offline タブ(主役 = Convert): Source/Reference(low)→ catalog chips → Pitch group → **Convert(filled・full)**
→ Output(白)。Voices タブ(主役 = 声の一覧): Add(tonal 1 個)→ 一覧(行 = chip 相当・選択で
primary-container・Play=tonal・✕=text)。

## 6. モーション

- 方針: M3E の spatial spring(オーバーシュート)≈ GPUI の `bounce(ease_in_out)`。効果は控えめに:
  1. loading indicator の 3 点の脈動(0.9s・無限・時差)
  2. REC の点の点滅(0.8s・無限)
  3. それ以外は静的(hover は即時)。GPUI に暗黙遷移が無いため、これ以上の遷移は作らない。
- 要求駆動の描画は維持(loading/REC の間だけ animation frame が回る)。

## 7. 検証

- 3 タブのスクリーンショット(docs/screenshots/m3e_realtime.png・m3e_voices.png)で確認済み(2026-11-22):
  主役カードが明るさ・文字・位置の 3 軸で支配すること、button group の選択肢の膨らみ
  (タブ・Pitch・threads・Buffer の 4 箇所で同じ扱い)、診断カードの白い四角の緊張、
  Target の「✓ chip」、メーターの停止ドット、日本語の描画(tofu なし)。
- `cargo build --release -p lightvc-app`・`cargo clippy -p lightvc-app --release` 通過。
