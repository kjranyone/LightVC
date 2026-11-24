# LightVC デザイン・システム(M3 Expressive 準拠・2026-11-22 v5)

status: 適用済み。トークンと部品 = `crates/lightvc-app/src/m3.rs`、画面 = `crates/lightvc-app/src/ui.rs`。
v4 までは色が手書き・部品の寸法が仕様外・ナビゲーション部品なしで、M3E に準拠していなかった(全面的に置き換えた)。

## 0. 前提

- **無駄な物は表示しない**: 機能・情報・操作に結びつかない装飾要素(意味のないアイコン枠・ダミーのバッジ等)を置かない。
- 筐体: GPUI 0.2.2(Windows)。hover / pressed は即時(暗黙の色遷移は無い)。形の変化と動きは
  `with_animation` と `canvas`(パスの描画)で作る。
- ブランド: きゃわわ Future Bass(ピンク × ラベンダー × ミント)を M3 の色ロールに載せる。
- 文言は日本語に統一する(英語を混ぜない)。フォントはシステム(DirectWrite)。
- **画面ごとに主役を 1 つ**決める。主役だけが大きい文字・濃い色・size M のボタンを持つ。

## 1. 色(HCT トーンパレットから生成)

seed: primary `#E05A8A`(ピンク)・secondary `#7C68BE`(ラベンダー)・tertiary `#2EB88C`(ミント)・
neutral = ラベンダーの色相で chroma 6(variant は 10)。Material の `material-color-utilities` の
`TonalPalette.fromHueAndChroma` で作り、M3 の標準トーン対応で取る。**hex を手で決めない**。

| ロール | light(tone) | dark(tone) |
|---|---|---|
| primary / on-primary | 40 / 100 | 80 / 20 |
| *-container / on-*-container | 90 / 30 | 30 / 90 |
| surface | 98 | 6 |
| surface-container / -high | 94 / 92 | 12 / 17 |
| on-surface / on-surface-variant | 10 / 30 | 90 / 80 |
| outline-variant | 80 | 30 |
| inverse-surface / inverse-on-surface | 20 / 95 | 90 / 20 |

- 値は `m3::c` の `role!(名前, light, dark)`。ダークは OS の外観に従う(`LVC_THEME=dark|light` で固定)。
- state layer: hover = 上に載る色を 8 %、pressed = 12 % 重ねる(`m3::mix` / `m3::alpha`)。
- 無効: 器 = on-surface 10 %・文字 = on-surface 38 %。
- 使い分け: primary = 主役の行動と「変換中」/ secondary = 選択(toggle・button group・chip・rail の指示子)/
  tertiary = 対になる状態(バイパス中・停止ボタン・変換結果)/ error = 音切れ・録音中・上限超過。

## 2. タイポグラフィ(M3 type scale)

| スタイル | size / line / weight | 用途 |
|---|---|---|
| display-large | 57 / 64 / bold | **主役の数値**(E2E 遅延)だけ |
| headline-small | 24 / 32 / semibold | 画面の題・補助の数値 |
| title-large | 22 / 28 / medium | hero の経路(あなた → 目標の声) |
| title-medium | 16 / 24 / semibold | 容器の見出し・size M のボタン |
| body-large / medium / small | 16 / 14 / 12 | 本文・パス・補足 |
| label-large / medium | 14 / 12 medium | ボタン・chip・rail・単位 |

## 3. シェイプ(M3 shape scale)

XS 4 / SM 8 / MD 12 / LG 16 / XL 28 / full。容器 = LG、主役の容器と本文の面 = XL、chip = SM、
snackbar = XS、ボタン = full。**形は状態で変わる**: 押すと角が締まる(full → 8)、選択で形が替わる
(toggle: round → square、segmented list: 選択行だけ丸い、開始 = round / 停止 = square)。

## 4. 部品(m3.rs)

| M3E | 実装 | 仕様 |
|---|---|---|
| Navigation rail | `nav_item` | 幅 96・指示子 56 × 32・選択でアイコンが塗り・指示子が spring で広がる |
| Button (S) | `button` + `Btn::{Filled,Tonal,Outlined,Text}` | 高さ 40・アイコン 20・1 つの容器に filled は 1 個まで |
| Button (M) | `button_m` | 高さ 56・画面の主役の行動に 1 個(開始/停止・変換する) |
| Toggle button | `toggle_button` | tonal。未選択 round(secondary-container)→ 選択 square(secondary) |
| Icon button | `icon_button` | 40 × 40・standard / tonal |
| Connected button group | `segments` + `segment` | 間隔 2・内側の角 8・外側と選択中は full |
| Filter chip | `chip` | 高さ 32・角 8・選択で secondary-container + check |
| Toolbar(floating) | ui.rs の `toolbar` | 高さ 64・full・surface-container-high・主役のボタンを隣に置く |
| Segmented list | `list` + `list_item` | 行ごとに面・間隔 2・端の行だけ外側の角 16・選択行は 28 |
| Linear indicator | `meter` | 太さ 8・針と溝の間 4・末端の停止点 |
| Loading indicator | `loading_indicator` | canvas のパス。6 つの形を spring で変形しながら回る |
| Snackbar | ui.rs(エラー) | inverse-surface・角 4・閉じる icon button |
| アイコン | `icon` | Material Symbols Rounded(Apache-2.0)を `assets/icons/*.svg` から埋め込む |

文字の絵文字・記号(✦ ⚙ ■ …)をアイコンの代わりに使わない。

## 5. レイアウト

window = surface-container の地に navigation rail(左 96)+ 本文の面(surface・角 28)。
本文は縦スクロールする(はみ出して切れない)。最小 900 × 600。

- **リアルタイム**(主役 = 今どう変換されているか): 主列 + 補助の列(320)。
  - hero(角 28): 状態で色が替わる(停止 = surface-container-high / 変換中 = primary-container /
    バイパス中 = tertiary-container)。状態の pill・ゲート・経路・**E2E 遅延(display-large)**と
    目標 30 ms / 上限 50 ms の判定・計算 p99 / RTF / ホスト・遅延の内訳・入出力の meter。
  - toolbar(バイパス・ミュートの toggle)+ **開始 / 停止(button M)**。
  - 調整(縁の容器): ピッチとバッファの connected button group。
  - 補助の列: 目標の声(区分「女性 / 男性 / 自分」の connected button group + その区分の chip。声が 30 以上あるので
    一覧には出さない)・操作者(録音 / WAV / 消去)・モデル(書き出しの場所・スレッド・読み込み)。
- **ファイル変換**(主役 = 変換する): 変換元と目標の声の 2 枚(目標の声はファイルのほか、同じ区分と chip でも選べる)→ 変換する(button M)+ ピッチ → 結果(tertiary-container)。
- **ボイス**(主役 = 一覧): 題の右に「ボイスを追加」(filled)・同梱のプリセット(見出しの右に表示フィルタ すべて / 女性 / 男性・
  segmented list(試聴))・自分で追加したボイス(segmented list(試聴・削除))・空の状態。
- **デバイス**: 表示フィルタ(すべて / ASIO / WASAPI)・入力と出力の segmented list(ホスト混在可)。
- **クレジット**: アプリ版数・学習済みモデルの由来・第三者ライセンス・`NOTICE` 全文展開。

## 6. モーション

M3E の spatial spring(行き過ぎて戻る)を `m3::spring` で作る(GPUI の easing は 0..1 に限られるため、
animator の中で掛ける)。

1. rail の指示子: 選択で幅 32 → 56(450 ms・spring)。
2. 画面の切り替え: 透明度と 14 px の上昇(260 ms・ease-out)。
3. loading indicator: 形の変形と回転(読み込み・変換の間だけ)。
4. 録音中の点の脈動。
5. 押下: 角が締まる(即時)。

描画は要求駆動のまま(動いている間だけ animation frame を回す)。

## 7. 検証

- `powershell -File tools\snap_tabs.ps1 -Prefix m3e_v5`(デモ表示で 4 画面を撮る。`-Theme dark` でダーク、
  `LVC_DEMO_STATE=loading` で読み込み中)。`docs/screenshots/m3e_v5_{realtime,offline,voices,devices}.png` と
  `m3e_v5_realtime_dark.png` で確認済み(2026-11-22): rail の指示子・hero の階層・button group の角・
  segmented list の選択行・ダークの配色・loading indicator の描画・日本語の描画。
- 画面で確認していないもの: 停止中 / バイパス中の hero・録音中・snackbar・変換結果・空のボイス一覧
  (デモ表示では出ない状態)。
- `cargo build --release -p lightvc-app` 通過・`cargo clippy -p lightvc-app` で ui.rs / m3.rs の警告なし。
- 同梱のボイスプリセット(2026-11-24): `voices_v1_{voices,realtime,offline}.png`(デモ表示・ライト)で確認済み: ボイスの見出しの右の
  表示フィルタ(すべて / 女性 / 男性)と同梱の行(試聴のみ・削除なし)・リアルタイムとファイル変換の「目標の声」の区分
  (女性 / 男性 / 自分)と chip。
- 他のインスタンスが動いている間は `tools\snap.ps1 -ProcessId <PID>` で撮る(既定では最初の lightvc-app を選ぶ)。
