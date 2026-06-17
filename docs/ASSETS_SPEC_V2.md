# LightVC Visual Assets Specification V2 (UI Icons & Elements)

V1のアセット（ロゴ、背景、ノブ）に加えて、UIの各所で使用する小物アイコンやイラストの指示書です。
「Kawaii Future Bass」のテーマ（パステルカラー、丸み、グロー）に準拠します。

---

## 必要な素材リスト

### 1. UIアイコンセット（ボタン・インジケータ用）

テキストのみのボタンに添えて使う、小型のUIアイコンです。
eguiの `Button::image_and_text()` で組み込みます。

| ファイル名 | サイズ | 配色 | 用途・デザイン内容 |
|----------|--------|------|-------------------|
| `icon_folder.png` | 24×24 | Cyan | 「Browse」ボタン用。開いたフォルダの形。 |
| `icon_play.png` | 24×24 | Mint | 「▶ Play」ボタン用。丸みを帯びた三角。 |
| `icon_stop.png` | 24×24 | Pink | 「■ Stop」ボタン用。丸みを帯びた四角。 |
| `icon_convert.png` | 24×24 | Pink | 「✦ Convert」CTA用。キラキラしたスパンクル（星形）。 |
| `icon_trash.png` | 24×24 | Text Dim | 「✕ Remove」ボタン用。可愛いゴミ箱、またはシンプルなバツ印。 |
| `icon_mic.png` | 24×24 | Lavender | 「Source」や入力系のラベル用。マイクのシルエット。 |
| `icon_speaker.png` | 24×24 | Lavender | 「Output」や出力系のラベル用。スピーカーのシルエット。 |

※全て透過PNG（RGBA）。背景に馴染むよう、輪郭に1pxの薄いグロー（不透明度30%程度）を入れると綺麗です。

---

### 2. 空状態（Empty State）イラスト

画面内のリスト等が空の時に表示する、少し大きめのイラストです。

| ファイル名 | サイズ | 用途・デザイン内容 |
|----------|--------|-------------------|
| `empty_stars.png` | 200×200 | `Voice Catalog`タブで、ボイスが1つも登録されていない時に表示。<br>「星が3つくらい集まって寝ている」または「スパンクルが点線で繋がっている」ような、余白のある可愛いイラスト。<br>色は薄いLavenderとText Muted。 |

---

### 3. CLAP/VST3プラグイン用ノブ素材（オプション）

現在、DAWプラグインのUIはeguiの描画APIだけでノブを描いていますが、スタンドアロンアプリと見た目を完全に統一するために、スタンドアロンと同じスプライトシート画像をCLAP側にも埋め込めます。

※すでに `knobs/knob_64_frames.png` として作成済みであれば、それをそのままCLAPアセット用にコピーするだけでOKです。

| ファイル名 | サイズ | 用途 |
|----------|--------|------|
| `knob_64_frames.png` | 64×768 | 既存のアセットを流用。CLAPプラグインの `egui_knob` に置き換え。 |

---

## 配置先

```
crates/lightvc-app/assets/
├── ui_icons/           ← 新規作成
│   ├── icon_folder.png
│   ├── icon_play.png
│   ├── icon_stop.png
│   ├── icon_convert.png
│   ├── icon_trash.png
│   ├── icon_mic.png
│   └── icon_speaker.png
├── illustrations/      ← 新規作成
│   └── empty_stars.png
└── (既存のディレクトリ)
```

## 実装への影響（アセット到着後）

アセットが揃ったら、以下のUIアップデートを実装します：

1. **アイコン付きボタン**
   `info_card` 内の `Browse` や `Play` ボタンを、テキストのみから `[画像] テキスト` の形式に変更します。
2. **リストアイコン**
   `Offline` タブや `Catalog` タブの項目（Source, Reference, Voice Name）の先頭に `[画像]` を付けて視認性を上げます。
3. **空状態の表示**
   `if voices.is_empty()` の表示テキストの上に、イラスト画像を中央寄せで表示します。
4. **CLAPノブのリッチ化**
   CLAPプラグインのUIでも画像アセットを使用するように変更します。

---

## 実装状況（[06-4] にて確認・更新）

| アセット | 状態 | 備考 |
|---------|------|------|
| `icon_folder.png` | ✅ 使用中 | Browse ボタン、Catalog タブ |
| `icon_play.png` | ✅ 使用中 | Offline タブ、Catalog タブ |
| `icon_convert.png` | ✅ 使用中 | Offline タブ Convert ボタン |
| `icon_trash.png` | ✅ 使用中 | Catalog タブ Remove ボタン |
| `icon_mic.png` | ✅ 使用中 | Offline タブ Source ラベル |
| `icon_speaker.png` | ✅ 使用中 | Offline タブ Output ラベル |
| `icon_stop.png` | ✅ 使用中 | Realtime タブ Stop ボタン ([06-4] で対応) |
| `empty_stars.png` | ✅ 使用中 | Catalog タブ空状態 |
| `knob_64_frames.png` | ✅ 使用中 | スタンドアロンアプリのノブ |
| `knob_64.png` | ⚠️ 未使用 | スプライト版のみ使用中。削除候補 |
| `logo_header@2x.png` | ⚠️ 未使用 | Retina 対応未実装。1x 版のみ使用 |
| `bg_texture@2x.png` | ⚠️ 未使用 | Retina 対応未実装。1x 版のみ使用 |
| CLAP プラグインのノブ画像化 | ❌ 未実装 | `lightvc-clap` は純 egui 描画のまま（§3） |
