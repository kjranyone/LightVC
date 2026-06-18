# LightVC GUI 設計書

> **ステータス**: 実装忠実（実コード `crates/lightvc-app/` に基づく）
> **対象バージョン**: egui 0.34 / eframe 0.34
> **最終更新**: 2026-06-18

---

## 1. デザイン概念

**"Kawaii Future Bass"** — ネオンパステル、グロー、きらめき、丸み。

LightVC の GUI は、プロ品質のリアルタイム音声変換ツールでありながら、親しみやすく華やかなビジュアル言語を採用する。ダークモードベースにパステルカラー（ピンク・ラベンダー・シアン・ミント）を載せ、アクティブ要素には柔らかいグロー効果を付与する。

### デザイン原則

| 原則 | 実装 |
|------|------|
| **丸み** | 角丸最低 8px、ボタンはピル型（`egui::Stroke` + rounding） |
| **グロー** | アクティブ要素に半透明ストローク（`PINK_BRIGHT` / `CYAN`） |
| **パーティクル** | 星形アイコン（`★`）、スパンキルCTAアイコン（`icon_convert.png`） |
| **ポップ体** | `.strong()` で太字、サイズ階層 10〜20px |
| **ホバー安定性** | `expansion = 0.0` で寸法不変、塗り色のみ変化（ちらつき防止） |

---

## 2. カラーシステム

### 2.1 パレット

全色 `theme.rs:16-33` の `colors` モジュールで定義。

| トークン | HEX | RGB | 役割 |
|---------|-----|-----|------|
| `BG_DARK` | `#1C1626` | `(28, 22, 38)` | 最暗背景・パネル塗り・スプラッシュ背景 |
| `BG_PANEL` | `#2A2038` | `(42, 32, 56)` | パネル背景・noninteractive fill |
| `BG_PANEL_LIGHT` | `#342844` | `(52, 40, 68)` | 非アクティブ要素 fill |
| `PINK` | `#FF82BE` | `(255, 130, 190)` | **主アクセント**・hovered fill・selection・CTA |
| `PINK_BRIGHT` | `#FFA0D2` | `(255, 160, 210)` | ハイライト・見出し・選択マーカー `★` |
| `LAVENDER` | `#AA8CFF` | `(170, 140, 255)` | 副アクセント・active fill・進行中表示 |
| `CYAN` | `#78E6FF` | `(120, 230, 255)` | グロー・ハイパーリンク・枠線ハイライト |
| `MINT` | `#82FFC8` | `(130, 255, 200)` | 正常状態・`LIVE` ステータス |
| `YELLOW` | `#FFDC82` | `(255, 220, 130)` | 警告・`BYPASS`・xrun 表示 |
| `TEXT` | `#F0EBFA` | `(240, 235, 250)` | 主要テキスト |
| `TEXT_DIM` | `#A096B4` | `(160, 150, 180)` | 補助テキスト・ラベル |
| `TEXT_MUTED` | `#6E6482` | `(110, 100, 130)` | 無効・最弱・パス表示 |

> **注意**: 赤色（エラー）のみ `colors` 定数ではなく、`app.rs:312` で `rgb(255, 100, 100)` としてインライン定義されている。トークン化を推奨。

### 2.2 ウィジェット状態マトリクス

`theme.rs:56-78`。**`expansion` は全状態で `0.0`**（ホバー時の寸法変化を排除）。

| 状態 | `bg_fill` | `fg_stroke` | `bg_stroke` | `expansion` |
|------|-----------|-------------|-------------|-------------|
| `noninteractive` | `BG_PANEL` | `1.0 × TEXT_DIM` | `1.0 × BG_PANEL_LIGHT` | `0.0` |
| `inactive` | `BG_PANEL_LIGHT` | `1.0 × TEXT` | `1.0 × LAVENDER` | `0.0` |
| `hovered` | `PINK` | `1.0 × TEXT` | `1.0 × PINK_BRIGHT` | `0.0` |
| `active` | `LAVENDER` | `1.0 × TEXT` | `1.0 × CYAN` | `0.0` |
| `open` | `LAVENDER` | `1.0 × TEXT` | — | `0.0` |

### 2.3 セレクション & ハイパーリンク

```rust
// theme.rs:81-85
selection.bg_fill    = PINK;
selection.stroke     = 1.0 × PINK_BRIGHT;
hyperlink_color      = CYAN;
```

---

## 3. タイポグラフィ

**カスタムフォント未使用** — egui デフォルトフォント。

### サイズ階層

| size | 用途 | 出現例 |
|------|------|--------|
| `20.0` | ページ見出し (`heading`) | 全タブタイトル |
| `18.0` | CTA 強調 | Convert ボタン (`offline_tab.rs:185`) |
| `16.0` | ステータス・モード名 | `● LIVE`, `Balanced` (`realtime_tab.rs:124,224`) |
| `14.0` | 強調テキスト・ボイス名 | タブラベル、ボイスカード名 |
| `13.0` | 通常テキスト | アイコンボタン、警告文 |
| `12.0` | 補助ラベル・db 表示 | `Latency: 46 ms \| RTF: 0.32` |
| `11.0` | xrun・ノブラベル | `xruns: 2 over / 0 under` |
| `10.0` | 最小詳細 | ノブ `0ms lookahead`、ボイスファイルパス |

### ウェイト

- `.strong()` = ボールド相当
- モノスペース: `FontId::proportional` または `.monospace()`（ノブラベル、db 表示、ボイス連番）

---

## 4. レイアウトシステム

### 4.1 ウィンドウ構成

```
┌──────────────────────────────────────────────┐
│  Top Bar (Panel::top)                        │  ← ロゴ + タブ [Offline][Realtime][Voices]
├──────────────────────────────────────────────┤
│                                              │
│  Central Panel (ScrollArea::vertical)        │  ← タブ別コンテンツ
│                                              │
│                                              │
├──────────────────────────────────────────────┤
│  Status Bar (Panel::bottom)                  │  ● dot + status/error message
└──────────────────────────────────────────────┘
```

- **デフォルトサイズ**: `800×600` (`cli.rs:271`)
- **最小サイズ**: 明示的制約なし（`ScrollArea` で内容スクロール）
- **背景**: `bg_texture.png` を全面描画（スプラッシュ終了後、毎フレーム）

### 4.2 パネル構造

| パネル | egui API | フレーム | 内容 |
|--------|---------|---------|------|
| Top Bar | `Panel::top("tabs")` | rgba(28,22,38,**220**) + margin 12 | ロゴ + タブボタン ×3 |
| Status Bar | `Panel::bottom("status")` | rgba(42,32,56,**180**) + margin 8 | status_dot + メッセージ |
| Central | `CentralPanel::default()` | デフォルト | `ScrollArea::vertical().auto_shrink([false,false])` で各タブをラップ |

> **egui 0.34 移行保留**: `CentralPanel::show` / `Panel::show` は deprecated だが、`show_inside()` への完全移行は UI ツリー再構築が必要なため保留（`app.rs:207-211`）。`render()` メソッド全体に `#[allow(deprecated)]` を付与。

### 4.3 スクロール

- **全タブ**: `ScrollArea::vertical().auto_shrink([false, false])` で縦スクロール可能
- **Catalog ボイスリスト**: 入れ子で `ScrollArea::vertical().max_height(400.0)`（`voice_catalog.rs:83-84`）

### 4.4 スペーシング

```rust
// theme.rs:43-45（グローバル）
style.spacing.item_spacing   = (8.0, 8.0);
style.spacing.button_padding = (16.0, 8.0);
```

セクション間隔は `ui.add_space(...)` で個別指定（4, 6, 8, 10, 12 を文脈で使い分け）。

### 4.5 レスポンシブ

| 要素 | 戦略 | 実装 |
|------|------|------|
| ロゴ | 利用可能幅 80〜140px にスケール | `app.rs:278` |
| Splash | 画面高さ × 0.25 の上余白 | `app.rs:233` |
| 背景テクスチャ | `content_rect()` 全体にフィル | `app.rs:256-262` |
| レベルメーター | `available_width() - 70.0`（最低 40px） | `theme.rs:200` |
| Offline フォーム | max 520px, min 300px | `offline_tab.rs:48-49` |

---

## 5. ウィジェットカタログ

### 5.1 `heading(text)` — `theme.rs:124`

ページ見出し。size 20, strong, `PINK_BRIGHT`。上下に `add_space(4.0)` / `add_space(2.0)`。

### 5.2 `pill_button(text, active) -> bool` — `theme.rs:136`

メインアクションボタン。

- min_size: `(80.0, 32.0)`
- text: size 14, strong, `TEXT`
- 色: active=`LAVENDER/CYAN`, 非active=`BG_PANEL_LIGHT/PINK`
- 使用: Load Converter, Browse, Start/Stop, Bypass, Mode切替, Add, Use

### 5.3 `icon_button(icon, text, active) -> bool` — `theme.rs:93`

アイコン + テキストのボタン。

- アイコン: 16×16
- min_size: `(80.0, 30.0)`
- text: size 13, strong, `TEXT`
- 色: pill_button と同一
- 使用: Browse, Play, Stop, Remove, Export, Import

### 5.4 `tab_button(text, selected) -> bool` — `theme.rs:157`

タブ切替ボタン。

- min_size: `(70.0, 30.0)`
- text: size 14, strong
- 色: selected=`PINK/TEXT/PINK_BRIGHT`, 非selected=`BG_PANEL/TEXT_DIM/BG_PANEL_LIGHT`

### 5.5 `status_dot(active, color)` — `theme.rs:173`

状態インジケータ。

- サイズ: 16×16 領域
- 構造: active時は半径 9.0 の半透明グロー（alpha 50）+ 半径 5.0 のメイン円
- 非active: 半径 5.0 の `TEXT_MUTED` 円のみ

### 5.6 `level_meter(level, label)` — `theme.rs:192`

水平レベルメーター（RMS）。

- 構成: `[ラベル 70px] [バー fillable] [db表示]`
- バー寸法: `(avail_width - 70).max(40)` × 16, rounding 8.0
- level 倍率: `level * 10.0` を clamp [0,1]
- 色分け（level×10 換算）:
  - `>0.85`: `PINK`（クリップ領域）
  - `>0.65`: `YELLOW`（高音量）
  - `>0.4`: `CYAN`（中音量）
  - その他: `MINT`（正常）
- 右端グロー: alpha 60 の半透明帯（幅 12px）
- db 表示: `20 * log10(level)`, ゼロ時 `-99.0`, size 11 monospace

### 5.7 `info_card(add_contents)` — `theme.rs:253`

セクション区切りカード。

- Frame::NONE ベース
- fill: `BG_PANEL`
- stroke: `1.5 × LAVENDER`
- inner_margin: 16

### 5.8 `knob(knob_tex, id, value, label) -> Option<f32>` — `theme.rs:275`

カスタムスプライトシートノブ。

- スプライト: `knob_64_frames.png`（64×768, 12フレーム縦スタック）
  - Frame 0 = 最小（7時）, Frame 11 = 最大（5時）
- 確保領域: `(64.0, 84.0)`（ノブ + ラベルスペース）
- ドラッグ: Y軸, 感度 `0.005`（200px = フルレンジ）, 反転
- ダブルクリック: 中央値 `0.5` にリセット
- tint: dragged=`rgba(255,200,240,255)`, hovered=`rgba(255,180,220,255)`, 通常=`WHITE`
- グローリング: dragged時のみ `2.0 × PINK`, radius `30.7`
- ラベル: ノブ下 6px, `FontId::proportional(11.0)`, `TEXT_DIM`

---

## 6. スクリーン仕様

### 6.1 スプラッシュ画面

- **表示期間**: 初回 30フレーム（約 0.5s @ 60fps）
- **フェード**: frame 0-19 = 不透明, frame 20-30 = リニアフェードアウト
- **内容**: `splash.png` (300×150) を中央寄せ + `ui.spinner()`
- **背景**: `BG_DARK` 完全不透明で全面塗りつぶし

### 6.2 Offline タブ

```
┌─ Offline Voice Conversion ─────────────────────┐
│                                                 │
│  ┌─ Source ─────────────────────────────────┐  │
│  │ 🎤 [source.wav_________] [▶ Play][Browse]│  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌─ Reference ──────────────────────────────┐  │
│  │ 🔊 [target.wav_________] [▶ Play][Browse]│  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  Or pick from Voice Catalog:                    │
│  [Voice A] [Voice B] [Voice C] ...              │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │          ✦  Convert   (CTA, 160×42)      │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌─ Output ─────────────────────────────────┐  │  ← 変換後のみ表示
│  │ Output  44100 samples (1.0s)              │  │
│  │ [▶ Play Output] [💾 Save As...]           │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**要素**:
- Source / Reference カード: `info_card` + mic/speaker アイコン + TextEdit + Play/Stop 切替 + Browse
- Catalog quick-pick: voices 非空時のみ表示
- Convert CTA: `icon_convert.png` (18×18) + size 18 text, fill=`PINK`, min `(160, 42)`
- 変換中: `ui.spinner()` + `"Converting..."`
- Output カード: 変換完了後に出現、Play/Stop 切替 + Save As...
- converter 未ロード時: 黄色警告 `"No converter loaded. Set model in Realtime tab."`
- **再変換時のクリア**: Convert ボタンクリック時に `converted_samples` / `offline_result` / `player` をクリアし、2回目以降の変換でも古い結果が残らない仕様

### 6.3 Realtime タブ

#### Converter 未ロード時（force_bypass）

```
┌─ Real-time Voice Conversion ───────────────────┐
│                                                 │
│  ┌─ Load Model ─────────────────────────────┐  │
│  │ Converter [path_________] [Browse]        │  │
│  │ Config    [path_________] [Browse]        │  │
│  │ [Load Converter]                          │  │
│  │ ⚠ No converter — Start will run in BYPASS │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ○ STOPPED  |  Reference: none (select in Cat)  │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │ Input  [████████████░░░░░░] -12.3 dB      │  │
│  │ Output [████████░░░░░░░░░░] -18.5 dB      │  │
│  │ Latency: 26 ms  |  RTF: 0.00              │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  BYPASS ON (no converter)                       │  ← 強制BYPASS表示
│                                                 │
│  [▶ Start (bypass)]                             │
│                                                 │
│  ▼ Audio Devices                                │  ← default_open
│    Inputs  (none = default)                     │
│    ○ (default)                                  │
│    ○ Microphone (48000Hz, 1ch)                  │
│    Outputs (none = default)                     │
│    ○ (default)                                  │
│    ○ Speakers (48000Hz, 2ch)                    │
└─────────────────────────────────────────────────┘
```

#### Converter ロード時

```
┌─ Real-time Voice Conversion ───────────────────┐
│                                                 │
│  ● LIVE  |  ★ Reference: Voice A               │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │ Input  [████████████████░░] -10.1 dB      │  │
│  │ Output [██████████░░░░░░░░] -15.7 dB      │  │
│  │ Latency: 72 ms  |  RTF: 0.34              │  │
│  │ xruns: 0 over / 2 under                   │  │  ← xrun>0時のみ
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌──────┐  Mode                                 │
│  │  ◐   │  Balanced                            │  ← knob (Quality モード選択)
│  └──────┘  ~46ms lookahead                      │
│                                                 │
│  [Bypass]                                       │
│  [■ Stop]                                       │
│  ▼ Audio Devices ...                            │
└─────────────────────────────────────────────────┘
```

**状態別表示**:
| 状態 | Load Model | Mode knob | Bypass | Start button | 参照音声 |
|------|-----------|-----------|--------|-------------|---------|
| 未ロード | 表示 | 非表示 | 強制ON `"BYPASS ON (no converter)"` | `"▶ Start (bypass)"` | 非表示 |
| ロード済 | 非表示 | 表示 | トグル可能 | `"▶ Start"` / `"■ Stop"` | 表示 |

### 6.4 Catalog タブ

```
┌─ Voice Catalog ─────────────────────────────────┐
│  Register reference audio for zero-shot VC       │
│                                                  │
│  ┌─ Add Voice ───────────────────────────────┐  │
│  │ Name [_____________] [Browse] [Add]        │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  3 voices                                        │
│  ┌────────────────────────────────────────────┐ │
│  │ ★ Voice A                  [Use][Play][🗑]  │ │  ← 選択中（★、PINK枠）
│  │   /path/to/voiceA.wav                       │ │
│  └────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────┐ │
│  │ 2  Voice B                 [Use][Play][🗑]  │ │  ← 非選択
│  │   /path/to/voiceB.wav                       │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ▶ Import / Export                               │  ← 折たたみ（初期閉じ）
└──────────────────────────────────────────────────┘
```

**空状態**: `empty_stars.png` (120×120, `LAVENDER` 半透明 tint) + `"No voices registered yet."`

**選択連携**: `Use` ボタン → `selected_voice = Some(i)` + `RtControl::LoadReference(wav44)` 送信 → Realtime タブに即時反映

---

## 7. 状態管理

### 7.1 状態の分類

| スコープ | 保持場所 | 例 |
|---------|---------|-----|
| **アプリ共有** | `Arc<Mutex<AppState>>` | pipeline, voices, converter path, status |
| **UI ローカル** | `LightVcApp` フィールド | current_tab, rt_running, rt_metrics, asset_cache |
| **タブ別 UI** | `OfflineState` | source_path, converting, converted_samples |
| **推論スレッド** | `inference_loop` ローカル | resamplers, accumulators, engine |
| **egui 一時** | `Context::data_mut()` | 入力バッファ、pick_target フラグ |

### 7.2 `AppState`（アプリ共有）

```rust
struct AppState {
    dac_weights: PathBuf,                    // 起動時固定
    converter_weights: Option<PathBuf>,
    converter_config: Option<PathBuf>,
    pipeline: Option<Arc<Mutex<VcPipeline>>>,
    pipeline_slot: Arc<Mutex<Option<Arc<Mutex<VcPipeline>>>>>,
    // ↑ 推論スレッドと共有するホットスワップ可能なスロット。
    //   スレッド起動後に converter をロードしても即座に反映される。
    voices: Vec<VoiceEntry>,
    selected_voice: Option<usize>,           // Catalog → Realtime 連携
    error: Option<String>,                   // Status bar 赤
    status: String,                          // Status bar 緑
    offline_result: Option<Vec<f32>>,        // 別スレッドから書込
    rt_control_tx: Option<Sender<RtControl>>,
    rt_metrics_rx: Option<Receiver<RtMetrics>>,
    rt_initialized: bool,
}
```

### 7.3 UI → 推論スレッド通信

```rust
enum RtControl {
    StartWithDevices { input_idx: Option<usize>, output_idx: Option<usize> },
    Stop,
    SetMode(LatencyMode),     // Strict / Balanced / Quality
    Bypass(bool),
    LoadReference(Vec<f32>),  // 44.1kHz mono PCM
}
```

### 7.4 推論スレッド → UI 通信

```rust
struct RtMetrics {
    input_rms: f32,
    output_rms: f32,
    latency_ms: f32,
    rtf: f32,
    disconnected: bool,    // デバイス切断（UI は停止して再選択へ）
    overrun: u64,          // capture overrun 数
    underrun: u64,         // playback underrun 数
    current_mode: LatencyMode,  // 現在の有効モード（自動劣化反映済み）
    auto_degraded: bool,        // 自動劣化発生中フラグ
}
```

### 7.5 egui temporary/persistent data

| キー | 種別 | 用途 |
|------|------|------|
| `"rt_pick"` | temp | `"converter"` or `"config"` 選別 |
| `"rt_mode_knob"` | persistent Id | knob 状態保持 |
| `"catalog_new_name"` | temp | ボイス名入力バッファ |
| `"catalog_pick"` | temp | Browse トリガフラグ |
| `"catalog_picked"` | temp | 選択パス |
| `"catalog_import"` | temp | Import トリガフラグ |

---

## 8. インタラクション仕様

### 8.1 ホバー

- **ボタン系**: `expansion=0.0` で寸法不変、fill色のみ `BG_PANEL_LIGHT → PINK` に変化
- **ノブ**: tint が `WHITE → rgba(255,180,220,255)` に変化、グローなし

### 8.2 ドラッグ

- **ノブ**: Y軸ドラッグ（上=増加）、感度 `0.005`、ダブルクリックで中央値リセット
- **スクロール**: `ScrollArea::vertical` でホイール/タッチスクロール

### 8.3 選択状態

| 対象 | 視覚表現 |
|------|---------|
| ボイスカード | 背景 `rgba(90,60,120,60)`, 枠線 `2.0×PINK`, インデックス `★` (`PINK_BRIGHT`) |
| 参照音声（Realtime） | `"★ Reference: {name}"` (`PINK_BRIGHT`) |
| タブ | fill=`PINK`, stroke=`PINK_BRIGHT` |

### 8.4 プレビュー再生

全 Play ボタンは再生中 `"■ Stop"` に切替。`AudioPlayer` を各タブの状態フィールドに保持し、Stop クリックで `stop()` 呼出：
- **Offline**: `OfflineState.player` / `.source_preview` / `.reference_preview`
- **Catalog**: `CatalogState.player`（単一、最後に再生した音声）
- **Realtime**: Stop ボタンは変換停止（プレビュー再生ではない）

### 8.5 ファイルダイアログ

| ライブラリ | 用途 |
|-----------|------|
| `rfd::FileDialog`（バックグラウンドスレッド） | Browse / Import / Save 系すべて |

egui-file-dialog は egui 0.31 を引き込んで egui 0.34 と二重化する問題があったため廃止。代わりに `file_pick::FilePick`（`Arc<Mutex<Option<PathBuf>>>` + バックグラウンド `rfd::FileDialog`）を導入。UI スレッドをブロックせず、毎フレーム `take()` で結果をポーリングする。各 picker はタブ/用途ごとに独立インスタンス（Offline: source/reference, Realtime: converter/config, Catalog: add/import）。

---

## 9. アセット管理

### 9.1 コンパイル時埋め込み

全アセットは `include_bytes!` でバイナリ埋め込み（`assets.rs:5-23`）。実行時のファイル I/O 不要。

### 9.2 遅延 TextureHandle

`AssetCache` は全フィールド `Option<TextureHandle>` で `None` 初期化。初回アクセス時に `get_or_insert_with` で GPU テクスチャ化。

### 9.3 アセット → ウィジェット対応表

| アセット | 使用箇所 |
|---------|---------|
| `splash.png` | Splash CentralPanel |
| `bg_texture.png` | 背景レイヤー |
| `logo_header.png` | Top Bar（レスポンシブ 80-140px） |
| `knob_64_frames.png` | `theme::knob()` スプライトシート |
| `icon_folder.png` | Browse ボタン（全タブ） |
| `icon_play.png` | Play ボタン |
| `icon_stop.png` | Stop ボタン（Realtime） |
| `icon_convert.png` | Convert CTA |
| `icon_trash.png` | Remove ボタン（Catalog） |
| `icon_mic.png` | Source ラベル |
| `icon_speaker.png` | Reference/Output ラベル、Save As... |
| `empty_stars.png` | Catalog 空状態 |

### 9.4 HiDPI / Retina

`logo_header@2x.png`, `bg_texture@2x.png` はアセットとして存在するが未使用。egui の `pixels_per_point` 連携による HiDPI 対応は今後課題（`ASSETS_SPEC_V2.md` に明記）。

---

## 10. フレーム描画制御

- **Realtime タブ**: `ctx.request_repaint()` で連続再描画（メーター・メトリクス更新用）
- **他タブ**: イベント駆動（クリック、ホバー、ファイルダイアログ終了時のみ再描画）
- **Splash 中**: フェードアニメーション用に `request_repaint()`

---

## 11. 推論スレッド設計

### 11.1 スレッド分離

```
UI Thread                    Inference Thread
─────────                    ─────────────────
render()  ──control_tx──→   inference_loop()
           ←─metrics_rx──   (capture → resample → convert → resample → playback)
```

- `crossbeam_channel` で `RtControl` / `RtMetrics` を双方向通信
- converter 未ロードでもスレッド起動（bypass モードでオーディオパステスト可能）

### 11.2 3ステージバッファリング

```
capture (device_sr) → [in_accum] → resample_up →
[pcm_44k_accum]     → process_chunk (converter) →
[out_44k_accum]     → resample_down → playback (device_sr)
```

4つのサンプルフレーム領域を分離し、キャプチャ/プレイバックのサンプルレート独立性を保証。

### 11.3 レイテンシ見積式

```
latency_ms = 10 (capture buffer) + 3 (resample up)
           + algorithmic (chunk + FRC lookahead)
           + 3 (resample down) + 10 (playback buffer)
```

Bypass 時は algorithmic 項をスキップ。

### 11.4 xrun ハンドリング

- **Overrun**: capture ring 満杯時、最古サンプルを破棄（"drop oldest" ポリシー）
- **Underrun**: playback ring 空時、サイレンス出力
- **自動劣化**: underrun 10連続で Quality→Balanced→Strict に自動ダウングレード
- **UI 同期**: 自動劣化発生時、`RtMetrics.current_mode` / `.auto_degraded` を UI に送信。ノブ表示と `rt_mode` が即座に更新され、`"⚠ Auto-degraded to Strict (underruns)"` 警告が表示される

---

## 12. 既知の制約・今後の課題

| 項目 | 現状 | 計画 |
|------|------|------|
| egui `show()` deprecated | `#[allow(deprecated)]` で保留 | `show_inside()` へ完全移行（UIツリー再構築） |
| HiDPI / Retina | `@2x` アセット未使用 | `pixels_per_point` 連携 |
| カスタムフォント | egui デフォルト | ポップで可愛い丸ゴシック体の導入検討 |
| エラー色トークン化 | `rgb(255,100,100)` イライン | `colors::ERROR` 定数化 |
| MANUAL の数値ズレ | `~40ms/~80ms` vs 実装 `~46ms/~93ms` | 実装値に合わせて修正 |
| CLAP プラグインのノブ画像化 | egui 描画のみ | スタンドアロンと統一 |

---

## 付録 A: ファイル構成

```
crates/lightvc-app/src/
├── main.rs              # エントリ、Windows exit ハック
├── cli.rs               # clap dispatch (roundtrip/convert/gui), ウィンドウ設定
├── app.rs               # LightVcApp, AppState, render(), スレッド起動
├── theme.rs             # Kawaii Future Bass テーマ、カスタムウィジェット
├── assets.rs            # include_bytes!, AssetCache (遅延 TextureHandle)
├── audio_playback.rs    # WAV 再生/保存/リサンプリング
├── offline_tab.rs       # タブ1: Offline 変換 + OfflineState
├── realtime_tab.rs      # タブ2: Realtime 変換 + inference_loop
├── voice_catalog.rs     # タブ3: Voice Catalog
└── widgets.rs           # rms() ヘルパ
```

## 付録 B: 依存クレート

| クレート | バージョン | 用途 |
|---------|-----------|------|
| `eframe` | 0.34 | egui フレームワーク |
| `egui-file-dialog` | 0.10 | ファイル選択ダイアログ |
| `rfd` | 0.15 | ネイティブ Save ダイアログ |
| `crossbeam-channel` | — | 推論スレッド通信 |
| `rtrb` | — | SPSC オーディオリングバッファ |
| `cpal` | 0.18 | オーディオ I/O |
| `image` | 0.25 | PNG デコード |
| `hound` | — | WAV I/O |
| `clap` | 4.5 | CLI パーサー |
