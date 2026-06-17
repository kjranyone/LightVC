# 06: プラグイン / アプリの乖離

> カテゴリ: F
> 関連資料: README.md, MANUAL.md §4-5, docs/ASSETS_SPEC_V2.md

## 概要

CLAP / VST3 プラグインとスタンドアロンアプリの UI / 挙動に、設計または実用品質を損なう問題がある。CLAP プラグインは現状 `default` config 固定で、実運用モデルの多くが読み込めない。

## 現状の乖離

| 項目 | 設計 / 正しい仕様 | 実装 |
|---|---|---|
| CLAP converter config | MANUAL §5.4「重みパスは永続化」 | `ConverterConfig::default()` 固定、config 読み込みなし |
| `latency_ms` 計算 | 実レイテンシ（ARCHITECTURE §1.3） | `chunk の再生時間` を latency と命名（`realtime_tab.rs:444`） |
| RTF 計算 | 実時間 / 処理時間 | realtime 側は OK、CLAP 側は `rtf: 0.0 // TODO`（未測定） |
| README ビルドコマンド | `cargo run -p lightvc-xtask -- bundle` | `cargo xtask bundle` と記載（機能しない） |

## タスクリスト

### [06-1] (P0) ✅ CLAP プラグインが converter config を読み込むように
- **現状**: `lightvc-clap/src/lib.rs:575` は `ConverterConfig::default()`。`hidden_dim=1024`, `model_type="converter"` 固定のため、以下のモデルが読み込めない:
  - **FlowConverter**（`model_type: "flow"`）— Phase C 本命モデル
  - `hidden_dim=256` の現行学習モデル（`phase_b.yaml` / `phase_c.yaml`）
  - `enable_timbre=true` の UTTE モデル（Phase 2）
- **影響**: DAW ユーザーが現在配布されるモデルのほとんどを読み込めない。実用上の重大問題。
- **作業**:
  1. `LightVcParams` に `config_path: Arc<Mutex<String>>` を追加（`#[persist = "config-path"]`）
  2. `load_pipeline`（563-585 行）で JSON から `ConverterConfig` を読み込むよう変更
  3. config ファイルが未指定の場合は重みファイルと同名の `_config.json` を探すフォールバック（`export_weights.py:61-78` が生成）
  4. プラグインエディタに config browse ボタン追加
- **受け入れ基準**: FlowConverter / `hidden_dim=256` / UTTE あり、全パターンの重みが読み込めること。
- **関連**: `crates/lightvc-clap/src/lib.rs:17-39, 563-585`, `training/export_weights.py:61-78`, `MANUAL.md:194-197`

### [06-2] (P1) latency / RTF 計算の修正
- **現状**:
  - `realtime_tab.rs:444`: `latency_ms = out_dev.len() / device_sr * 1000` は単なるチャンク再生時間。実際の end-to-end latency は capture buf + resample + encode + convert + decode + resample + playback buf。
  - `lightvc-clap/src/lib.rs:549`: `rtf: 0.0 // TODO: measure`
- **影響**:
  - UI 表示の latency が意味をなさない（ユーザーが「80ms と書いてあるが実測 200ms」等で混乱）
  - RTF が見えないと CPU 負荷の判断ができない
- **作業**:
  1. **realtime 側**: capture 時の timestamp を ring buffer に乗せる、または capture フレーム数と playback フレーム数の差から遅延を算出
  2. **CLAP 側**: `inference_thread` 内で `process_chunk` 前後の `Instant::now()` 差から RTF 計算
  3. 目標値との比較表示（Strict=~60ms / Balanced=~100ms / Quality=~140ms、RTF < 1.0）
- **受け入れ基準**: UI 表示の latency が実測（例: strict 60ms / quality 140ms）に近いこと。RTF が 1.0 未満であること。
- **関連**: `crates/lightvc-app/src/realtime_tab.rs:429-455`, `crates/lightvc-clap/src/lib.rs:541-552`, `ARCHITECTURE.md:52-67`

### [06-3] (P2) README の `cargo xtask` 記載修正
- **現状**: README.md:35-46 の Quick Start は `cargo xtask bundle`。cargo-subcommand 版 xtask は未導入で機能しない。AGENTS.md / MANUAL.md:60-74 は `cargo run -p lightvc-xtask -- bundle`。
- **作業**: README.md のコマンド表記を `cargo run -p lightvc-xtask -- bundle` / `cargo run -p lightvc-xtask -- install` に統一。
- **受け入れ基準**: README のコマンドがそのまま実行できること。
- **関連**: `README.md:35-46`, `AGENTS.md` (Build Commands), `MANUAL.md:60-74`

### [06-4] (P2) MANUAL / ASSETS_SPEC の実装反映確認
- **現状**: `docs/ASSETS_SPEC_V2.md` が UI アイコン（`icon_folder.png`, `icon_play.png` 等）・空状態イラスト（`empty_stars.png`）・CLAP ノブアセットを要求している。実装反映状況が不明。`crates/lightvc-app/assets/` の中身と ASSETS_SPEC が食い違いないか確認が必要。
- **作業**:
  1. `crates/lightvc-app/assets/` の現状確認（Glob でファイル一覧取得）
  2. ASSETS_SPEC_V2.md / ASSETS_SPEC.md との突き合わせ
  3. 未作成アセットがある場合は ASSETS_SPEC を更新、またはプレースホルダ実装（egui の `Painter` のみで代替）
  4. MANUAL.md §4.2（99-143 行）の画面説明と実 UI を突き合わせ
- **受け入れ基準**: MANUAL の説明と実アプリの UI が一致。ASSETS_SPEC が実装状況を正しく反映。
- **関連**: `docs/ASSETS_SPEC_V2.md`, `docs/ASSETS_SPEC.md`, `crates/lightvc-app/assets/`, `docs/MANUAL.md:99-143`

## 関連文書
- [05_audio_io.md](05_audio_io.md)
- [07_unimplemented_phases.md](07_unimplemented_phases.md)
