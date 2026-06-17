# 06: プラグイン / アプリの乖離

> カテゴリ: F
> 関連資料: README.md, MANUAL.md §4-5, docs/ASSETS_SPEC_V2.md

## 概要

CLAP / VST3 プラグインとスタンドアロンアプリの UI / 挙動に、設計または実用品質を損なう問題がある。CLAP プラグインは現状 `default` config 固定で、実運用モデルの多くが読み込めない。

## 現状の乖離

| 項目 | 設計 / 正しい仕様 | 実装 |
|---|---|---|
| CLAP converter config | MANUAL §5.4「重みパスは永続化」 | explicit/sidecar/default フォールバック実装済み [06-1]。**ただし Browse UI 無し** ⚠️ [06-7] |
| `latency_ms` 計算 | 実レイテンシ（ARCHITECTURE §1.3） | realtime 側は解消 [06-2]。**CLAP 側は硬编码 `chunk_44k*3.0`・モード非依存・UI 非表示** ⚠️ [06-8] |
| RTF 計算 | 実時間 / 処理時間 | realtime ✓ / CLAP ✓（[06-2] 解消）|
| README ビルドコマンド | `cargo run -p lightvc-xtask -- bundle` | 解消 [06-3] |
| **CLAP リサンプリング** | host SR ↔ 44.1kHz | **lib.rs に Resampler が存在しない**（≠44.1k でピッチシフト）⚠️ [06-5] |
| **CLAP mode パラメータ** | Strict/Balanced/Quality 切替 | **デッドコード**（UI ボタンは動くが process が読まない）⚠️ [06-6] |
| **CLAP モデルロード UI** | Browse ボタン | **一切無し**（persist のみ）⚠️ [06-7] |
| **CLAP 推論スレットのゼロ埋め** | [08-7] で解消のはず | **realtime_tab のみ解消、lib.rs:553-559 は残存** ⚠️ [06-9] |
| **offline_tab の処理パス** | process_full（Python 完全一致）| **process_chunk でループ**（最終チャンクゼロ埋め）⚠️ [06-10] |

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

### [06-2] (P1) ✅ latency / RTF 計算の修正
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

### [06-3] (P2) ✅ README の `cargo xtask` 記載修正
- **現状**: README.md:35-46 の Quick Start は `cargo xtask bundle`。cargo-subcommand 版 xtask は未導入で機能しない。AGENTS.md / MANUAL.md:60-74 は `cargo run -p lightvc-xtask -- bundle`。
- **作業**: README.md のコマンド表記を `cargo run -p lightvc-xtask -- bundle` / `cargo run -p lightvc-xtask -- install` に統一。
- **受け入れ基準**: README のコマンドがそのまま実行できること。
- **関連**: `README.md:35-46`, `AGENTS.md` (Build Commands), `MANUAL.md:60-74`

### [06-4] (P2) ✅ MANUAL / ASSETS_SPEC の実装反映確認
- **現状**: `docs/ASSETS_SPEC_V2.md` が UI アイコン（`icon_folder.png`, `icon_play.png` 等）・空状態イラスト（`empty_stars.png`）・CLAP ノブアセットを要求している。実装反映状況が不明。`crates/lightvc-app/assets/` の中身と ASSETS_SPEC が食い違いないか確認が必要。
- **作業**:
  1. `crates/lightvc-app/assets/` の現状確認（Glob でファイル一覧取得）
  2. ASSETS_SPEC_V2.md / ASSETS_SPEC.md との突き合わせ
  3. 未作成アセットがある場合は ASSETS_SPEC を更新、またはプレースホルダ実装（egui の `Painter` のみで代替）
  4. MANUAL.md §4.2（99-143 行）の画面説明と実 UI を突き合わせ
- **受け入れ基準**: MANUAL の説明と実アプリの UI が一致。ASSETS_SPEC が実装状況を正しく反映。
- **関連**: `docs/ASSETS_SPEC_V2.md`, `docs/ASSETS_SPEC.md`, `crates/lightvc-app/assets/`, `docs/MANUAL.md:99-143`

### [06-5] (P0) ✅ CLAP プラグインが host SR ↔ 44.1kHz のリサンプリングを行わない
- **現状**: `crates/lightvc-clap/src/lib.rs` の `inference_thread`（537-575 行）は `crx` から読んだ host SR（例: 48kHz）のサンプルを**そのまま** `pipeline.process_chunk()`（lib.rs:564、44.1kHz PCM を期待）へ流す。`lib.rs` 内に `Resampler` のインスタンス化は**一切ない**（grep で 0 件）。realtime standalone アプリは `realtime_tab.rs:460-473,525-542` で正しくリサンプリングするが、CLAP 側のみ未対応。
- **影響**:
  - 44.1kHz 以外の DAW でピッチシフト（48kHz 音声を 44.1kHz として解釈 → ~8% 低く/遅く）
  - RTF 計算も破綻（`out.len()/44100` で割るが out は host SR 相当）
  - `set_latency_samples` も単位が混在
- **作業**:
  1. `lib.rs` に `Resampler` を導入（capture: host→44.1k, playback: 44.1k→host）
  2. realtime_tab.rs の 3 段バッファリング（[05-4]）と同等の構造を CLAP 側に実装
  3. host SR が 44.1kHz のときはバイパス
- **受け入れ基準**: 48kHz DAW でピッチシフト無く再生。RTF・レイテンシ表示が正確。
- **関連**: `crates/lightvc-clap/src/lib.rs:537-575`, `crates/lightvc-audio/src/resample.rs`, `crates/lightvc-app/src/realtime_tab.rs:460-473,525-542`

### [06-6] (P0) ✅ CLAP の `mode`（Strict/Balanced/Quality）パラメータがデッドコード
- **現状**:
  - `lib.rs:23-24`: `mode: IntParam` を定義
  - `lib.rs:346-366`: Strict/Balanced/Quality ボタンが `setter.set_parameter(&params.mode, val)` を呼ぶ
  - **しかし** `process()` も `inference_thread` も `self.params.mode.value()` を**一切読まない**
  - pipeline 構築（`lib.rs:704`）は `LatencyMode::Balanced` 硬编码
  - 対照的に standalone アプリは `realtime_tab.rs:380-394` で `RtControl::SetMode` を正しくディスパッチ
- **影響**: プラグイン UI でモードを切り替えても何も起きない（cosmetic のみ）。ユーザーに誤解を与える。
- **作業**:
  1. `inference_thread` で `params.mode.value()` を監視し、変更時に `VcPipeline` の chunk_mode を切替
  2. または CLAP host 経由で `process()` が mode 変更を検知し制御チャネルで通知
  3. モード切替時に streaming state を reset（`pipeline.reset()`）
- **受け入れ基準**: UI で Quality → Strict に切替るとレイテンシ・チャンクサイズが即座に変化する。
- **関連**: `crates/lightvc-clap/src/lib.rs:23-24,346-366,495-575,704`, `crates/lightvc-app/src/realtime_tab.rs:380-394`

### [06-7] (P0) ✅ CLAP エディタにモデルロード UI が無い（Browse ボタン不在）
- **現状**: `LightVcParams.model_path / dac_path / config_path` は `#[persist]` 付き `Arc<Mutex<String>>`（lib.rs:32-43）。`load_converter_config`（lib.rs:620-667）は explicit/sidecar/default フォールバックを実装済み。**しかしエディタ（lib.rs:184-398）にはファイル選択 UI が一切ない**。Browse ボタン・テキストフィールド・ファイルダイアログのいずれも不在。[06-1] の plan step 4「プラグインエディタに config browse ボタン追加」が未実施。
- **影響**: DAW ユーザーが初回起動時にモデルをロードする手段がない（persist に頼るのみで、未設定時は default config で空モデル相当になる）。
- **作業**:
  1. `rfd`（Rust File Dialog）または nice-plug-egui のファイルピッカーで Browse ボタン実装
  2. model_path / dac_path / config_path の 3 つに Browse + パス表示
  3. 変更時に pipeline 再構築トリガ
- **受け入れ基準**: DAW 内から 3 つの重み/config パスを設定し、VC が動作する。
- **関連**: `crates/lightvc-clap/src/lib.rs:32-43,184-398,620-667`

### [06-8] (P1) ✅ CLAP レイテンシ表示が硬编码・モード非依存・UI 非表示
- **現状**:
  - `lib.rs:424-427`: `latency_44k = chunk_44k * 3.0`（= 6144 sample ≈ 139ms）で Balanced 固定。mode や capture/playback buffer を含まない
  - `set_latency_samples` は `initialize()` の初回のみ（lib.rs:427）。mode 変更時の再 publish 無し（[06-6] と関連）
  - CLAP `Metrics` 構造体（lib.rs:87-97）に `latency_ms` フィールド**不在**。エディタは RTF のみ表示（lib.rs:317-321）
- **影響**: プラグイン UI でレイテンシが見えない。host への latency 報告も不正確（DAW の PDC が狂う）。
- **作業**:
  1. `Metrics` に `latency_ms` 追加、`algorithmic_latency_ms()` + cpal buffer 分を含む（realtime_tab.rs:518 と同等）
  2. mode 変更時に `set_latency_samples` を再 publish
  3. エディタに latency_ms 表示追加
- **受け入れ基準**: UI 表示が実測（strict ~60ms / balanced ~100ms / quality ~212ms）に近い。host への報告が mode に追従。
- **関連**: `crates/lightvc-clap/src/lib.rs:87-97,317-321,424-427`, `crates/lightvc-app/src/realtime_tab.rs:497-518`, `ARCHITECTURE.md:52-67`

### [06-9] (P1) ✅ CLAP 推論スレットのゼロ埋めが残存（[08-7] 部分解消）
- **現状**: `lib.rs:553-559`
  ```rust
  if cap.len() < needed.min(512) { sleep; continue; }
  if cap.len() < needed { cap.resize(needed, 0.0); }   // ゼロ埋め!
  ```
  [08-7] は realtime_tab.rs のみ解消済み。CLAP 側は依然ゼロ埋めで無音区間混入。
- **作業**: realtime_tab.rs の 3 段バッファリング（[05-4]）と同等の構造を CLAP 側へ移植。必要量溜まるまで待機、ゼロ埋め廃止。
- **受け入れ基準**: CLAP で長時間実行しても無音区間が混入しない。
- **関連**: `crates/lightvc-clap/src/lib.rs:553-559`, `crates/lightvc-app/src/realtime_tab.rs:328-548`

### [06-10] (P1) ✅ offline_tab が process_full ではなく process_chunk でループ
- **現状**: `offline_tab.rs:324-336`
  ```rust
  let chunk_size = p.chunk_samples();
  while i < src_padded.len() {
      let end = (i + chunk_size).min(src_padded.len());
      let mut chunk = src_padded[i..end].to_vec();
      if chunk.len() < chunk_size { chunk.resize(chunk_size, 0.0); }   // 最終チャンクゼロ埋め
      let out = p.process_chunk(&chunk)?;
      ...
  }
  ```
  `process_full`（pipeline.rs:185-194、encode→convert→decode 一括、wave_corr > 0.997 を謳う ARCHITECTURE §3.4.2）が存在するのに使わない。`cli.rs:217` は `--mode full` で process_full を呼ぶが GUI は呼ばない。
- **影響**: ファイル変換の品質が chunk 境界アーティファクトを含む可能性。最終チャンクのゼロ埋めで尾部に無音が入る。SOTA 品質パスが GUI から使えない。
- **作業**: offline_tab の変換ループを `process_full(&src_pcm)` 1 回呼出に変更。進捗表示は別途検討（フル処理は高速なので不要な可能性）。
- **受け入れ基準**: offline_tab の出力が cli `--mode full` と一致。尾部無音無し。
- **関連**: `crates/lightvc-app/src/offline_tab.rs:324-336`, `crates/lightvc-core/src/pipeline.rs:185-194`, `crates/lightvc-app/src/cli.rs:217`, `ARCHITECTURE.md:359-361`

## 関連文書
- [05_audio_io.md](05_audio_io.md)
- [07_unimplemented_phases.md](07_unimplemented_phases.md)
