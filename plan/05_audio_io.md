# 05: オーディオ I/O の乖離

> カテゴリ: E
> 関連資料: ARCHITECTURE.md §1, §5, RESEARCH.md §4

## 概要

リアルタイム性に直結するオーディオ I/O 周りに複数の問題。rubato が 0.16（非 RT-safe API）、スレッド分離が不十分、デバイス SR と 44.1k の境界処理が乱雑。リアルタイム VC としての信頼性を損なう。

## 現状の乖離

| 項目 | 設計 | 実装 |
|---|---|---|
| rubato | v3.0 `AsyncFixedIn/Out` + `process_into_buffer` | v0.16 `SincFixedIn/Out` + `process()`（Vec 確保） |
| `DuplexStream::start` 引数 | `(device, device, sample_rate, channels, buffer_size)` | `(input, output, capture_tx, playback_rx)` |
| スレッド分離 | capture / playback / inference / UI の 4 分離（§1.1） | `inference_loop` が `start_audio()` 兼務 |
| リサンプリング境界 | device_sr ↔ 44.1k の明確な境界 | `chunk_sz`(44.1k) と `input_frames_needed_up()`(device_sr) 混在 |

## タスクリスト

### [05-1] (P0) ✅ rubato を 3.0 に上げ、RT-safe API へ移行
- **現状**: `Cargo.toml:29` は `rubato = "0.16"`。`crates/lightvc-audio/src/resample.rs` は `process()` 呼び出しで毎回 `Vec::to_vec()` と `Vec` 返却。リアルタイムスレッドでのアロケーションは RT-safe でない。
- **影響**:
  - リアルタイムスレッドでヒープアロケーションが発生し、xrun / 音割れの原因
  - 設計資料（ARCHITECTURE §5.2, RESEARCH §4）が明示する v3.0 の `process_into_buffer`（ゼロアロケーション）と乖離
- **作業**:
  1. `Cargo.toml`: `rubato = "3.0"`（workspace dependencies 更新）
  2. `resample.rs` を `AsyncFixedIn<f32>` / `AsyncFixedOut<f32>` に置換
  3. `process_into_buffer()` を使い、入出力バッファは事前確保して再利用（`Resampler` 構造体にフィールドとして保持）
  4. API 変更に伴う呼び出し側（`realtime_tab.rs`, `cli.rs`）の更新
- **受け入れ基準**:
  - `cargo build --release` が通る
  - ステディ状態でヒープアロケーションゼロ（`tracing-allocator` 等のプロファイラで確認）
- **関連**: `Cargo.toml:29`, `crates/lightvc-audio/src/resample.rs`, `ARCHITECTURE.md:449-469`, `RESEARCH.md:142-145`

### [05-2] (P1) ✅ `DuplexStream::start` シグネチャ修正
- **現状**: `stream.rs:75-140` の `DuplexStream::start` 引数は ring buffer の producer/consumer。デバイス設定は `default_input_config()` / `default_output_config()` に固定で、ユーザーが SR / channels / buffer_size を選べない。
- **影響**:
  - MANUAL §4.2 のオーディオデバイス UI が実質機能していない（一覧表示のみ、選択しても SR 等は default 固定）
  - ASIO / WASAPI exclusive 等、低遅延設定が使えない
- **作業**:
  1. `start(input_device, output_device, sample_rate, channels, buffer_size)` に変更
  2. ring buffer は `DuplexStream` 内部で生成、`capture_rx` / `playback_tx` をフィールドまたは戻り値で返す
  3. `StreamConfig` を明示的に構築（`BufferSize::Fixed(n)` 等）
  4. realtime_tab の Start ボタンからユーザー選択デバイス・SR を渡す
- **受け入れ基準**: 任意の SR / channels / buffer_size でストリーム開始できる。MANUAL §4.2 のデバイス列挙 UI から選択可能。
- **関連**: `crates/lightvc-audio/src/stream.rs:75-140`, `ARCHITECTURE.md:407-447`, `crates/lightvc-app/src/realtime_tab.rs:245-284`

### [05-3] (P1) ✅ capture / playback / inference スレッドの完全分離
- **現状**: `realtime_tab.rs:291-477` の `inference_loop` は `start_audio()`（459-477 行）で cpal ストリームを立ち上げる。ARCHITECTURE.md §1.1（11-39 行）は capture callback / inference / playback callback / UI の 4 スレッド分離を想定。
- **影響**:
  - cpal Stream のライフサイクルと inference loop が密結合
  - デバイス切断時の再接続、モード変更時のスムーズな切り替えが困難
  - ARCHITECTURE §1.2 の channel 通信図と実装が不一致
- **作業**:
  1. `AudioEngine` 構造体を新設（`lightvc-audio` に配置）し、cpal ストリーム + ring buffer を管理
  2. `inference_loop` は ring buffer との I/O のみ担当
  3. `RealtimeTab` の Start ボタン → `AudioEngine::start()` → inference thread 起動、と分離
  4. cpal Stream は `AudioEngine` がドロップ時に自動停止
- **受け入れ基準**: スレッド構成が ARCHITECTURE.md §1.1 の図と一致する。
- **関連**: `crates/lightvc-app/src/realtime_tab.rs:291-477`, `ARCHITECTURE.md:11-49`

### [05-4] (P0) ✅ リサンプリング・チャンクサイズ境界の整理
- **現状**: `realtime_tab.rs:376-413` で以下の処理が混在:
  - `chunk_sz = pipeline.chunk_samples()`（44.1k 換算のサンプル数）
  - `needed = resampler.input_frames_needed_up()`（device_sr 揰りの必要フレーム数）
  - `cap.len() < needed` で `cap.resize(needed, 0.0)`（ゼロ埋め）
  - その後 `chunk.len() < chunk_sz` でさらにゼロ埋め、`chunk.len() > chunk_sz` で truncate
- **影響**:
  - リサンプラ内部状態がドリフトし、音歪み・ピッチシフトが発生する可能性
  - ゼロ埋めが頻発すると無音区間が混入
  - リアルタイム動作の信頼性を根本から損なう（[08-2] [08-7] と重複）
- **作業**:
  1. **入力側**: device_sr のフレームを `input_frames_needed_up()` だけ集める → `process_up()` → 44.1k PCM を得る
  2. **処理側**: 44.1k PCM を `samples_per_chunk()` 単位で process_chunk に渡す。端数は次チャンクに持ち越し（`remainder_buf` を導入）
  3. **出力側**: process_chunk の出力（44.1k）を蓄積 → `process_down()` で device_sr へ
  4. ゼロ埋めは最終手段（バッファアンダーフロー時のみ）
- **受け入れ基準**: 長時間実行（5 分以上）で位相・ピッチがドリフトしない。`sox` 等で入出力を突き合わせて可聴範囲のズレなし。
- **関連**: `crates/lightvc-app/src/realtime_tab.rs:376-456`

## 関連文書
- [02_streaming_lookahead.md](02_streaming_lookahead.md)
- [06_plugin_app.md](06_plugin_app.md)
- [08_known_bugs.md](08_known_bugs.md)
