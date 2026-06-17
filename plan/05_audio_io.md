# 05: オーディオ I/O の乖離

> カテゴリ: E
> 関連資料: ARCHITECTURE.md §1, §5, RESEARCH.md §4

## 概要

リアルタイム性に直結するオーディオ I/O 周りに複数の問題。rubato が 0.16（非 RT-safe API）、スレッド分離が不十分、デバイス SR と 44.1k の境界処理が乱雑。リアルタイム VC としての信頼性を損なう。

## 現状の乖離

| 項目 | 設計 | 実装 |
|---|---|---|
| rubato | v3.0 `Async` + `process_into_buffer` | 解消（内部は zero-alloc）[05-1]。**ただし呼び出し側が毎回 Vec 確保** ⚠️ [05-5] |
| `DuplexStream::start_with` 引数 | `(device, device, sr, ch, buf)` | API 実装済み [05-2]。**ただし realtime_tab は `start_default()` のみ使用** ⚠️ [05-6] |
| スレッド分離 | capture / playback / inference / UI の 4 分離（§1.1） | AudioEngine でカプセル化 [05-3]。**ただし cpal ライフサイクルが依然 inference thread 内** ⚠️ [05-7] |
| リサンプリング境界 | device_sr ↔ 44.1k の明確な境界 | 3 段バッファリングで解消 [05-4]。**ただし capture/playback 異 SR に未対応** ⚠️ [05-8] |
| ARCH §5.2 の型名 | `AsyncFixedIn/Out` | **rubato 3.0 に存在しない虚構型名**（実体は `Async<T>` + `FixedAsync` enum）⚠️ [08-8] |

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
  - `needed = resampler.input_frames_needed_up()`（device_sr 揃りの必要フレーム数）
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

### [05-5] (P1) ✅ リサンプラ呼び出し側が毎回 Vec 確保（[05-1] 受け入れ基準未達）
- **現状**:
  - `realtime_tab.rs:464`: `let input_chunk: Vec<f32> = in_accum.drain(..needed).collect();`
  - `realtime_tab.rs:531`: 出力側も同様に `Vec` 確保
  - リサンプラ内部（`resample.rs`）は `process_into_buffer` + 事前確保 scratch buffer で zero-alloc だが、**呼び出し側が毎イテレーションで新規 Vec をヒープ確保**
  - [05-1] の受け入れ基準「ステディ状態でヒープアロケーションゼロ」は**realtime path 全体としては未達成**
- **影響**: RT スレッドでのヒープ確保が xrun・音割れの原因になり得る。内部だけ zero-alloc でも意味が薄い。
- **作業**:
  1. `realtime_tab.rs` に事前確保の作業バッファをフィールド追加（`in_chunk_buf`, `out_chunk_buf` 等）
  2. `drain(..needed).collect()` を `drain(..needed).for_each(|s| buf.push(s))` + クリア、またはリングから直接スライスコピー
  3. `tracing-allocator` 等でステディ状態の alloc ゼロを確認
- **受け入れ基準**: [05-1] の本来の受け入れ基準「ステディ状態でヒープアロケーションゼロ」を realtime path 全体で達成。
- **関連**: `crates/lightvc-app/src/realtime_tab.rs:464,531`, `crates/lightvc-audio/src/resample.rs:5-7`

### [05-6] (P1) ✅ `start_with` API が実装されたが呼び出し側が `start_default()` のみ使用
- **現状**:
  - `stream.rs:112-173`: `DuplexStream::start_with(input, output, capture_sr, playback_sr, in_ch, out_ch, buffer_size, ...)` 実装済み
  - `engine.rs:77-173`: `AudioEngine::start_with` も実装済み
  - **しかし** `realtime_tab.rs:346` は `AudioEngine::start_default()` のみ呼出
  - MANUAL §4.2 のオーディオデバイス UI はデバイス一覧を**表示するだけ**（クリックハンドラ無し）。SR / channels / buffer_size を選ぶ UI も無い
- **影響**:
  - ユーザーがデバイス・SR・バッファサイズを選べない（常に default）
  - ASIO / WASAPI exclusive 等の低遅延設定が使えない
  - `start_with` がデッドコード化
- **作業**:
  1. `realtime_tab.rs` のデバイス一覧にクリックハンドラ追加（入出力デバイス選択）
  2. SR / buffer_size のコンボボックス追加
  3. Start ボタンから `AudioEngine::start_with(選択値)` を呼出
- **受け入れ基準**: MANUAL §4.2 のデバイス列挙 UI から選択可能。任意の SR / buffer_size でストリーム開始できる。
- **関連**: `crates/lightvc-audio/src/stream.rs:112-173`, `crates/lightvc-audio/src/engine.rs:77-173`, `crates/lightvc-app/src/realtime_tab.rs:261-299,346`

### [05-7] (P1) ✅ AudioEngine のライフサイクルが依然 inference thread 内に結合
- **現状**:
  - `engine.rs` に `AudioEngine` 構造体は存在（cpal streams + ring buffers + fault flags をカプセル化）
  - しかし `realtime_tab.rs:338-407` の inference thread は `RtControl::Start` を受けて**スレッド内で** `AudioEngine::start_default()`（346行）を呼出
  - cpal Stream のライフサイクルと inference loop が依然密結合。[05-3] plan step 2/3「`inference_loop` は ring buffer との I/O のみ」「RealtimeTab の Start → AudioEngine::start() → inference thread 起動、と分離」が未達成
- **影響**:
  - デバイス切断時の再接続、モード変更時のスムーズな切り替えが困難
  - ARCHITECTURE §1.2 の channel 通信図と実装が不一致
- **作業**:
  1. `AudioEngine` を `RealtimeTab`（または `AudioEngineHandle`）が所有
  2. Start ボタン → `AudioEngine::start()`（UI thread 側）→ 別途 inference thread 起動
  3. inference thread は ring buffer のみ操作、cpal Stream を触らない
- **受け入れ基準**: スレッド構成が ARCHITECTURE.md §1.1 の図と一致。cpal lifecycle が inference から分離。
- **関連**: `crates/lightvc-audio/src/engine.rs`, `crates/lightvc-app/src/realtime_tab.rs:338-407`, `ARCHITECTURE.md:11-49`

### [05-8] (P1) ✅ capture と playback で SR が異なる場合の出力経路が未対応
- **現状**:
  - `engine.rs:49,80-81,169-170`: `capture_sample_rate` と `playback_sample_rate` を個別に追跡
  - しかし `realtime_tab.rs:348` は `eng.capture_sample_rate` のみ読む
  - `realtime_tab.rs:441`: `device_sr == 44_100` でリサンプル バイパス判定を**両方向**に適用
  - capture 44.1k + playback 48k（例: 44.1k マイク + 48k HDMI 出力）の場合、Stage 4（543-548行）が 44.1k サンプルをそのまま 48k playback ring へ → 速度/ピッチズレ
- **影響**: 非対称デバイス構成で出力ピッチがずれる。
- **作業**: リサンプル バイパス判定を `capture_sr == 44_100 && playback_sr == 44_100` に変更。playback 側に `process_down` を常に適用（playback_sr != 44_100 时）。
- **受け入れ基準**: capture 44.1k + playback 48k でピッチズレ無し。
- **関連**: `crates/lightvc-audio/src/engine.rs:49,80-81,169-170`, `crates/lightvc-app/src/realtime_tab.rs:348,441,543-548`

## 関連文書
- [02_streaming_lookahead.md](02_streaming_lookahead.md)
- [06_plugin_app.md](06_plugin_app.md)
- [08_known_bugs.md](08_known_bugs.md)
