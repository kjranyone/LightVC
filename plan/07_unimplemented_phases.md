# 07: 未実装の設計機能

> カテゴリ: G
> 関連資料: CONCEPT.md（全般）, DESIGN.md §3, ARCHITECTURE.md §4.3-4.4, §8

## 概要

CONCEPT.md / DESIGN.md で構想されているが、Rust / Python とも実装されていない機能群。研究的新規性（Phase 3）やプロダクト価値（Phase 4）に直結するが、Phase 1-2 の安定化（[02] [03] [04]）が先決。

## 現状の乖離

| 機能 | 設計 | 実装 |
|---|---|---|
| Phase 3: Progressive RVQ-depth factorized FM heads | CONCEPT「新規性の核心」, DESIGN §3, MODEL_TRAINING 表 | Rust 側 Quantizer 実装済み（[07-1] 🚧）。Python 側 factorized heads は学習待ち |
| Phase 4: Prosody/Rhythm factorization | ARCHITECTURE §4.4, CONCEPT SOTAネタ5 | `prosody_mode` enum すらない（[07-2] ⬜）|
| dual-path converter | CONCEPT「さらに攻めるなら」 | なし（[07-3] ⬜）|
| エッジケース処理 | ARCHITECTURE §8（6 項目） | 3/6 完全・3/6 不完全（[07-4] ✅ としたが [07-5] で修正）|

## タスクリスト

### [07-1] (P✅) Phase 3: Progressive RVQ-depth factorized FM heads
- **現状**: 実装なし。現在のパイプラインは continuous latent のみで、quantizer を通らない（ARCHITECTURE §3.2）。Phase 3 には DAC quantizer encode path の実装（L2 距離 + argmin + residual subtraction, ~100 LOC）が前提。
- **位置づけ**: CONCEPT.md（339-376 行）で「研究としての一番強い新規性」と明記。Astrape 超えの核心的差別化要素。
- **作業**:
  1. **Rust 側**に DAC Quantizer 実装（nearest-neighbor codebook lookup）
     - `dac_model.rs` に `Quantizer` 構造体追加
     - 入力: continuous latent `[B, 1024, T]` → 出力: codes `[B, 9, T]`、quantized latent `[B, 1024, T]`
  2. **学習側**に RVQ depth ごとの factorized flow head を FlowConverter に追加
     - 現在の単一 `vel_proj` を depth 別（9 codebook × factorized heads）に拡張
  3. coarse layer（content / timbre）/ mid layer（spectral）/ fine layer（texture）の変換強度を独立制御
  4. 低遅延モード（layer 1-3 のみ変換、4-9 passthrough）実装
  5. privacy モード（timbre-bearing layers 強変換）実装
- **受け入れ基準**:
  - RVQ depth 制御で音質 / 遅延 / プライバシーが独立に変化する
  - CONCEPT.md の「低遅延モード / 高品質モード / privacy モード / natural モード」が動作
  - 論文ネタとして測定可能（depth 別 SECS / UTMOS / WER テーブル作成）
- **前提**: [02-1] [02-2] [03-1] [04-5] 完了が必須
- **関連**: `CONCEPT.md:196-238, 339-376`, `DESIGN.md:56-59`, `ARCHITECTURE.md:367-383`, `MODEL_TRAINING.md:460-474`

### [07-2] (P2) Phase 4: Prosody/Rhythm factorization
- **現状**: `prosody_mode` enum（PreserveSource / Blend / ImitateTarget / FlattenPrivacy）は Rust / Python とも未実装、UI にもノブなし。
- **位置づけ**: プロダクト価値の向上（CONCEPT.md「単に似せるVCよりプロダクト価値が高い」）。Discl-VC / R-VC 系の要素。
- **作業**:
  1. `ProsodyMode` enum を Rust `converter.rs` に追加
  2. content path / prosody path / rhythm path の分離実装
     - content: 低帯域 latent（linguistic）
     - prosody: latent residual（F0 / energy）
     - rhythm: frame energy envelope（duration pattern）
  3. UI に prosody mode ノブ追加（`realtime_tab.rs` + CLAP プラグイン）
  4. Python 側 `converter.py` に prosody token 予測ヘッド追加
- **受け入れ基準**: PreserveSource モードで韻律が source 保持、ImitateTarget で target 響律に置換されること。ABX で確認。
- **関連**: `CONCEPT.md:150-194`, `ARCHITECTURE.md:385-399`, `RESEARCH.md:57-65` (Discl-VC, R-VC)

### [07-3] (P2) dual-path converter
- **現状**: なし。CONCEPT.md（319-334 行）の「さらに攻めるなら」で言及のみ。
- **作業**: Phase 3 安定後、fast path（coarse 変換）+ refine path（detail 補正）の 2 パス構成を検討。
- **受け入れ基準**: 単パス比で低遅延かつ高品質を両立すること。
- **関連**: `CONCEPT.md:319-334`

### [07-4] (P1) ✅ ARCHITECTURE §8 エッジケースの実装（3/6 完全・3/6 は [07-5] で修正）
- **現状**: ARCHITECTURE.md §8（613-622 行）が列挙する 6 シナリオのうち実装状況:
  - ✅ **サイレンス検出**: `pipeline.process_chunk` 入力 RMS < 閾値（`SILENCE_RMS_THRESHOLD=1e-4`）で encode/convert/decode を skip。ただし ARCH §8 は「output silence」だが実装は入力パススルー（`chunk_pcm.to_vec()`）。機能等価だが厳密には不一致
  - ✅ **NaN/Inf クランプ**: `pipeline.rs:158-161` で `[-1, 1]` clamp、NaN → 0
  - ✅ **参照音声長チェック**: `pipeline.rs:88-95` で 44100 samples（1 秒）未満はエラー
  - ⚠️ **Ring buffer overrun**: [07-5] へ（仕様「最古破棄」が実装は「最新破棄」）
  - ⚠️ **Ring buffer underrun**: [07-5] へ（自動低品質化が未実装）
  - ⚠️ **デバイス切断**: [07-5] へ（デバイス選択画面が存在しない）
- **関連**: `ARCHITECTURE.md:613-622`, `crates/lightvc-core/src/pipeline.rs:88-95,114-117,158-161`

### [07-5] (P1) ✅ ARCHITECTURE §8 エッジケース 3 件の仕様違反修正
- **現状**: [07-4] で ✅ としたが、詳細検証で以下 3 件が仕様を満たさない:

  1. **overrun ポリシー逆転** (`engine.rs:114-116`):
     ```rust
     if cap_tx.push(sample).is_err() {
         cap_flags.overrun.fetch_add(1, Ordering::Relaxed);
     }
     ```
     ARCH §8 / plan 07-4 line 73 は「最古サンプル破棄（drop OLDEST）」。実装は `push` 失敗で**入力サンプル（NEWEST）を黙って破棄**。rtrb の `Producer::push` は退避しない。「drop incoming sample」というコメント（111-113行）も「drop incoming」と明記し ARCH §8 と矛盾。

  2. **underrun 自動品質ダウングレード不在** (`engine.rs:142-149`):
     ```rust
     for sample in data.iter_mut() {
         *sample = pb_rx.pop().unwrap_or(0.0);  // 無音
         if pb_rx.pop().is_err() { pb_flags.underrun.fetch_add(1,...); }
     }
     ```
     ARCH §8 は「無音出力 + 警告ログ + **自動 low quality 化**」。実装は無音 + カウンタのみ。`LatencyMode`/`ChunkMode` を下げるロジックが `engine.rs`・`realtime_tab.rs`・`app.rs` のいずれにも不在。`realtime_tab.rs:136-145` は underrun 数を黄色ラベル表示するだけ。

  3. **切断時のデバイス選択画面が存在しない**:
     - cpal error callback → `disconnected: AtomicBool`（`engine.rs:119-124,152-157`）は実装済み
     - `realtime_tab.rs:411-431` で tear down、`app.rs:344-353` で「Audio device disconnected」表示は動作
     - しかし ARCH §8 / `app.rs:19-21` doc-comment が言う「return to the device-selection screen」の**画面が存在しない**。`realtime_tab.rs:261-299` は read-only ラベルのみ（クリックハンドラ無し、[05-6] と関連）。切断後は同じ default デバイスへ再接続するのみ。

- **作業**:
  1. **overrun**: rtrb は退避 API を持たないため、リング容量を 1 フレーム余分に確保し `push` 失敗時に `Consumer::pop()` で最古を破棄してから再 push。または `rtrb` を `heapless::spsc::Queue` 等の退避可能キューへ交換
  2. **underrun**: `realtime_tab.rs` で underrun 発生率を監視（例: 直近 100 chunk で 5 回超）→ `RtControl::SetMode(LatencyMode::Strict)` 等の自動ダウングレード。UI に「auto-degraded」表示
  3. **切断画面**: デバイス選択可能な画面を実装（[05-6] と統合）
- **受け入れ基準**:
  - overrun で最古サンプルが破棄され入力サンプルは保持される
  - underrun 多発で自動的に低レイテンシ/低負荷モードへ移行
  - 切断後にデバイス選択画面へ遷移し別デバイスを選べる
- **関連**: `crates/lightvc-audio/src/engine.rs:111-124,142-157`, `crates/lightvc-app/src/realtime_tab.rs:136-145,261-299,411-431`, `crates/lightvc-app/src/app.rs:19-21,344-353`, `ARCHITECTURE.md:613-622`

## 関連文書
- [03_converter_model.md](03_converter_model.md)
- [04_training_pipeline.md](04_training_pipeline.md)
