# 07: 未実装の設計機能

> カテゴリ: G
> 関連資料: CONCEPT.md（全般）, DESIGN.md §3, ARCHITECTURE.md §4.3-4.4, §8

## 概要

CONCEPT.md / DESIGN.md で構想されているが、Rust / Python とも実装されていない機能群。研究的新規性（Phase 3）やプロダクト価値（Phase 4）に直結するが、Phase 1-2 の安定化（[02] [03] [04]）が先決。

## 現状の乖離

| 機能 | 設計 | 実装 |
|---|---|---|
| Phase 3: Progressive RVQ-depth factorized FM heads | CONCEPT「新規性の核心」, DESIGN §3, MODEL_TRAINING 表 | なし（quantizer 自体をスキップ） |
| Phase 4: Prosody/Rhythm factorization | ARCHITECTURE §4.4, CONCEPT SOTAネタ5 | `prosody_mode` enum すらない |
| dual-path converter | CONCEPT「さらに攻めるなら」 | なし |
| エッジケース処理 | ARCHITECTURE §8（6 項目） | なし |

## タスクリスト

### [07-1] (P2) 🚧 Phase 3: Progressive RVQ-depth factorized FM heads
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

### [07-4] (P1) ✅ ARCHITECTURE §8 エッジケースの実装
- **現状**: ARCHITECTURE.md §8（613-622 行）が列挙する 6 シナリオのうち、1 つも実装されていない:
  - Ring buffer underrun（capture too fast）→ 古いサンプル破棄
  - Ring buffer overrun（inference too slow）→ 無音出力 + 自動 low quality 化
  - 参照音声短すぎ（<1s）→ UI でエラー
  - サイレンス検出（無入力）→ skip で CPU セーブ
  - DAC decode NaN/Inf → clamp [-1, 1]
  - デバイス切断 → pipeline 停止、デバイス選択画面へ
- **影響**: リアルタイム運用時の堅牢性不足。異常系でクラッシュやノイズ音が発生し得る。
- **作業**:
  1. **サイレンス検出**: `pipeline.process_chunk` 入力 RMS < 閾値（例: 1e-4）で encode/convert/decode を skip、入力をそのまま出力
  2. **NaN/Inf クランプ**: `streaming.decode_step` 出力を `[-1, 1]` に clamp、NaN は 0 に置換
  3. **参照音声長チェック**: `pipeline.set_target` で 44100 samples（1 秒）未満はエラー
  4. **Ring buffer overrun**: `capture_tx.push` 失敗時、最古サンプル破棄（`rtrb` は自動でできるか要確認）
  5. **デバイス切断**: cpal error callback で `RtControl::Stop` を送信、UI に通知
- **受け入れ基準**: 各シナリオでクラッシュせず、適切にフォールバックまたは UI 通知されること。
- **関連**: `ARCHITECTURE.md:613-622`, `crates/lightvc-core/src/pipeline.rs`, `crates/lightvc-app/src/realtime_tab.rs`, `crates/lightvc-clap/src/lib.rs`

## 関連文書
- [03_converter_model.md](03_converter_model.md)
- [04_training_pipeline.md](04_training_pipeline.md)
