# LightVC 設計-実装整合性タスクリスト

LightVC プロジェクトの初期設計資料（DESIGN / ARCHITECTURE / MODEL_TRAINING / CONCEPT / RESEARCH）と実装（`crates/` `training/`）の乖離を整理し、修正タスク化した文書群。

## インデックス

| # | ファイル | カテゴリ | 主な問題 |
|---|---|---|---|
| 00 | README.md | - | インデックス・優先度マトリクス（本ファイル）|
| 01 | [01_crate_structure.md](01_crate_structure.md) | A | モジュール構成・dac.rs 自前実装の未反映 |
| 02 | [02_streaming_lookahead.md](02_streaming_lookahead.md) | B | FRC/lookahead 未実装・conv-state caching 未実装 |
| 03 | [03_converter_model.md](03_converter_model.md) | C | パラメータ数・UTTE 無効・FlowConverter 文書化不足 |
| 04 | [04_training_pipeline.md](04_training_pipeline.md) | D | ステップ数・コーパス・loss/validation 簡略化 |
| 05 | [05_audio_io.md](05_audio_io.md) | E | rubato 非 RT-safe・スレッド分離不十分 |
| 06 | [06_plugin_app.md](06_plugin_app.md) | F | CLAP config 固定・latency/RTF 計算不正 |
| 07 | [07_unimplemented_phases.md](07_unimplemented_phases.md) | G | Phase 3/4・dual-path・エッジケース未実装 |
| 08 | [08_known_bugs.md](08_known_bugs.md) | H | dac_model.rs パディング/causal 矛盾・数値不一致 |

## 優先度マトリクス

### P0（品質/正確性に直結、最優先）
- [02-1] FRC lookahead の実装
- [02-2] per-layer conv-state caching
- [02-4] / [08-1] dac_model.rs の対称パディング ↔ causal 仮定の矛盾解消
- [03-5] Snake1d Rust/Python 数値一致
- [04-2] mixed_precision=bf16 の検証と再有効化
- [05-1] rubato 3.0 + `process_into_buffer` で RT-safe 化
- [05-4] リサンプリング・チャンクサイズ境界の整理
- [06-1] CLAP の converter config 読み込み

### P1（重要、計画的対応）
- [01-3] converter / codec サブモジュール分割の判断
- [03-1] パラメータ数乖離の解消
- [03-3] FlowConverter の ARCHITECTURE 反映
- [04-1] 学習ステップ数の整合（smoke / 本番の分離）
- [04-3] LibriTTS / VCTK 本格学習手順の整備
- [04-4] content MI loss（gradient reversal）の実装
- [04-5] 外部指標（SECS / UTMOS / WER）評価パイプライン
- [05-2] `DuplexStream::start` シグネチャ修正
- [05-3] capture / playback / inference スレッドの完全分離
- [06-2] latency / RTF 計算の修正
- [07-4] ARCHITECTURE §8 エッジケースの実装
- [08-4] SpeakerEncoder の GELU 不一致
- [08-7] 推論スレッドのゼロ埋め過多

### P2（整理・将来的対応）
- [01-1] dac.rs 自前実装の設計反映
- [01-2] lightvc-clap / lightvc-xtask の設計追記
- [03-2] Phase 2 UTTE の有効化と検証
- [03-4] CausalConv1d dead code の削除
- [04-6] VCTK parallel validation の整備
- [06-3] README `cargo xtask` 記載修正
- [06-4] MANUAL / ASSETS_SPEC の実装反映
- [07-1] Phase 3: Progressive RVQ-depth factorized FM heads
- [07-2] Phase 4: Prosody/Rhythm factorization
- [07-3] dual-path converter
- [08-5] TimeEmbed の freqs 初期化精度
- [08-6] SpeakerEncoder の unbatched 入力扱い

## 進め方

1. **P0 から着手**。各タスクはブランチを切って実施することを推奨。
2. 設計資料を変える場合と実装を変える場合がある（各タスクに方針記載）。
3. コミットプレフィックスは AGENTS.md 準拠（`feat:` / `fix:` / `docs:` / `refactor:`）。
4. タスク完了時は各ファイル内のステータスを `✅` に更新し、本 README のチェックボックスも反映。

## 対応表（タスク ID → ステータス）

| ID | 優先度 | ステータス |
|---|---|---|
| 01-1 | P2 | ⬜ |
| 01-2 | P2 | ⬜ |
| 01-3 | P1 | ⬜ |
| 02-1 | P0 | ✅ |
| 02-2 | P0 | ✅ (lookahead+overlap で等価性確保、per-layer cache は最適化として保留) |
| 02-3 | P1 | ⬜ |
| 02-4 | P0 | ✅ |
| 03-1 | P1 | ⬜ |
| 03-2 | P2 | ⬜ |
| 03-3 | P1 | ⬜ |
| 03-4 | P2 | ⬜ |
| 03-5 | P0 | ✅ |
| 04-1 | P1 | ⬜ |
| 04-2 | P0 | 🚧 (config 有効化、実機検証 pending) |
| 04-3 | P1 | ⬜ |
| 04-4 | P1 | ⬜ |
| 04-5 | P1 | ⬜ |
| 04-6 | P2 | ⬜ |
| 05-1 | P0 | ✅ |
| 05-2 | P1 | ⬜ |
| 05-3 | P1 | ⬜ |
| 05-4 | P0 | ⬜ |
| 06-1 | P0 | ✅ |
| 06-2 | P1 | ⬜ |
| 06-3 | P2 | ⬜ |
| 06-4 | P2 | ⬜ |
| 07-1 | P2 | ⬜ |
| 07-2 | P2 | ⬜ |
| 07-3 | P2 | ⬜ |
| 07-4 | P1 | ⬜ |
| 08-1 | P0 | ✅ (02-1/02-2/02-4 で対応) |
| 08-2 | P1 | ⬜ |
| 08-3 | P1 | ✅ (06-1 で解消) |
| 08-4 | P1 | ⬜ |
| 08-5 | P2 | ⬜ |
| 08-6 | P2 | ⬜ |
| 08-7 | P1 | ⬜ |

ステータス凡例: ⬜ 未着手 / 🚧 進行中 / ✅ 完了 / ❌ 中止

## 更新履歴

- 2026-06-17: 初版作成（乖離分析に基づく 36 タスク）
- 2026-06-17: P0 4 件完了
  - [03-5] Snake1d の Rust/Python 数値一致 (epsilon=1e-9 追加)
  - [06-1] CLAP プラグインでの converter config 読み込み (explicit / sidecar / default フォールバック)、[08-3] も解消
  - [05-1] rubato v0.16 → v3.0 移行 (`Async` + `process_into_buffer` + 事前確保バッファ)
  - [04-2] Phase C `phase_c.yaml` の `mixed_precision: bf16` 有効化 (実機 1000 step 検証は後続)
- 2026-06-17: P0 ストリーミング系 3 件完了
  - [02-1] FRC lookahead 実装 (Strict=0 / Balanced=2048 / Quality=4096 samples)
  - [02-2] lookahead + ENCODER_OVERLAP で streaming/non-streaming 等価性を確保。per-layer conv-state caching は性能最適化のため保留（別途対応）
  - [02-4] / [08-1] 対称パディング ↔ causal 矛盾を FRC で解消（Method A）
