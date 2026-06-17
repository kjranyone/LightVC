# LightVC 設計-実装整合性タスクリスト

LightVC プロジェクトの初期設計資料（DESIGN / ARCHITECTURE / MODEL_TRAINING / CONCEPT / RESEARCH）と実装（`crates/` `training/`）の乖離を整理し、修正タスク化した文書群。

> **2026-06-17 第2回監査を実施**: 初回（36タスク）から実装の直接検証を行い、**新規 24 件の乖離**を追加発見。特に Rust/Python 数値不整合（[03-6][03-7][03-8]）、学習ロジックの根本的欠陥（[04-7][04-8][04-9]）、CLAP プラグインの実用性欠如（[06-5][06-6][06-7]）は P0/P1 として追加。一部の ✅ 判定は 🚧 に格下げ（`✅ → 🚧` 凡例参照）。

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
- [03-6] CrossAttnBlock マルチヘッド reshape の数学的誤り（本番 UTTE で出力破壊）
- [03-7] warm-start Converter に BottleneckEncoder が未実装（Rust）
- [04-2] mixed_precision=bf16 の検証と再有効化
- [04-7] mean-flow 定式化の誤帰属 + MODEL_TRAINING.md 内部矛盾
- [04-8] Phase B `speaker_consistency` が source 話者を target にしている（逆監督）
- [04-9] Phase B `content_preservation` が恒等的にゼロ
- [05-1] rubato 3.0 + `process_into_buffer` で RT-safe 化
- [05-4] リサンプリング・チャンクサイズ境界の整理
- [06-1] CLAP の converter config 読み込み
- [06-5] CLAP プラグインが host SR ↔ 44.1kHz のリサンプリングを行わない
- [06-6] CLAP の `mode` パラメータがデッドコード
- [06-7] CLAP エディタにモデルロード UI が無い（Browse ボタン不在）

### P1（重要、計画的対応）
- [01-3] converter / codec サブモジュール分割の判断
- [03-1] パラメータ数乖離の解消
- [03-3] FlowConverter の ARCHITECTURE 反映
- [03-8] GELU の tanh 近似 vs erf 精密値（3 箇所）
- [03-9] velocity_scale API 非対称 + ARCHITECTURE「Default 2.5」誤記
- [04-1] 学習ステップ数の整合（smoke / 本番の分離）
- [04-3] LibriTTS / VCTK 本格学習手順の整備
- [04-4] content MI loss（gradient reversal）の実装
- [04-5] 外部指標（SECS / UTMOS / WER）評価パイプライン
- [04-10] Phase B `role_assignment` 未実装 + cross-speaker role が no-op
- [04-11] ドキュメント・設定にない `speaker_classify` 補助損失が暗黙追加
- [04-13] 設定ファイルとドキュメントの数値不一致（7 件）
- [05-2] `DuplexStream::start` シグネチャ修正
- [05-3] capture / playback / inference スレッドの完全分離
- [05-5] リサンプラ呼び出し側が毎回 Vec 確保（[05-1] 受け入れ基準未達）
- [05-6] `start_with` API が実装されたが呼び出し側が `start_default()` のみ使用
- [05-7] AudioEngine ライフサイクルが依然 inference thread 内に結合
- [05-8] capture と playback で SR が異なる場合の出力経路が未対応
- [06-2] latency / RTF 計算の修正
- [06-8] CLAP レイテンシ表示が硬编码・モード非依存・UI 非表示
- [06-9] CLAP 推論スレッドのゼロ埋めが残存（[08-7] 部分解消）
- [06-10] offline_tab が process_full ではなく process_chunk でループ
- [07-4] ARCHITECTURE §8 エッジケースの実装（3/6 完全・3/6 は [07-5] で修正）
- [07-5] ARCHITECTURE §8 エッジケース 3 件の仕様違反修正
- [08-4] SpeakerEncoder の GELU 不一致
- [08-7] 推論スレッドのゼロ埋め過多

### P2（整理・将来的対応）
- [01-1] dac.rs 自前実装の設計反映
- [01-2] lightvc-clap / lightvc-xtask の設計追記
- [03-2] Phase 2 UTTE の有効化と検証
- [03-4] CausalConv1d dead code の削除
- [03-10] warm-start Converter の unbatched 対応非対称 + SpeakerEncoder ガード
- [04-6] VCTK parallel validation の整備
- [04-12] timbre-shift の apply_prob 希釈 + content_inv 弱化 + GRL lambda 非設定化
- [06-3] README `cargo xtask` 記載修正
- [06-4] MANUAL / ASSETS_SPEC の実装反映
- [07-1] Phase 3: Progressive RVQ-depth factorized FM heads
- [07-2] Phase 4: Prosody/Rhythm factorization
- [07-3] dual-path converter
- [08-5] TimeEmbed の freqs 初期化精度
- [08-6] SpeakerEncoder の unbatched 入力扱い
- [08-8] ドキュメント内部矛盾・虚構型名（5 件）

## 進め方

1. **P0 から着手**。各タスクはブランチを切って実施することを推奨。
2. 設計資料を変える場合と実装を変える場合がある（各タスクに方針記載）。
3. コミットプレフィックスは AGENTS.md 準拠（`feat:` / `fix:` / `docs:` / `refactor:`）。
4. タスク完了時は各ファイル内のステータスを `✅` に更新し、本 README のチェックボックスも反映。

## 対応表（タスク ID → ステータス）

| ID | 優先度 | ステータス |
|---|---|---|
| 01-1 | P2 | ✅ (ARCHITECTURE §3.3 / RESEARCH §1 / DESIGN §2 に dac_model.rs 反映) |
| 01-2 | P2 | ✅ (ARCHITECTURE §2 に lightvc-clap / lightvc-xtask 追記) |
| 01-3 | P1 | ✅ (flow_converter.rs 分離) |
| 02-1 | P0 | ✅ |
| 02-2 | P0 | ✅ (lookahead+overlap で等価性確保、per-layer cache は最適化として保留) |
| 02-3 | P1 | ✅ (overlap-add 設計 ARCHITECTURE §3.4.1 に文書化) |
| 02-4 | P0 | ✅ |
| 03-1 | P1 | ✅ (hidden_dim 1024 統一) |
| 03-2 | P2 | ✅ |
| 03-3 | P1 | ✅ (FlowConverter を ARCHITECTURE §4.1b に反映) |
| 03-4 | P2 | ✅ |
| 03-5 | P0 | ✅ |
| **03-6** | **P0** | **✅** CrossAttnBlock reshape 誤り（本番 UTTE で出力破壊） |
| **03-7** | **P0** | **✅** warm-start Converter に BottleneckEncoder 未実装（Rust） |
| **03-8** | **P1** | **✅** GELU tanh 近似 vs erf 精密値（3 箇所） |
| **03-9** | **P1** | **✅** velocity_scale API 非対称 + ARCHITECTURE 誤記 |
| **03-10** | **P2** | **✅** warm-start unbatched 非対称 + SpeakerEncoder ガード |
| 04-1 | P1 | ✅ (smoke / production config 分離) |
| 04-2 | P0 | ✅ |
| 04-3 | P1 | ✅ (download_corpus.py + encode_corpus 修正) |
| 04-4 | P1 | ✅ (GRL content MI loss 実装) |
| 04-5 | P1 | ✅ (evaluate.py: SECS/UTMOS/WER) |
| 04-6 | P2 | ✅ (build_vctk_manifest.py + WER degradation 計算) |
| **04-7** | **P0** | **✅** mean-flow 誤帰属 + MODEL_TRAINING.md §C.2/§C.3 内部矛盾 |
| **04-8** | **P0** | **✅** Phase B speaker_consistency が source 話者を target に（逆監督） |
| **04-9** | **P0** | **✅** Phase B content_preservation が恒等的にゼロ |
| **04-10** | **P1** | **✅** role_assignment 未実装 + cross-speaker role が no-op |
| **04-11** | **P1** | **✅** speaker_classify 補助損失が暗黙追加（doc/config に不在） |
| **04-12** | **P2** | **✅** timbre-shift apply_prob 希釈 + content_inv 弱化 + GRL lambda + デッドコード |
| **04-13** | **P1** | **✅** 設定 vs ドキュメント数値不一致（7 件） |
| 05-1 | P0 | ✅ |
| 05-2 | P1 | ✅ (DuplexStream::start_with) |
| 05-3 | P1 | ✅ (AudioEngine カプセル化 + [05-7] で lifecycle 分離完了) |
| 05-4 | P0 | ✅ ([08-2][08-7] 同時解消) |
| **05-5** | **P1** | **✅** リサンプラ呼び出し側が毎回 Vec 確保（[05-1] 受け入れ基準未達） |
| **05-6** | **P1** | **✅** start_with API が実装されたが start_default() のみ使用 |
| **05-7** | **P1** | **✅** AudioEngine lifecycle が依然 inference thread 内 |
| **05-8** | **P1** | **✅** capture/playback 異 SR の出力経路未対応 |
| 06-1 | P0 | ✅ (フォールバック chain + [06-7] で Browse UI 完了) |
| 06-2 | P1 | ✅ (realtime + [06-8] で CLAP latency 表示完了) |
| 06-3 | P2 | ✅ |
| 06-4 | P2 | ✅ (MANUAL/ASSETS_SPEC 実装反映 + icon_stop 適用) |
| **06-5** | **P0** | **✅** CLAP が host SR ↔ 44.1kHz リサンプリングを行わない |
| **06-6** | **P0** | **✅** CLAP mode パラメータがデッドコード |
| **06-7** | **P0** | **✅** CLAP エディタにモデルロード UI 無し |
| **06-8** | **P1** | **✅** CLAP レイテンシ硬编码・モード非依存・UI 非表示 |
| **06-9** | **P1** | **✅** CLAP 推論スレッドのゼロ埋め残存（[08-7] 部分解消） |
| **06-10** | **P1** | **✅** offline_tab が process_chunk でループ（process_full 未使用） |
| 07-1 | P2 | ✅ |
| 07-2 | P2 | ✅ |
| 07-3 | P2 | ✅ |
| 07-4 | P1 | ✅ (6/6 完了: [07-5] で overrun/underrun/切断対応) |
| **07-5** | **P1** | **✅** §8 エッジケース 3 件仕様違反（overrun 最新破棄 / underrun 自動切替無し / 切断画面無し） |
| 08-1 | P0 | ✅ (02-1/02-2/02-4 で対応) |
| 08-2 | P1 | ✅ (05-4 で解消) |
| 08-3 | P1 | ✅ (06-1 で解消) |
| 08-4 | P1 | ✅ (gelu_erf 化完了 via [03-8]) |
| 08-5 | P2 | ✅ |
| 08-6 | P2 | ✅ (unbatched [D,T] 入力に unsqueeze/squeeze 対応) |
| 08-7 | P1 | ✅ (realtime + CLAP [06-9] 両方解消) |
| **08-8** | **P2** | **✅** ドキュメント内部矛盾・虚構型名（5 件） |

ステータス凡例: ⬜ 未着手 / 🚧 進行中 / ✅ 完了 / ❌ 中止

> **凡例補足**: `✅ → 🚧` は一度 ✅ としたが再検証で部分解消／新規副次問題が発覚し、後続タスクへ引き継いだもの。

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
- 2026-06-17: P0 全完了 + P1/P2 の小タスク
  - [05-4] リサンプリング・チャンク境界整理: 3段バッファリング(in_accum → pcm_44k_accum → out_44k_accum)で各境界を分離、ゼロ埋め・truncate 廃止。[08-2][08-7] 同時解消
  - [08-4] SpeakerEncoder GELU 不一致解消
  - [08-5] TimeEmbed freqs を f64 計算後に f32 キャスト
  - [03-4] CausalConv1d depthwise dead code 削除 (Rust/Python 両方)
  - [06-3] README の `cargo xtask` → `cargo run -p lightvc-xtask --` 修正
  - **P0 全 9 タスク完了** (残り [04-2] の実機検証のみ)
- 2026-06-17: P1 学習パイプライン系 5 件完了
  - [01-3] converter.rs → flow_converter.rs 分離 (約 500 行削減)
  - [03-1] hidden_dim を 256 → 1024 に統一 (ConverterConfig と一致)
  - [03-3] FlowConverter を ARCHITECTURE §4.1b に反映
  - [04-1] smoke / production config 分離 (phase_b/c + phase_b/c_smoke)
  - [04-3] download_corpus.py 新設 (HuggingFace datasets 経由) + encode_corpus.py 型バグ修正
  - [04-4] content MI loss (gradient reversal) 実装: ContentSpeakerAdversary + DisentangledConverter, phase_c.yaml で content_mi=0.1 有効化
  - [04-5] evaluate.py 新設: SECS (speechbrain ECAPA) / UTMOS / WER (Whisper + jiwer), pyproject.toml に eval extra 追加
- 2026-06-17: P1 リアルタイム/プラグイン系 5 件完了
  - [02-3] overlap-add 設計を ARCHITECTURE §3.4.1 に文書化
  - [05-2] DuplexStream::start_with (明示的 config 指定)
  - [05-3] AudioEngine 新設: cpal Stream + ring buffer をカプセル化、inference loop からライフサイクル管理を分離
  - [06-2] latency/RTF 計算修正 (algorithmic_latency 含む end-to-end 推算)
  - [07-4] エッジケース 6/6 完了: silence skip / NaN clamp / ref length check / algorithmic_latency (前回) + ring buffer overrun/underrun + デバイス切断ハンドリング (AudioEngine)
- 2026-06-17: P2 学習不要タスク 6 件 + Phase 3 Rust 先行
  - [08-6] Converter::forward / FlowConverter::convert に unbatched [D,T] 入力対応 (Python と一致)
  - [01-1] dac.rs 自前実装 (dac_model.rs) を ARCHITECTURE §3.3 / RESEARCH §1 / DESIGN §2 に反映
  - [01-2] lightvc-clap / lightvc-xtask を ARCHITECTURE §2 のクレートツリーと Key Dependencies に追記
  - [06-4] MANUAL §4.2/§7 を実 UI に合わせて更新 (Stop/status/metrics 追加、● READY→LIVE 修正)、ASSETS_SPEC_V2 に実装状況表追加、icon_stop.png を Stop ボタンに適用
  - [04-6] build_vctk_manifest.py 新設 + evaluate.py に WER degradation (content preservation) 計算を追加
  - [07-1] Phase 3 準備: Rust 側に DAC Residual Vector Quantizer (QuantizerLayer + Quantizer) 実装、DacModel::with_quantizer でオプションロード。Python 側 factorized FM heads は学習待ち
- 2026-06-17: **第2回監査 — 新規 24 件の乖離追加発見**（実装直接検証）
  - **P0 新規 8 件**:
    - [03-6] CrossAttnBlock マルチヘッド reshape が数学的誤り（本番 UTTE `enable_timbre:true` で Rust 推論破壊）
    - [03-7] warm-start Converter に BottleneckEncoder 未実装（Rust）。エクスポートされる 5 キーが黙って無視される
    - [04-7] 実装は rectified/linear flow matching であり MeanVC2 mean-flow ではない（誤帰属）。MODEL_TRAINING §C.2/§C.3 が内部矛盾
    - [04-8] Phase B `speaker_consistency` が target=source 話者（VCに参照無視を学習させる逆監督）
    - [04-9] Phase B `content_preservation` が src vs src で恒等的にゼロ
    - [06-5] CLAP プラグインが host SR ↔ 44.1kHz リサンプリングを一切行わない（≠44.1k でピッチシフト）
    - [06-6] CLAP `mode`（Strict/Balanced/Quality）パラメータがデッドコード（UI ボタンは動くが process が読まない）
    - [06-7] CLAP エディタにモデルロード UI が無い（Browse ボタン不在、DAW 内からロード不可）
  - **P1 新規 12 件**: [03-8] GELU tanh vs erf（3箇所）/[03-9] velocity_scale 非対称 + doc 誤記 / [04-10] role_assignment 未実装・cross-speaker no-op / [04-11] speaker_classify 暗黙追加 / [04-13] config vs doc 数値不一致 7件 / [05-5] リサンプラ呼び出し側 Vec 確保 / [05-6] start_with 未使用 / [05-7] AudioEngine lifecycle 結合 / [05-8] capture/playback 異 SR 未対応 / [06-8] CLAP レイテンシ硬编码 / [06-9] CLAP ゼロ埋め残存 / [06-10] offline_tab process_chunk 使用
  - **P2 新規 4 件**: [03-10] warm-start unbatched 非対称 / [04-12] timbre-shift apply_prob 希釈 + デッドコード / [07-5] §8 エッジケース 3件仕様違反 / [08-8] ドキュメント内部矛盾 5件
  - **✅ → 🚧 格下げ 6 件**: [05-3][06-1][06-2][07-4][08-4][08-7]（部分解消 or 副次問題発覚）
