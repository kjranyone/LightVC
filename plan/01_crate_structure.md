# 01: クレート / モジュール構成の乖離

> カテゴリ: A
> 関連資料: ARCHITECTURE.md §2, RESEARCH.md §1, DESIGN.md §2

## 概要

ARCHITECTURE.md §2 が記載する `lightvc-core` のサブモジュール構造と、実際のフラット構成が異なる。また「Candle 標準の dac.rs を使う」という設計根拠が崩れ、自前実装（`dac_model.rs`）に置き換わっている。`lightvc-clap` / `lightvc-xtask` クレートは実装済みだが設計資料に未記載。

## 現状の乖離

| 設計（ARCHITECTURE.md §2） | 実装 |
|---|---|
| `codec/{mod,encoder,decoder,streaming}.rs` | `codec.rs` フラット |
| `converter/{mod,conv_block,timbre,config}.rs` | `converter.rs` フラット |
| `pipeline.rs`, `weights.rs` | 一致 |
| `candle-transformers::models::dac` 利用（RESEARCH §1, DESIGN §2） | `dac_model.rs` 自前実装（~400 LOC） |
| `lightvc-app` / `lightvc-audio` のみ記載 | `lightvc-clap` / `lightvc-xtask` も実装済み |
| `models/converter_p1.safetensors` `converter_p2.safetensors` | 実運用は `converter.safetensors` 1つ（model_type 切替） |

## タスクリスト

### [01-1] (P2) ✅ dac.rs 自前実装の事実を設計資料に反映
- **現状**: RESEARCH.md §1（28-37 行）と DESIGN.md §2 は「Candle に DAC 実装あり、encoder だけ wire up（~50 LOC）」と記載。実際は encoder / decoder / Snake / ResidualUnit / EncoderBlock / DecoderBlock のすべてを `dac_model.rs`（396 行）で再実装。
- **背景**: HuggingFace `descript/dac_44khz` の safetensors キー名が transformers 流であり、Candle 標準 `dac.rs`（PyTorch オリジナル名前提）と一致しないため。ARCHITECTURE.md §6.3 で既出の問題だが、対応（自前実装）が文書化されていない。
- **作業**:
  1. ARCHITECTURE.md §3.3 を更新：「Candle 標準 dac.rs ではなく、HF safetensors キー名に対応する自前実装（`dac_model.rs`）を採用」と明記
  2. RESEARCH.md §1 の "Candle's Existing Codec Implementations" 表に注記追加
  3. DESIGN.md §2 の "already in Candle" 根拠を修正
- **判断**: 実装維持（自前実装は妥当）。ドキュメント更新のみ。
- **受け入れ基準**: ARCHITECTURE / RESEARCH / DESIGN に `dac_model.rs` が明示され、設計根拠が整合していること。
- **関連**: `crates/lightvc-core/src/dac_model.rs`, `RESEARCH.md:23-37`, `ARCHITECTURE.md:200-217`, `DESIGN.md:29-43`

### [01-2] (P2) ✅ lightvc-clap / lightvc-xtask を設計に追記
- **現状**: ARCHITECTURE.md §2 のクレート構成（71-131 行）に含まれていない。README.md の Project Structure（88-100 行）には記載あり。
- **作業**:
  1. ARCHITECTURE.md §2 に `lightvc-clap`（CLAP/VST3 plugin）と `lightvc-xtask`（bundle / install 自動化）を追記
  2. 依存関係（`nice-plug` / `nice-plug-egui` / `clap-wrapper` / `clap-sys`）とライセンス（MIT/ISC）を明記
  3. AGENTS.md の「Licensing」セクションと整合確認
- **受け入れ基準**: ARCHITECTURE.md §2 が実際の `Cargo.toml` の workspace members と完全一致すること。
- **関連**: `Cargo.toml:1-15`, `crates/lightvc-clap/Cargo.toml`, `crates/lightvc-xtask/src/main.rs`, `ARCHITECTURE.md:71-131`

### [01-3] (P1) ✅ converter / codec サブモジュール分割の判断
- **現状**: ARCHITECTURE.md §2 は `codec/` と `converter/` をサブモジュールディレクトリで記載。実装は単一ファイル（`codec.rs` 112 行、`converter.rs` **767 行**）。`converter.rs` は長大で読みにくい。
- **作業**: 以下いずれかを選択
  - **(a) 実装を設計に合わせて分割**
    - `codec/{mod,encoder,decoder}.rs`
    - `converter/{mod,conv_block,timbre,config,flow}.rs`
  - **(b) 設計を実装に合わせてフラット表記に修正**
- **推奨**: **(a) のうち、`flow_converter.rs` の切り出しのみ実施**。
  - `converter.rs`（767 行）には Phase 1 `Converter` と Phase C `FlowConverter` + 共有モジュールが混在
  - `FlowConverter`（120 行）と `AnyConverter` enum（30 行）を `flow_converter.rs` に分離すれば、`converter.rs` は ~600 行に収まる
  - `codec.rs`（112 行）はフラットのまま（分割コストに見合わない）
  - 設計資料は実装に合わせて修正
- **受け入れ基準**: 設計と実装の構成記述が一致すること。`cargo check --workspace` が通る。
- **関連**: `crates/lightvc-core/src/converter.rs`（767 行）, `crates/lightvc-core/src/codec.rs`, `ARCHITECTURE.md:71-131`

## 関連文書
- [02_streaming_lookahead.md](02_streaming_lookahead.md)
- [03_converter_model.md](03_converter_model.md)
