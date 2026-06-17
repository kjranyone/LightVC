# 03: コンバータモデルの乖離

> カテゴリ: C
> 関連資料: DESIGN.md §3, §4, ARCHITECTURE.md §4, MODEL_TRAINING.md C.2-C.4

## 概要

コンバータのアーキテクチャ実装は概ね設計通りだが、パラメータ数、Phase 2 UTTE の状態、FlowConverter の文書化、細かい数値整合性に乖離がある。Rust/Python 間の数値不一致は学習済み重みの移植性を損なう。

## 現状の乖離

| 項目 | 設計 | 実装 |
|---|---|---|
| `hidden_dim` | 1024（DESIGN §4.1, README） | 256（`phase_b.yaml` / `phase_c.yaml`） |
| パラメータ数 | 8-12M / Phase2 15-30M | ~6M（Phase 1） |
| Phase 2 UTTE | K=32 token bank + cross-attn | `enable_timbre: false`、実質無効 |
| FlowConverter | 設計に明記なし | 実装あり（`model_type` 切替） |
| CausalConv1d keys | 標準 + depthwise フォールバック | depthwise は XPU 不可で dead code |
| Snake1d 数式 | `1/(alpha+eps) * sin^2(alpha*x)` | Rust: epsilon なし / Python: `1/(alpha+1e-9)` |

## タスクリスト

### [03-1] (P1) ✅ パラメータ数乖離の解消
- **現状**: DESIGN.md §4.1（295-315 行）は「Conv1d(1024→1024)」、README.md:106 は「~10M params」。実装の `hidden_dim: 256` だと `CausalResBlock` は `1024→256→1024` 結合で約 6M パラメータ。
- **影響**: 設計で想定した表現力が得られていない可能性。ただし現在の学習ステップ数（30K）では差は顕在化しにくい。
- **作業**: 以下いずれかを選択して整合
  - **(a) 実装を設計に合わせる**: `hidden_dim` を 1024 に戻し、再学習。実際に 86 Hz × 10M でも CPU 1 chunk ~5ms は達成可能（ARCHITECTURE §7.1）。
  - **(b) 設計を実装に合わせる**: DESIGN.md と README を「`hidden_dim=256`, ~6M params」に修正
- **推奨**: **(a)**。Phase C 学習はまだ小規模（30K step）なので再学習コストは低い。品質底上げが期待できる。ただし [04-1] と併せて smoke / 本番の位置づけを明確にしてから着手。
- **受け入れ基準**: 設計記載のパラメータ数と、`sum(p.numel())` の出力が ±10% 以内で一致。
- **関連**: `training/configs/phase_b.yaml:4`, `training/configs/phase_c.yaml:4`, `DESIGN.md:53-60, 314`, `README.md:106`

### [03-2] (P2) Phase 2 UTTE の有効化と検証
- **現状**: `enable_timbre: false` がデフォルト（`ConverterConfig::default()` / `phase_b.yaml` / `phase_c.yaml`）。`TimbreTokenBank` と `CrossAttnBlock` は実装されているが学習・推論で使われていない。
- **影響**: ゼロショット VC の品質向上（RESEARCH §2 MeanVC2 の主要成果）が未検証。Astrape 超えの差別化要素の 1 つ。
- **作業**:
  1. `enable_timbre: true` の Phase C 学習設定追加（`configs/phase_c_utte.yaml`）
  2. timbre なし / ありの A/B 比較（SECS, UTMOS）— [04-5] の評価パイプラインが必要
  3. `n_timbre_tokens=32`, `n_attn_heads=8`（MeanVC2 準拠）で学習
- **受け入れ基準**: UTTE 有効版が学習完了し、ゼロショット SECS が改善すること（目標: +0.05 以上）。
- **関連**: `crates/lightvc-core/src/converter.rs:234-332`, `training/converter.py:150-199`, `MODEL_TRAINING.md:295-296`

### [03-3] (P1) ✅ FlowConverter の ARCHITECTURE 反映
- **現状**: ARCHITECTURE.md §4（289-399 行）は Phase 1 `Converter`（residual-prediction）だけ記載。Phase C `FlowConverter` と `AnyConverter` enum による `model_type` 切替が未記載。
- **作業**:
  1. ARCHITECTURE.md §4 に「Phase C: FlowConverter」セクションを追加
  2. `forward_velocity`（学習時）と `convert`（1-NFE 推論時）のフローを図解
  3. `model_type` 切替（`AnyConverter::new`）の仕組みを記載
  4. TimeEmbed / CondMlp / BottleneckEncoder の役割を記載
- **受け入れ基準**: ARCHITECTURE.md を読めば `model_type` 切替の仕組みと FlowConverter の役割が分かること。
- **関連**: `crates/lightvc-core/src/converter.rs:608-731`, `MODEL_TRAINING.md:202-234`

### [03-4] (P2) ✅ CausalConv1d の dead code 削除
- **現状**: `converter.rs:100-105` に `conv.weight` / `conv.bias` の depthwise フォールバックがある。AGENTS.md「Known Issues」で depthwise conv (`groups=in_ch`) は XPU backward で失敗すると明記。Python 側 `converter.py:60-89` にも `depthwise=True` オプションがあるが、`CausalResBlock` は `depthwise=False`（標準 conv）で使用。
- **作業**:
  1. `CausalConv1d::new` の `or_else(|_| vb.get(..., "conv.weight"))` フォールバック削除
  2. Python 側 `CausalConv1d.__init__` の `depthwise` 引数と分岐削除
  3. AGENTS.md の Known Issues から depthwise 行を残す（他での再発防止のため）
- **受け入れ基準**: 到達不能コードが除去され、`cargo clippy --workspace` が通る。`ruff` / `mypy`（Python）も通る。
- **関連**: `crates/lightvc-core/src/converter.rs:99-112`, `training/converter.py:59-89`, `AGENTS.md` (Known Issues)

### [03-5] (P0) ✅ Snake1d の Rust/Python 数値一致
- **現状**:
  - Python (`converter.py:55-56`): `x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x).pow(2)`
  - Rust (`converter.rs:64-72`): `x + alpha.recip() * sin(alpha*x)^2`（epsilon なし）
  - `dac_model.rs:23-31` の `Snake1d` も同様に epsilon なし
- **影響**:
  - 学習済み Python モデルを Rust に持って来ると推論結果が厳密には一致しない
  - `alpha` が 0 に近い場合（初期値 `torch.ones`、学習で小さくなりうる）に差が顕在化
  - ゼロ除算の潜在的リスク（alpha が厳密に 0 になった場合）
- **作業**:
  1. Rust 側を `(alpha + 1e-9).recip()` に修正
  2. `converter.rs` と `dac_model.rs` の両方の `Snake1d::forward` を修正
  3. ユニットテスト追加: 既知の alpha / x で Rust / Python 出力が 1e-6 以内で一致することを確認
- **受け入れ基準**: 同一入力・同一重みで Rust / Python の出力が 1e-6 以内で一致。`cargo test` で確認。
- **関連**: `crates/lightvc-core/src/converter.rs:54-72`, `crates/lightvc-core/src/dac_model.rs:13-31`, `training/converter.py:48-56`

## 関連文書
- [02_streaming_lookahead.md](02_streaming_lookahead.md)
- [04_training_pipeline.md](04_training_pipeline.md)
- [08_known_bugs.md](08_known_bugs.md)
