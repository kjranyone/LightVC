# 03: コンバータモデルの乖離

> カテゴリ: C
> 関連資料: DESIGN.md §3, §4, ARCHITECTURE.md §4, MODEL_TRAINING.md C.2-C.4

## 概要

コンバータのアーキテクチャ実装は概ね設計通りだが、パラメータ数、Phase 2 UTTE の状態、FlowConverter の文書化、細かい数値整合性に乖離がある。Rust/Python 間の数値不一致は学習済み重みの移植性を損なう。

## 現状の乖離

| 項目 | 設計 | 実装 |
|---|---|---|
| `hidden_dim` | 1024（DESIGN §4.1, README） | smoke=256 / 本番=1024（[03-1] で解消） |
| パラメータ数 | 8-12M / Phase2 15-30M | ~6M（Phase 1 smoke）/ ~12M（本番）|
| Phase 2 UTTE | K=32 token bank + cross-attn | `enable_timbre: false`（smoke）、本番 `true`（[03-2] 未検証）|
| FlowConverter | ARCHITECTURE §4.1b に反映済み | 実装あり（`model_type` 切替） |
| CausalConv1d keys | 標準 conv のみ | depthwise は削除済み（[03-4]） |
| Snake1d 数式 | `1/(alpha+eps) * sin^2(alpha*x)` | Rust/Python 共に epsilon=1e-9（[03-5] で解消） |
| **CrossAttnBlock マルチヘッド reshape** | Python: `reshape(B,T,H,d).transpose(1,2)` | **Rust: 直接 `(B*H,T,d)` へ reshape（数学的誤り）** ⚠️ [03-6] |
| **warm-start Converter の BottleneckEncoder** | Python は `self.bottleneck` 適用 | **Rust `Converter` に bottleneck フィールド不在** ⚠️ [03-7] |
| **GELU 種別** | Python: erf 精密値 | **Rust `.gelu()`: tanh 近似**（~3e-4 誤差）⚠️ [03-8] |
| **velocity_scale** | ARCHITECTURE「Default 2.5」 | Rust 1.0 / Python はパラメータ自体不在 ⚠️ [03-9] |

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

### [03-6] (P0) ✅ CrossAttnBlock マルチヘッド reshape の数学的誤り（本番 UTTE で出力破壊）
- **現状**: Rust `CrossAttnBlock::forward` がヘッド分割を以下のように実装:
  ```rust
  // converter.rs:335-338
  let q = q.reshape((b * self.n_heads, t, head_dim))?;   // ❌ 誤り
  let k = k.reshape((b * self.n_heads, n_tok, head_dim))?;
  let v = v.reshape((b * self.n_heads, n_tok, head_dim))?;
  ```
  入力 q は `[B, T, attn_dim]`（row-major）。これを直接 `[B*H, T, head_dim]` へ reshape すると、メモリ再解釈により `q_rust[b*H+h, t, d]` が意図した `q_src[b, t, h*head_dim+d]` と一致しない（`T==1` または `n_heads==1` の場合を除く）。
- **Python 正解**: `converter.py:186-188`
  ```python
  q = self.q(z_t).reshape(B, T, self.n_heads, head_dim).transpose(1, 2)
  k = self.k(keys).reshape(B, -1, self.n_heads, head_dim).transpose(1, 2)
  v = self.v(vals).reshape(B, -1, self.n_heads, head_dim).transpose(1, 2)
  ```
- **影響**:
  - `enable_timbre: true`（`phase_c.yaml:11` 本番設定・`phase_b.yaml`）で Rust 推論が**数値的に破壊**
  - `n_attn_heads=8` なので T>1 の全ケースで誤り
  - 重みは正しくロードされても注意力計算で誤った要素の積和を取るため、UTTE の cross-attention 出力が無意味化
- **作業**: Rust `CrossAttnBlock::forward` を以下に修正:
  ```rust
  let q = q.reshape((b, t, self.n_heads, head_dim))?
      .permute((0, 2, 1, 3))?
      .reshape((b * self.n_heads, t, head_dim))?;
  // k, v も同様に (b, n_tok, n_heads, head_dim) → permute(0,2,1,3) → reshape
  ```
  出力側も `(b, t, self.attn_dim)` への reshape 前に `(b, n_heads, t, head_dim) → (b, t, n_heads, head_dim)` の逆置換が必要。
- **受け入れ基準**: 同一入力・同一重みで Rust / Python の CrossAttnBlock 出力が 1e-6 以内で一致。`enable_timbre: true` でエンドツーエンド変換結果が一致。
- **関連**: `crates/lightvc-core/src/converter.rs:326-352`, `training/converter.py:180-195`

### [03-7] (P0) ✅ warm-start Converter に BottleneckEncoder が未実装（Rust）
- **現状**:
  - Python `Converter.__init__` (`converter.py:264`): `self.bottleneck = BottleneckEncoder(D, config.bottleneck_dim)`
  - Python `Converter.forward` (`converter.py:290`): `content = self.bottleneck(src_latent); z = self.film(content, ...)`
  - Python `Converter.content_code` (`converter.py:309-310`): `train_warmstart.py:228-229` が呼ぶ
  - Rust `Converter` 構造体 (`converter.rs:429-481`): `bottleneck` フィールド**不在**
  - Rust `Converter::forward` (`converter.rs:508`): `self.film.forward(&src_latent, ...)` — 生 latent を直接 FiLM へ
- **影響**:
  1. `export_weights.py` が `bottleneck.down.conv.{weight,bias}`, `bottleneck.act.alpha`, `bottleneck.up.conv.{weight,bias}`（5キー）を書き出すが、Rust は**黙って無視**（VarBuilder の未使用キー）
  2. Rust warm-start 推論が Python と一致しない（bottleneck の 1024→256→Snake→1024 変換をスキップ）
  3. DESIGN.md §4.1 / ARCHITECTURE.md §4.1b / MODEL_TRAINING.md §B.2 は「bottleneck autoencoder」を明記 → Python が設計正しく Rust が乖離
- **作業**: Rust `Converter` に `bottleneck: BottleneckEncoder` フィールドを追加し、forward で `content = bottleneck.forward(&src_latent)` を適用してから FiLM へ。`content_code()` メソッドも追加。`flow_converter.rs` の `BottleneckEncoder` を再利用可能。
- **受け入れ基準**: warm-start checkpoint を Rust にロードし、Python `Converter.forward` と出力が 1e-5 以内で一致。`bottleneck.*` 5キーが欠落警告なくロードされる。
- **関連**: `crates/lightvc-core/src/converter.rs:429-537`, `training/converter.py:251-310`, `DESIGN.md:60,134-135`, `ARCHITECTURE.md:434`

### [03-8] (P1) ✅ GELU の tanh 近似 vs erf 精密値（3箇所）
- **現状**: Candle 0.10 の `Tensor::gelu()` は **tanh 近似** `0.5x(1+tanh(√(2/π)(x+0.044715x³)))`。erf 精密値は `Tensor::gelu_erf()`。Python の `F.gelu` / `nn.GELU` はデフォルト `approximate='none'`（erf 精密値）。3箇所で不一致:
  1. `converter.rs:244` SpeakerEncoder: `h.gelu()?` → `h.gelu_erf()?` に修正
  2. `flow_converter.rs:77` TimeEmbed mlp: gelu → gelu_erf
  3. `flow_converter.rs:105` CondMlp: gelu → gelu_erf
- **影響**: `x ≈ -2` 付近で最大 ~3e-4 の誤差。学習済み重みの Rust 移植で bit-exact 性が損なわれる。speaker embedding・time embedding・条件付けの全てに影響。
- **作業**: 3箇所を `.gelu_erf()` に変更。`cargo test` で Python との一致（1e-6 以内）を確認。
- **受け入れ基準**: 同一入力で Rust / Python の GELU 出力が 1e-6 以内で一致。
- **関連**: `crates/lightvc-core/src/converter.rs:244`, `crates/lightvc-core/src/flow_converter.rs:77,105`, `training/converter.py:142,235,348`

### [03-9] (P1) ✅ velocity_scale API 非対称 + ARCHITECTURE「Default 2.5」誤記
- **現状**:
  - Rust `FlowConverter::convert` (`flow_converter.rs:232-255`): `velocity_scale: f64` 引数を取り `v.affine(velocity_scale, 0.0)` で乗算
  - Python `FlowConverter.convert` (`converter.py:419-443`): そのようなパラメータ**不在**、`infer_flow.py:105` も2引数で呼出
  - `pipeline.rs:79`: `velocity_scale: 1.0`（デフォルト）
  - `cli.rs:87`: `default_value = "1.0"`
  - `ARCHITECTURE.md:465`: **「Default 2.5」** と明記（コードと矛盾）
- **影響**:
  - デフォルト 1.0 では Rust と Python は一致するが、ARCHITECTURE に従って 2.5 を設定すると学習時（Python）と推論時（Rust）で不一致
  - API 非対称: Python には velocity_scale を表現する手段がない
- **作業**: 以下いずれか:
  - **(a) 推奨**: ARCHITECTURE.md:465 を「Default 1.0」に修正（classifier-free guidance 的用途は将来拡張として残す）。Python `convert` にもオプション引数 `velocity_scale: float = 1.0` を追加し対称化
  - **(b)**: デフォルトを本当に 2.5 にする（Python にも実装）。ただし 2.5 は学習目標から外れるため出力品質に影響
- **受け入れ基準**: ARCHITECTURE.md とコードのデフォルトが一致。Rust/Python 双方で velocity_scale が制御可能。
- **関連**: `crates/lightvc-core/src/flow_converter.rs:232-255`, `crates/lightvc-core/src/pipeline.rs:54,79`, `crates/lightvc-app/src/cli.rs:87-88`, `ARCHITECTURE.md:462-467`, `training/converter.py:419-443`

### [03-10] (P2) ✅ warm-start Converter の unbatched 対応非対称 + SpeakerEncoder ガード
- **現状**:
  - Rust `Converter::forward` (`converter.rs:499-505`): `was_unbatched = src_latent.rank() == 2` で unsqueeze 対応あり
  - Python `Converter.forward` (`converter.py:285-304`): そのような対応なし、`[D, T]` 入力で shape エラー
  - plan [08-6] は FlowConverter.convert のみ対応。warm-start 側は未対応（NEW-D4）
  - `SpeakerEncoder::forward` (`converter.rs:238`): `let n = (t_len as f64).max(1.0);` ガードだが、`t_len==1` で `n=1.0` → divisor `0.0` → inf/nan。実は PyTorch `std(unbiased=True)` も nan なので数値は一致するが、ガードが実質 no-op（NEW-D6）
- **影響**: 軽微。本番は FlowConverter のため実害なし。ただし Python/Rust API 対称性と可読性の観点で要対応。
- **作業**:
  1. Python `Converter.forward` に `was_unbatched = src_latent.ndim == 2` 対応を追加（FlowConverter と統一）
  2. Rust `SpeakerEncoder` のガードを `t_len <= 1` のとき `var = zeros` を返すよう実効化、またはコメントを「PyTorch 挙動に合わせて nan になるのが正しい」に修正
- **受け入れ基準**: Rust/Python で unbatched 入力挙動が一致。ガードの意図と実装が整合。
- **関連**: `crates/lightvc-core/src/converter.rs:237-241,499-505`, `training/converter.py:140-143,285-304`

## 関連文書
- [02_streaming_lookahead.md](02_streaming_lookahead.md)
- [04_training_pipeline.md](04_training_pipeline.md)
- [08_known_bugs.md](08_known_bugs.md)
