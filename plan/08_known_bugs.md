# 08: 既知のバグ・潜在的問題

> カテゴリ: H
> 関連資料: AGENTS.md (Known Issues), ARCHITECTURE.md §3, §8

## 機能不在ではなく、現状実装に潜む数値的・論理的バグ。動作はするが品質または正確性を損なうもの。

## バグリスト

### [08-1] (P0) ✅ dac_model.rs の対称パディング ↔ streaming.rs の causal 仮定の矛盾
- **現状**:
  - `dac_model.rs:83-88` (ResidualUnit): `pad1 = ((7-1) * dilation) / 2`（両側パディング）
  - `dac_model.rs:138-141` (EncoderBlock): `padding: stride.div_ceil(2)`（両側）
  - `dac_model.rs:186-190` (Encoder conv1): `padding: 3`（両側）
  - `dac_model.rs:202-211` (Encoder conv2): `padding: 1`（両側）
  - 一方 `streaming.rs:114-143` の `encode_step` は `input_tail`（過去サンプル）のみ prepend し、未来サンプルを補填しない
- **影響**:
  - 対称パディングされた conv に「左側にしか context がない入力」が入る
  - 右側（未来）はゼロパディング相当となり、チャンク境界でエンコーダ出力がフルバッチ処理と大幅に不一致
  - 聴感上のチャンク境界アーティファクトの主要因
- **作業**: [02-1] [02-4] に集約。lookahead バッファを導入して未来 context を与える。
- **受け入れ基準**: ストリーミングエンコード結果がフルバッチと一致（誤差 1e-4 以内）。
- **関連**: `crates/lightvc-core/src/dac_model.rs:78-99, 132-163, 183-220`, `crates/lightvc-core/src/streaming.rs:114-143`, [02_streaming_lookahead.md](02_streaming_lookahead.md) [02-4]

### [08-2] (P1) ✅ realtime_tab のリサンプリング状態ドリフト（再掲）
- **詳細**: [05_audio_io.md の 05-4](05_audio_io.md) を参照。本ファイルでは「バグ」として明示。
- **影響**: 長時間実行でピッチシフト・位相ズレが蓄積する可能性。

### [08-3] (P1) ✅ CLAP が `model_type` を見ないため FlowConverter 重みが読めない（再掲）
- **詳細**: [06_plugin_app.md の 06-1](06_plugin_app.md) を参照。Phase C 本命モデルが DAW で読み込めない重大問題。

### [08-4] (P1) ✅ `SpeakerEncoder` の活性化関数不一致（GELU vs ReLU）
- **現状**:
  - Python (`converter.py:147`): `F.gelu(self.p1(pooled))`
  - Rust (`converter.rs:227`): `self.proj1.forward(&pooled)?.relu()?`
- **影響**:
  - speaker embedding の数値が厳密一致しない
  - ゼロショット VC の話者類似度（SECS）に影響
  - GELU と ReLU は負入力の振る舞いが大きく異なるため、無視できない差
- **作業**: Rust 側を `candle_nn::ops::gelu` に変更。
- **受け入れ基準**: Rust / Python の speaker embedding 出力が 1e-6 以内で一致。
- **関連**: `crates/lightvc-core/src/converter.rs:217-229`, `training/converter.py:136-147`

### [08-5] (P2) ✅ `TimeEmbed` の freqs 初期化精度の差
- **現状**:
  - Python (`converter.py:233-235`): `1.0 / (10000 ** (arange(0, half) / half))` — `torch.arange` は float32
  - Rust (`converter.rs:546-549`): `(1.0f32 / 10000.0f32).powf(i as f32 / half as f32)`、`i` は `0..half`
- **影響**:
  - 数式は等価だが、`pow(double, double)` と `powf(float, float)` の精度差、および `10000 ** x` が PyTorch では double で計算されてから float にキャストされる可能性、で微小に異なる
  - 実用上は無視できるレベル（< 1e-6）だが、厳密一致を狙うなら対処
- **作業**: 必要に応じて `f64::powf` で計算後に `as f32` キャスト。
  ```rust
  let freqs_data: Vec<f32> = (0..half)
      .map(|i| (1.0f64 / 10000.0f64.powf(i as f64 / half as f64)) as f32)
      .collect();
  ```
- **受け入れ基準**: Rust / Python の time_embed 出力が 1e-7 以内で一致。
- **関連**: `crates/lightvc-core/src/converter.rs:543-549`, `training/converter.py:228-236`

### [08-6] (P2) ✅ `SpeakerEncoder.forward` の unbatched 入力扱い
- **現状**: Rust `converter.rs:223-228` は `ref_latent.mean(D::Minus1)`。入力 `[B, D, T_ref]` を T_ref 方向に平均 → `[B, D]`。Python も `mean(dim=-1)` で `[B, D]`。batched では一致する。
- **注記**: 一見問題ないが、`ref_latent` が unbatched `[D, T]` で渡された場合の挙動が Rust / Python で異なる:
  - Python (`converter.py:427-451`): `FlowConverter.convert` で `was_unbatched = z_src.ndim == 2` をチェックし、必要に応じて unsqueeze
  - Rust: そのようなチェックがなく、必ず batched を仮定
  - CLI の `pipeline.set_target` は `[1, D, T]` を返すので問題ないが、`Converter::forward` を直接呼ぶ場合は注意
- **作業**: 
  - オプション A: Rust 側の `forward` / `convert` で入力ランクをチェックし、3D でなければ unsqueeze
  - オプション B: ドキュメントに batch 次元必須と明記
- **受け入れ基準**: unbatched 入力時の挙動が Rust / Python で一致、または明文化。
- **関連**: `crates/lightvc-core/src/converter.rs:461-471, 720-731`, `training/converter.py:427-451`

### [08-7] (P1) ✅ 推論スレッドでのゼロ埋め過多
- **現状**: `realtime_tab.rs:389-395` で capture 不足時に `cap.resize(needed, 0.0)`。CPAL バッファサイズに対して chunk サイズが大きいと頻繁にゼロ埋めが発生し、出力に無音区間が混入。
- **影響**:
  - 動作はするが、聴感上「プツプツ」と無音が混入
  - リスニング体験を著しく損なう
- **作業**: [05-4] のリファクタリングで解消。`chunk_sz` と device buffer size の最小公倍数調整、または必要量溜まるまで待機。
- **関連**: `crates/lightvc-app/src/realtime_tab.rs:382-395`, [05_audio_io.md](05_audio_io.md) [05-4]
- **注記**: realtime_tab のみ解消。CLAP 側 `lib.rs:553-559` は残存 → [06-9] で対応。

### [08-8] (P2) ✅ ドキュメント内部矛盾・虚構型名（5 件）
- **現状**: 設計資料自体が実装と合わない、または内部で矛盾する項目:

  1. **rubato 型名が虚構** (`ARCHITECTURE.md:631-633`, plan/05 line 14,28):
     - 文書: `AsyncFixedIn<f32>` / `AsyncFixedOut<f32>`
     - 実体: rubato 3.0 にはこれらの型は**存在しない**。正しくは `Async<T>` + `FixedAsync::{Input,Output}` enum（`rubato-3.0.0/src/asynchro.rs:96`, `lib.rs:57` で確認）
     - コードは正しい、ドキュメントが誤り

  2. **ARCH §1.3 と §3.5 のレイテンシ数値矛盾**:
     - §1.3: Quality「Algorithmic lookahead (mode) ~80 ms」「Total (quality mode, 80ms) ~141 ms」
     - §3.5: Quality「Total algorithmic latency ~186 ms」（chunk 4096 + lookahead 93ms）
     - 実装（`pipeline.rs:218-220`）は §3.5 に従い、`realtime_tab.rs:518` は ~212ms を表示
     - §1.3 の 141ms は古い、または lookahead を chunk 含まずに算出した古い設計

  3. **モードノブラベルが lookahead を過小表示** (`realtime_tab.rs:194-198`):
     - 表示: 「Balanced = ~40ms lookahead」「Quality = ~80ms lookahead」
     - 実装（ARCH §3.5）: Balanced 2048 sample ≈ 46ms、Quality 4096 sample ≈ 93ms
     - ~13-14% 過小表示。同一タブ内の `latency_ms` 表示（正確）とも不一致

  4. **`widgets.rs` 構造が ARCH §2 と不一致**:
     - ARCH §2（105-110行）: `widgets/{level_meter.rs, device_combo.rs, param_knob.rs}` を約束
     - 実体: 単一 `widgets.rs`（`pub fn rms(...)` のみ 4 行）。`device_combo` はリポジトリ全体に存在しない（[05-6] のデバイス選択 UI 欠如と一致）

  5. **CLAP bypass のレイテンシ補償不整合** (`lib.rs:462-470`):
     - bypass 時は即時ドライパススルー（リング遅延ゼロ）
     - ウェット経路は ring buffer + 報告レイテンシ `chunk_44k*3`（~6144 sample）
     - bypass トグルで可聴ジャンプ（ドライが ~6144 sample 早く到着）
     - ほとんどの DAW は bypass でも報告レイテンシを維持することを期待

- **作業**:
  1. ARCH §5.2 と plan/05 を `Async<T>` + `FixedAsync` に修正
  2. ARCH §1.3 を §3.5 と整合（Quality ~186ms algorithmic / ~212ms total）に修正、または §1.3 削除
  3. `realtime_tab.rs:194-198` のラベルを「~46ms」「~93ms」に修正
  4. ARCH §2 の `widgets/` サブモジュール記載を実態（単一 `widgets.rs`）に合わせる、または [05-6] で `device_combo` を実装してから整合
  5. CLAP bypass を遅延一致型（ドライにも報告レイテンシ分の遅延を挿入）に変更
- **受け入れ基準**: ARCHITECTURE / plan の記述が実装と完全一致。ラベルと実測値が一致。
- **関連**: `ARCHITECTURE.md:52-67,105-110,367-383,631-633`, `crates/lightvc-app/src/realtime_tab.rs:194-198`, `crates/lightvc-app/src/widgets.rs`, `crates/lightvc-clap/src/lib.rs:462-470`, `plan/05_audio_io.md`

## AGENTS.md 既知問題（参考、対応不要）

これらは AGENTS.md に既に記載されており、対処済みまたは回避策運用中:

- **XPU backward で depthwise conv 失敗** → 標準 conv 使用（[03-4] で関連整理）
- **XPU 学習中 PC ハング** → バッチサイズ / フレーム長調整で運用回避
- **Windows で safetensors mmap drop 遅延** → `std::process::exit()` で対処済み

## 関連文書
- [02_streaming_lookahead.md](02_streaming_lookahead.md)
- [05_audio_io.md](05_audio_io.md)
- [06_plugin_app.md](06_plugin_app.md)
