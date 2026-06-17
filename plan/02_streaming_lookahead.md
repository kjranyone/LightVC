# 02: ストリーミング / Lookahead の乖離

> カテゴリ: B
> 関連資料: ARCHITECTURE.md §3, CONCEPT.md SOTAネタ3, RESEARCH.md §2 (MeanVC2)

## 概要

LightVC の差別化要件的に最重要な **bounded future context（MeanVC2 FRC）** と **per-layer conv-state caching** が実装されていない。現在の `streaming.rs` は入力テール（過去サンプル）のオーバーラップのみで、Strict / Balanced / Quality の違いはチャンクサイズのみ。lookahead 吸収がなく、設計が謳う「latency / quality トレードオフ」が成立していない。この問題は **Astrape 超え** の核心的セールスポイントの未達成を意味する。

## 現状の乖離

| 設計 | 実装 |
|---|---|
| FRC: layer 1 のみ未来参照、他は causal（RESEARCH §2 MeanVC2） | 未来参照なし |
| Strict=0ms / Balanced=~40ms / Quality=~80ms lookahead（ARCHITECTURE §3.5, DESIGN §2） | 全モード lookahead 実質 0ms |
| `conv_states: Vec<ConvState>` per-layer caching（ARCHITECTURE §3.4） | `input_tail: VecDeque<f32>` のみ（単一バッファ） |
| Overlap-add with cross-fade region（ARCHITECTURE §3.4） | 単純リニアクロスフェード（streaming.rs:153-181） |

## タスクリスト

### [02-1] (P0) ✅ FRC（Future-Receptive Chunking）の実装
- **現状**: `streaming.rs:114-143` の `encode_step` は過去 `ENCODER_OVERLAP = HOP*4` サンプルを prepend するだけ。未来サンプルを buffer する機構がないため、non-causal DAC エンコーダの受容野が満たされず、チャンク境界でアーティファクトが発生。
- **影響**:
  - 設計の「Strict / Balanced / Quality で品質が段階的に上がる」が実現しない
  - Quality モード（~80ms lookahead を想定）でも実際は lookahead 0 と同等の品質
  - MeanVC2 FRC（RESEARCH.md:50-55）の核心要素の未達成
- **作業**:
  1. `LatencyMode` ごとの lookahead サンプル数を定義
     - Strict: 0
     - Balanced: `0.040 * 44100 ≈ 1764` samples（`hop * 3.4`、`hop` 境界に揃えるなら `hop * 4 ≈ 2048`）
     - Quality: `0.080 * 44100 ≈ 3528` samples（`hop * 8 = 4096`）
  2. `StreamingCodec` に `lookahead_buf: VecDeque<f32>` を追加
  3. `encode_step` を「現在チャンク + lookahead 分を取り込んでからエンコード、新規フレームのみ出力」に変更
  4. デコーダ側も同様に future context を考慮
  5. `ChunkMode` と lookahead の組み合わせ表を ARCHITECTURE.md §3.5 に正確反映
- **受け入れ基準**:
  - Strict / Balanced / Quality で lookahead が仕様通り（ログ出力で確認）
  - チャンク境界アーティファクトが Quality モードで聴感上消える
  - ラウンドトリップテストで波形相関が Quality > Balanced > Strict の順
- **関連**: `crates/lightvc-core/src/streaming.rs:22-47, 114-143`, `ARCHITECTURE.md:219-285`, `CONCEPT.md:102-119`, `RESEARCH.md:50-55`

### [02-2] (P0) ✅ per-layer conv-state caching 実装
- **現状**: 入力 PCM の末尾のみキャッシュ。各エンコーダブロック（`ResidualUnit` の dilated conv）内部の状態は毎回ゼロから再計算され、チャンク先頭でエッジ効果が残る。
- **影響**:
  - 同じ PCM を streaming / non-streaming でエンコードすると、出力 latent が不一致
  - 推論精度の低下
- **作業**:
  1. ARCHITECTURE.md §3.4 の `ConvState` 構造体を実装
  2. `dac_model.rs` の各 `Snake1d` / `Conv1d` / `ConvTranspose1d` / `ResidualUnit` に state 保持機能を追加
  3. `forward_with_state(&input, &mut states)` API を追加（既存 `forward` は state なしのまま残し、ストリーミング時のみ state 版を使用）
  4. `StreamingCodec` で各層の state をキャッシュ
- **受け入れ基準**: 同一入力を streaming / non-streaming でエンコードした際、出力 latent が一致（誤差 1e-5 以内）。
- **関連**: `crates/lightvc-core/src/streaming.rs:49-59`, `crates/lightvc-core/src/dac_model.rs`, `ARCHITECTURE.md:240-275`

### [02-3] (P1) ✅ デコーダ overlap-add の設計文書化と検証
- **現状**: `streaming.rs:153-181` にリニアクロスフェード実装があるが、設計資料に詳細なし。クロスフェード長が `DAC_HOP_LENGTH`（512 sample = 約 11.6ms）固定で、チャンクサイズに依存しない。
- **作業**:
  1. 実装の根拠を ARCHITECTURE.md §3 に追記
  2. クロスフェード長をチャンクサイズ（`samples_per_chunk`）に比例させるか検討（例: Quality の 8 frame chunk ではより長いクロスフェードが自然）
  3. Quality モードで 8 frame chunk の場合のクロスフェード長を検証
- **受け入れ基準**: 設計文書に decoder overlap-add の記載があり、実装と一致。
- **関連**: `crates/lightvc-core/src/streaming.rs:145-181`

### [02-4] (P0) ✅ dac_model.rs 対称パディング ↔ causal 仮定の矛盾解消
- **現状**: 
  - `dac_model.rs:83-88` の `ResidualUnit`: `padding = ((7-1) * dilation) / 2`（両側パディング）
  - `dac_model.rs:138-141` の `EncoderBlock::conv1`: `padding = stride.div_ceil(2)`（両側）
  - `dac_model.rs:186-190` の `Encoder::conv1`: `padding: 3`（両側）
  - 一方 `streaming.rs:114-143` の `encode_step` は `input_tail`（過去）を prepend するだけの半 causal 挙動
- **影響**:
  - 対称パディングされた conv に「左側にしか context がない入力」が入る
  - 右側（未来）はゼロパディング相当となり、チャンク境界でエンコーダ出力がフルバッチ処理と大幅に不一致
  - 聴感上のチャンク境界アーティファクトの主要因（[08-1] と同一問題）
- **作業**:
  - **方法 A（推奨）**: [02-1] の lookahead 実装で未来側バッファを追加し、対称パディングと両立
  - **方法 B**: エンコーダ内部 conv を完全 causal（左パディングのみ）に改造。ただし学習済み DAC 重みと非互換になるため不可。
  - 方法 A を選択する場合、[02-1] [02-2] [02-4] は単一 PR で対応するのが自然
- **受け入れ基準**: ストリーミング出力がフルバッチ処理と一致（誤差 1e-4 以内）、または聴感上問題ないことが ABX テストで確認。
- **関連**: `crates/lightvc-core/src/dac_model.rs:78-99, 132-163, 183-220`, `crates/lightvc-core/src/streaming.rs:114-143`, [08_known_bugs.md](08_known_bugs.md) [08-1]

## 関連文書
- [01_crate_structure.md](01_crate_structure.md)
- [08_known_bugs.md](08_known_bugs.md)
