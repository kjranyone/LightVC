# AGENTS.md - Project Rules

## Environment

- **Python環境分離は必ず uv で行う**
- **conda は禁止**。`conda install` / `conda create` / `conda search` は一切使わない
- Intel加速は **IPEX (CPU) ではなく XPU (Intel GPU)** を使う
  - device は `xpu`
  - torch >= 2.6 は XPU が本体統合済み（別途IPEX不要）
- Rust の lint/typecheck: `cargo check --workspace` / `cargo clippy`

## Build Commands

```bash
# スタンドアロンアプリ
cargo build --release -p lightvc-app

# CLAP+VST3プラグインバンドル
cargo run -p lightvc-xtask -- bundle

# プラグインインストール
cargo run -p lightvc-xtask -- install

# ASIO有効ビルド（スタンドアロンのみ）
cargo build --release --features asio -p lightvc-app
```

## Training Commands

```bash
cd training && uv sync

# コーパスエンコード
uv run python encode_corpus.py --source /path/to/corpus --output data/latents

# Phase B: warm-start
uv run python train_warmstart.py --config configs/phase_b.yaml --data data/latents

# Phase C: flow matching
uv run python train_flow.py --config configs/phase_c.yaml --data data/latents

# エクスポート
uv run python export_weights.py --checkpoint checkpoints/phase_c/best.pt --output ../models/converter.safetensors
```

## Architecture Rules

- **推論は全て Rust (Candle)** — Pythonランタイム不要
- **学習は全て PyTorch (uv環境)** — Rustコードに依存しない
- **PyTorchのconverter.pyとRustのconverter.rsはキー名完全一致** — 変更時は両方更新
- **VC teacher蒸留は禁止** — 別VCモデルの変換音声を target とする synthetic parallel distillation は使わない
- **補助モデルによる表現監督は許可** — WavLM-SV/ECAPA 等を speaker loss・評価・SpeakerEncoder 蒸留に使ってよい。ただし推論時依存にしない
- **VC teacher不要** — target latent = 実音声または信号処理/同一内容ペア由来のDACエンコード

## Licensing

- **プロジェクト全体は MIT**
- **GPLv3依存は禁止**:
  - VST3出力は `clap-wrapper`（MIT）経由 — `vst3-sys`(GPLv3)は使わない
  - Steinberg VST3 SDK は2025年にMIT化済み
- **ASIO SDK** はプロプライエタリ（再配布禁止）— リポジトリにコミットしない
- 依存チェーン: lightvc-clap (MIT) → nice-plug (ISC) → clap-sys (MIT/Apache) / clap-wrapper (MIT)

## Code Style

- **Rust**: コメントは最小限（AGENTS.md/設計書に書く）
- **Python**: コメントなし、型ヒント推奨
- **コミットメッセージ**: 英語、`feat:` / `fix:` / `docs:` / `refactor:` プレフィックス

## Known Issues

- XPU backwardでdepthwise conv (groups=in_ch) が失敗する → 標準conv (groups=1) を使用
- XPU学習中にPCハングする場合あり → CPU学習またはバッチサイズ/フレーム長調整
- Windowsでsafetensors mmap drop時にプロセス終了が遅延 → `std::process::exit()` で対処済み
