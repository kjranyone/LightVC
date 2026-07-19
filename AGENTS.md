# AGENTS.md - Project Rules

## Communication

- **AIエージェントは常に日本語で応答すること**

## Active Research Document

- 研究・学習・評価・設計作業では、必ず最初に `current/README.md`（採用＋設計索引）と `current/RESEARCH.md`（研究中）を読むこと。
- 現役資料は `current/` 配下: `README.md`＝正典（§1採用アルゴリズム／§2設計概観・索引, 安定）、`RESEARCH.md`＝研究中の仮説・反証実験・negative results（揮発）、`current/<network>.md`＝フル設計ネットワーク（1ネットワーク=1ファイル, status: PROPOSED/ADOPTED）。例: `current/vocoder.md`。
- 昇格フロー: RESEARCH（仮説・PROPOSED 設計）→ overfit gate / 人間の耳ゲート → 勝てば設計を ADOPTED 化し README §1採用が指す。RESEARCH→採用の直行禁止。採用は耳で勝ったものだけ（proxy 単独昇格禁止）。証拠=`results/`、横断=`memory/`（重複させずポインタ）。
- `.archive/`、旧checkpoint、旧training scriptは歴史的資料・失敗 artifact として扱い、現行設計の根拠にしない。
- 特に旧B1/B3 staged系、`train_stage1_content.py`、`train_stage2_generator.py`、`train_stage2_adv.py`、`train_b3.py` を現行経路として開始・継続しない。

## Current Direction

- 目標は ASMR・官能バ美肉向けリアルタイムVC、E2E 50ms未満、Human-in-the-Loop Kansei評価。
- 現行設計は `B4 Dual-Path Kansei VC`。
- DAC latentからspeaker-free contentを学習するStage1依存設計は不採用。
- contentはSSL/ASR系、prosodyは明示特徴、target voice/style/textureは別経路で扱う。
- GANは最後のtexture fine-tuneのみ。失敗したcontent表現をGANで救済しない。

## Environment

- **Python環境分離は必ず uv で行う**
- **conda は禁止**。`conda install` / `conda create` / `conda search` は一切使わない
- Intel加速は **IPEX (CPU) ではなく XPU (Intel GPU)** を使う
  - device は `xpu`
  - torch >= 2.6 は XPU が本体統合済み（別途IPEX不要）
- Rust の lint/typecheck: `cargo check --workspace` / `cargo clippy`

## Build Commands

```bash
cargo build --release -p lightvc-app
cargo run -p lightvc-xtask -- bundle
cargo run -p lightvc-xtask -- install
cargo build --release --features asio -p lightvc-app
```

## Architecture Rules

- **推論は全て Rust (Candle)** — Pythonランタイム不要
- **学習は全て PyTorch (uv環境)** — Rustコードに依存しない
- **PyTorchとRustの推論キー名は完全一致** — 変更時は両方更新
- **VC teacher蒸留は禁止** — 別VCモデルの変換音声を target とする synthetic parallel distillation は使わない
- **補助モデルによる表現監督は許可** — WavLM-SV/ECAPA/ASR 等をloss・評価・蒸留に使ってよい。ただし推論時依存にしない
- **VC teacher不要** — targetは実音声、同一内容ペア、または信号処理由来に限定

## Licensing

- **プロジェクト全体は MIT**
- **GPLv3依存は禁止**
- VST3出力は `clap-wrapper`（MIT）経由
- ASIO SDK はプロプライエタリ（再配布禁止）— リポジトリにコミットしない

## Code Style

- **Rust**: コメントは最小限
- **Python**: コメントなし、型ヒント推奨
- **コミットメッセージ**: 英語、`feat:` / `fix:` / `docs:` / `refactor:` プレフィックス

## Known Issues

- XPU backwardでdepthwise conv (groups=in_ch) が失敗する → 標準conv (groups=1) を使用
- XPU学習中にPCハングする場合あり → CPU学習またはバッチサイズ/フレーム長調整
- Windowsでsafetensors mmap drop時にプロセス終了が遅延 → `std::process::exit()` で対処済み

## GUI Review Protocol

GUIレビューが必要な場合のみ、archive済みGUI資料ではなく実コードとスクショを確認する。

```powershell
.\dev.ps1 -Watch
powershell -NoProfile -ExecutionPolicy Bypass -File tools\snap.ps1 -Out docs/screenshots/live.png
```
