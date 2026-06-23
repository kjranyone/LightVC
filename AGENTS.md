# AGENTS.md - Project Rules

## Communication

- **AIエージェントは常に日本語で応答すること**

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

## GUI Review Protocol

AIエージェント（glm-5.2）は **画像分析MCP（`zai-mcp-server`）経由でGUIスクショを読める**。
ユーザーはアプリを開いたまま、AIにキャプチャを渡してレイアウトレビューを依頼できる。

### ワークフロー

**モードA: Watch（推奨・AIレビュー用）** — `.rs` 編集で自動再ビルド＋再起動＋スクショ更新

```powershell
# 1. watch ループ起動（debug ビルド＋デモモード、ターミナル占有）
.\dev.ps1 -Watch
# → 初回ビルド(11s) → 起動 → live.png 撮影
# → 以降 .rs 保存のたび: kill → build(~3s) → 起動 → snap（サイクル 7-10s）

# 2. AI が .rs を編集 → watch が自動でリロード → AI が live.png を読んでレビュー
#    （AI は snap.ps1 を手動実行する必要なし。Ctrl+C で watch 終了）
```

**モードB: ワンショット** — 手動ビルド＋起動。AI が snap.ps1 で再撮影

```powershell
# 1. アプリ起動（debug ビルド・デモ・自動スクショ付き）
.\dev.ps1
→ [3] を選ぶ（debug・高速・全タブのモックデータ入り）

# 2. AI に「スクショ撮って」と言われたら
#    AI が tools\snap.ps1 を直接実行して docs\screenshots\live.png を更新（新しい窓は開かない）
powershell -NoProfile -ExecutionPolicy Bypass -File tools\snap.ps1 -Out docs/screenshots/live.png

# 3. AIが画像分析ツールでスクショを読んでレイアウト指摘
#    → 修正 → 反復
```

### ビルドプロファイル

| モード | コマンド | 差分ビルド | 用途 |
|--------|----------|-----------|------|
| Debug + Demo | `[3]` or `-DebugBuild -Demo` | ~3s | UI レイアウトレビュー |
| Watch | `-Watch` | ~3s（自動） | AI 反復レビューループ |
| Release | `[1]` / `[2]` | 数分 | 実機推論・パフォーマンス検証 |

### AI側の使い方

```python
# スクショ読込（zai-mcp-server の画像分析ツール）
analyze_image(
    image_source="docs/screenshots/live.png",
    prompt="LightVC GUI。レイアウト問題（縦伸び、ボタン不揃い、余白偏り、コントラスト）を指摘"
)
```

### AI側の使い方

```python
# スクショ読込（zai-mcp-server の画像分析ツール）
analyze_image(
    image_source="docs/screenshots/live.png",
    prompt="LightVC GUI。レイアウト問題（縦伸び、ボタン不揃い、余白偏り、コントラスト）を指摘"
)
```

### 注意

- `dev.ps1` は **UTF-8 BOM + CRLF** が必要（PowerShell 5.x 仕様）。`.gitattributes` で `*.ps1 eol=crlf` を強制済み
- スクショは現在 **全画面キャプチャ**（ウィンドウ単体は P/Invoke が不安定のため）。背景に他ウィンドウが混入する点に留意
- `zai-mcp-server` は `opencode.jsonc` で `enabled: true` に設定済み（`~/.config/opencode/opencode.jsonc`）

