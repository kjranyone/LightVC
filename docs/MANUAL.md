# LightVC 使い方ガイド

本ドキュメントでは、LightVCのインストールから学習、各種アプリケーションの使い方までを解説します。

---

## 目次

1. [概要](#1-概要)
2. [システム要件](#2-システム要件)
3. [インストール](#3-インストール)
4. [スタンドアロンアプリの使い方](#4-スタンドアロンアプリの使い方)
5. [CLAP/VST3プラグインの使い方](#5-clapvst3プラグインの使い方)
6. [モデルの学習](#6-モデルの学習)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. 概要

LightVCは、ニューラルコーデック（DAC）の潜在空間でリアルタイムに声質変換を行うシステムです。

**特徴:**
- 1ステップ推論（フローマッチング、ODEループなし）
- ゼロショット声質変換（5〜30秒の参照音声で任意の声を再現）
- 3つのレイテンシモード（Strict / Balanced / Quality）
- スタンドアロンアプリ、CLAPプラグイン、VST3プラグインの3形態
- 推論は全てRust（Candle）で動作、Pythonランタイム不要
- 学習はTeacher不要（フローマッチング直接学習）

---

## 2. システム要件

### 推論（Rustアプリ・プラグイン）

| 項目 | 要件 |
|------|------|
| OS | Windows 10/11、macOS 12+、Linux |
| CPU | x86_64、AVX2推奨 |
| GPU | オプション（CUDA / Metal / Intel XPU） |
| RAM | 512MB以上 |
| ストレージ | 500MB以上（モデル重み込み） |

### 学習（Pythonパイプライン）

| 項目 | 要件 |
|------|------|
| Python | 3.10〜3.12 |
| パッケージ管理 | [uv](https://github.com/astral-sh/uv) |
| GPU | Intel Arc GPU（XPU）または NVIDIA GPU（CUDA） |
| RAM | 16GB以上推奨 |

---

## 3. インストール

### 3.1 ビルド

```bash
# リポジトリをクローン
git clone https://github.com/kjranyone/LightVC.git
cd LightVC

# スタンドアロンアプリをビルド
cargo build --release -p lightvc-app

# CLAP/VST3プラグインをビルド＆バンドル生成
cargo run -p lightvc-xtask -- bundle
# → target/bundled/LightVC.vst3 と LightVC.clap が生成される

# システムのプラグインディレクトリにインストール
cargo run -p lightvc-xtask -- install
```

### 3.2 モデル重みの準備

```bash
# DAC重みをダウンロード
mkdir -p models
curl -L -o models/dac_44khz.safetensors \
  "https://huggingface.co/descript/dac_44khz/resolve/main/model.safetensors"
```

学習済みコンバーター重み（`converter.safetensors`）は [モデルの学習](#6-モデルの学習) で生成するか、配布されているものを `models/` に配置します。

### 3.3 ASIO対応（Windows、オプション）

低遅延（5ms未満）が必要な場合のみASIO SDKを導入します。詳細は [docs/ASIO_SETUP.md](ASIO_SETUP.md) を参照してください。

```bash
# ASIO SDKを配置後
export CPAL_ASIO_DIR="C:/ASIOSDK"
cargo build --release --features asio -p lightvc-app
```

---

## 4. スタンドアロンアプリの使い方

### 4.1 起動

```bash
./target/release/lightvc-app gui --dac-weights models/dac_44khz.safetensors
```

### 4.2 GUI画面構成

アプリは3つのタブで構成されています。

#### タブ1: Offline（オフライン変換）

ファイルベースの声質変換を行います。

1. **Source** — 変換元の音声ファイル（WAV）を指定
2. **Reference** — 変換先の声の参照音声（WAV）を指定
3. **Convert** — 変換を実行
4. **Output** — 変換結果を再生・保存

```
[Source 音声] + [参照音声] → [変換] → [出力音声]
```

#### タブ2: Realtime（リアルタイム変換）

マイク入力をリアルタイムに変換してスピーカーに出力します。

1. **モデルロード** — Converter重みとConfigを指定して「Load Converter」をクリック
2. **Start / Stop** — 「▶ Start」でリアルタイム変換を開始、「■ Stop」で停止
3. **ステータス** — 現在の状態を表示: `● LIVE`（変換中）/ `BYPASS`（バイパス中）/ `STOPPED`（停止中）
4. **モード切替** — Strict（0ms）/ Balanced（~40ms）/ Quality（~80ms）をノブで切替
5. **Bypass** — 変換をバイパス（入力をそのまま出力）
6. **レベルメーター** — 入出力の音量を可視化
7. **メトリクス** — エンドツーエンドレイテンシ（ms）、RTF（リアルタイムファクター）、xrun（オーバーラン/アンダーラン）カウントを表示
8. **Audio Devices** — 認識されている入出力オーディオデバイスの一覧（折りたたみセクション）

#### タブ3: Voices（ボイスカタログ）

ゼロショット変換用の参照音声を管理します。

1. **Add Voice** — 名前とWAVファイルを指定してボイスを登録
2. **Play** — 登録したボイスを試聴
3. **Remove** — ボイスを削除
4. **Import/Export** — JSON形式でボイスリストを入出力

### 4.3 CLIサブコマンド

GUI以外にコマンドラインツールも利用可能です。

```bash
# DACラウンドトリップテスト（符号化→復号の品質確認）
lightvc-app roundtrip \
  -i input.wav -o output.wav \
  --dac-weights models/dac_44khz.safetensors

# オフライン変換
lightvc-app convert \
  -i source.wav -r reference.wav -o converted.wav \
  --dac-weights models/dac_44khz.safetensors \
  --converter-weights models/converter.safetensors \
  --converter-config models/converter_config.json
```

---

## 5. CLAP/VST3プラグインの使い方

### 5.1 インストール

```bash
cargo run -p lightvc-xtask -- install
```

自動的に以下のディレクトリに配置されます。

| プラットフォーム | VST3 | CLAP |
|-----------------|------|------|
| Windows | `C:\Program Files\Common Files\VST3\` | `%LOCALAPPDATA%\Programs\Common\CLAP\` |
| macOS | `~/Library/Audio/Plug-Ins/VST3/` | `~/Library/Audio/Plug-Ins/CLAP/` |
| Linux | `~/.vst3/` | `~/.clap/` |

### 5.2 DAWでの使用

1. DAW（REAPER、Bitwig、Ableton Live等）を起動
2. プラグインブラウザから「LightVC」を検索
3. オーディオトラックに挿入
4. プラグインエディタを開く

### 5.3 プラグインパラメータ

| パラメータ | 範囲 | 説明 |
|-----------|------|------|
| Bypass | On/Off | 変換をバイパス |
| Mode | 0-2 | 0=Strict, 1=Balanced, 2=Quality |
| Mix | 0-100% | ドライ/ウェットミックス |
| Output | -24〜+24dB | 出力ゲイン |

### 5.4 モデルのロード

プラグインは起動時にConverter重みとDAC重みを読み込みます。重みのパスはパラメータとして永続化されるため、一度設定すればDAWを再起動しても保持されます。

---

## 6. モデルの学習

### 6.1 環境セットアップ

```bash
cd training
uv sync
```

### 6.2 学習パイプライン

LightVCの学習は3フェーズで構成されています。Teacher不要です。

```
Phase A: コーパスエンコード（実音声 → DAC潜在表現）
    ↓
Phase B: ウォームスタート（ボトルネック自己符号化器）
    ↓
Phase C: フローマッチング（本工学習）
```

#### Phase A: コーパスエンコード

```bash
# 多話者音声コーパスをDAC潜在表現にエンコード
uv run python encode_corpus.py \
    --source /path/to/speech_corpus \
    --output data/latents
```

音声コーパスの要件:
- 100話者以上推奨（ゼロショット汎化のため）
- 話者IDごとにディレクトリが分かれていること
- 16kHz以上のサンプリングレート
- LibriTTS、VCTK等の公開コーパスを推奨

#### Phase B: ウォームスタート

```bash
uv run python train_warmstart.py \
    --config configs/phase_b.yaml \
    --data data/latents \
    --output checkpoints/phase_b
```

ボトルネック自己符号化器でDAC潜在表現の取り扱いを学習します。情報ボトルネックにより話者情報を意図的に落とし、参照音声から話者情報を補完する構造を獲得します。

#### Phase C: フローマッチング

```bash
uv run python train_flow.py \
    --config configs/phase_c.yaml \
    --data data/latents \
    --output checkpoints/phase_c
```

メインの学習フェーズです。平均フローマッチング（mean-flow matching）により、1ステップ推論を可能にする変換器を学習します。ターゲットは実際の話者のDAC潜在表現（Teacherの出力ではない）です。

### 6.3 重みのエクスポート

```bash
uv run python export_weights.py \
    --checkpoint checkpoints/phase_c/best.pt \
    --output ../models/converter.safetensors \
    --model-type flow
```

### 6.4 学習データについて

**コーパス音声が必須ではありません。** 以下の条件を満たす音声データであれば利用可能です。

- 複数話者の音声（最低100話者以上推奨）
- 話者IDが分離できていること
- 1話者あたり最低2発話以上
- 比較的クリーンな音声（極端なノイズ・BGMなし）

独自の録音データ、ポッドキャスト、YouTube音声等も利用可能です。

### 6.5 timbre_shifter（データ拡張）

LightVCはTeacher不要ですが、Seed-VCの信号処理拡張（timbre shifter）を借用しています。これはピッチとフォルマントを摂動することで、変換器がソース音声の話者情報をコピーすることを防ぎ、参照音声からの話者条件付けに依存するように訓練します。

```python
from timbre_shifter import timbre_shift
shifted_wav = timbre_shift(wav, sr, apply_prob=0.5)
```

### 6.6 Edge TTSコーパス生成（クイックスタート）

実音声コーパスがない場合、Edge TTSで合成音声コーパスを生成できます。

```bash
uv run python generate_tts_corpus.py
# → 17話者、170発話のTTSコーパスが生成される
```

---

## 7. トラブルシューティング

### 音が出ない

- **モデルがロードされているか確認** — Realtimeタブのステータスが「● LIVE」になっているか（停止中は「STOPPED」）
- **Bypassがオンになっていないか確認** — 「BYPASS ON」の場合は入力がそのまま出力される
- **オーディオデバイスを確認** — Realtimeタブの「Audio Devices」で入出力デバイスが認識されているか

### 変換品質が悪い

- **学習ステップ数** — Phase Cは最低100Kステップ推奨。スモークテスト（1K〜3Kステップ）では品質が不十分
- **データ多様性** — 話者数が少ないとゼロショット汎化が弱くなる
- **モード** — Qualityモード（~80ms lookahead）が最高品質

### XPU学習でPCがハングする

Intel Arc GPUで学習中にハングする場合:
- `groups=in_ch`（depthwise conv）のbackwardに失敗する → 標準conv（groups=1）を使用
- バッチサイズを下げる（4→2）
- フレーム長を短くする（400→200）

### Windowsでプロセスが終了しない

safetensorsのmmap drop時にプロセス終了が遅延する問題があります。`std::process::exit()`で対処済みですが、Ctrl+Cで強制終了しても問題ありません。

### プラグインがDAWで認識されない

- `cargo run -p lightvc-xtask -- install` で正しいディレクトリに配置されたか確認
- DAWを再起動
- プラグインキャッシュをクリア（DAWの設定から）

### CLAPプラグインが読み込めないエラー

CLAPネイティブ対応DAW（REAPER、Bitwig等）では直接読み込み可能です。VST3のみ対応のDAW（Ableton Live等）では、clap-wrapper経由でVST3として読み込まれます。

---

## 参考ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| [DESIGN.md](../DESIGN.md) | 設計の全体像と根拠 |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | システムアーキテクチャ詳細 |
| [MODEL_TRAINING.md](../MODEL_TRAINING.md) | 学習パラダイムの技術解説 |
| [RESEARCH.md](../RESEARCH.md) | 文献調査と技術選定の根拠 |
| [docs/ASIO_SETUP.md](ASIO_SETUP.md) | ASIO SDK設定手順 |
| [docs/ASSETS_SPEC.md](ASSETS_SPEC.md) | ビジュアルアセット仕様書 |
