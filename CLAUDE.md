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

- 目標は ASMR・官能バ美肉向けリアルタイムVC、Human-in-the-Loop Kansei評価。
- **E2E レイテンシ（`current/README.md` が正本）: 設計目標 p95 &lt; 30ms / 理想 20ms級 / p95 ≥ 50ms は失格上限**。50ms は合格ラインではない。「50ms未満ならよい」と読まない。
- 現行設計は `B4 Dual-Path Kansei VC`。
- DAC latentからspeaker-free contentを学習するStage1依存設計は不採用。
- contentはSSL/ASR系、prosodyは明示特徴、target voice/style/textureは別経路で扱う。
- GANは最後のtexture fine-tuneのみ。失敗したcontent表現をGANで救済しない。

## Shipping Gate (出荷前提) — 長時間学習を起動する前に必ず通す

**このゲートが無かったため、ストリーミングできない front-end の上で 55 時間（gvoc f0版 27.5h ＋ NHV版 27.5h）を消費した。** 二度とやらない。

- **1時間を超える学習を起動する前に `uv run python ship_check.py` 相当の台帳を出し、PASS を確認してからにする。** FAIL のまま起動しない。ログの先頭に台帳を残す。
- **合否は 2 条件**:
  1. **未来不変性** — 入力の t 以降を書き換えたとき、t 以前を担当する出力フレームが変化しないこと。**実音声プローブで測る**（白色雑音は argmax/閾値の離散性で偽 PASS を出す。実測済み: `harmonic_sum_f0` が雑音 1.81ms / 実音声 1882ms）。編集が出力を動かさない場合は INCONCLUSIVE ＝ PASS にしない。
  2. **静的遅延台帳** — framing 先読みの合計 ＋ content encoder 予備 10ms &lt; 30ms。
- **推論経路に発話全体の統計を置かない**（禁止例: 発話中央値による正規化、`amax()` による包絡正規化、発話 RMS によるレベル決定、発話中央値 f0 から倍音数を決める）。因果な走行推定か固定定数にする。これらは **front-end を直すとネットの入力分布が変わるので、重みの流用ができず学習し直しになる**。
- **centered framing は診断専用**（`ROADMAP.md` path A）。製品経路は左寄せ（`causal_mel.py`）。**左寄せ窓は n_fft をいくつにしても先読み 0** — 窓長は遅延ではなく過渡のにじみのツマミ（実測確認済み）。
- **ゲートを通らない構成で学習してよいのは、tag に `diag_` を付けて診断と明示した場合のみ。** その結果を製品判断・昇格の根拠に引用しない。
- **「あとで front-end だけ差し替える」は不可**と前提する。差し替えられるのは、ネットが見る値が変わらないと示せた場合だけ。

## Data (学習データ) — 必ずフルデータでフル学習

- **本番学習は必ずフルコーパスを使う**。`female-dataset`（実女性2775話者＝`data/female_real_feat`）と irodori-tts コーパス（`data/female_tts_corpus` 669話者、full encode すること）を**全量**利用する。男性ソースは `data/male_feat`。
- **少数話者・部分集合での学習は誤り**（例: `rcav_feat` の42話者だけ、極小overfit setだけ）。切り分け用の overfit gate を除き、**本番・比較・昇格の学習は full data 前提**。
- 新コーパスは学習前に content/f0/energy へ full encode し、`*_feat` を揃える。「まず小さく回す」で結論を出さない。

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

## Results Directory Convention

- **命名**: `diag_`=診断専用（部分集合・ゲート外構成。製品判定に引用しない）／`cart_`=cartridge FT／`v2X_`=V2 ラダー腕／無印=本命走行。`*_smoke` は動作確認のみ。
- **寿命**: smoke は当日中に削除。負け腕は判定記録後に best（+last）のみ残し中間 snap は削除。勝ち腕も採用確定後に snap を削除。
- **削除台帳**: 削除前に `results/DELETED_<date>.txt` に一覧を追記する（不可逆のため）。
- **走行中の run が参照するパス（ckpt・キャッシュ・データ）を消さない**。整理前に `pgrep -f train` で走行中ジョブを確認する。
- **例外**: RESEARCH.md が数値・パスで参照する証拠（耳軸測定・帰属 arm の wav/json）は削除しない。

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
