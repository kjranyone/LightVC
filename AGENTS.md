# AGENTS.md - Project Rules

## Communication

- **AIエージェントは常に日本語で応答すること**

## Active Research Document

- 研究・学習・評価・設計作業では、必ず最初に `current/README.md`（製品要件＋設計索引）と `current/RESEARCH.md`（研究中）を読むこと。UI 変更時は `DESIGN.md` も読むこと。
- **研究の正本は `current/paper_snv.md`**（2026-11-06〜）。オーナー方針(11-06)「では論文からのみ学習／評価を行うこと」: 学習・評価は論文の段と物差しとしてだけ走らせる。論文に無い腕は起動しない。
- 現役資料は `current/` 配下: `README.md`＝正典（§1目的と採用状況／§2設計と現役資料の索引, 安定）、`RESEARCH.md`＝研究中の仮説・反証実験・negative results（揮発）、`paper_snv.md`＝論文、`current/<network>.md`＝フル設計ネットワーク（1ネットワーク=1ファイル, status: PROPOSED/ADOPTED、例: `current/renderer.md`）。
- 昇格フロー: RESEARCH（仮説・PROPOSED 設計）→ overfit gate / 客観関門 / **人間の耳ゲート** → 勝てば設計を ADOPTED 化し `current/README.md` §1 採用が指す。RESEARCH→採用の直行禁止。採用は耳で勝ったものだけ（proxy 単独昇格禁止）。証拠は `results/`。
- 削除した旧資料・旧 checkpoint・旧 training script（一覧は `results/DELETED_*.txt`、本文は git 履歴）は歴史的資料・失敗 artifact として扱い、現行設計の根拠にしない。特に旧 B1/B3/B4 系（`train_stage1_content.py`・`train_stage2_*.py`・`train_b3.py`・`artic_*`・`paper_mf` 等）を履歴から戻さない。

## Current Direction

- 目標は ASMR・官能バ美肉向けリアルタイム VC、Human-in-the-Loop Kansei 評価。
- **E2E レイテンシ（`current/README.md` が正本）: 設計目標 p95 < 30ms / 理想 20ms級 / p95 ≥ 50ms は失格上限**。50ms は合格ラインではない。「50ms未満ならよい」と読まない。
- **出力の高さは利用者が決める固定の半音**（オーナー判断 2026-10-25・運用と評価は `+0 / +5 / +7 / +12`）。目標の声域への自動のずらしは運用に無い。
- **2026-11-01 研究リセット（オーナー判断）**: 本線は `current/synth_nam_vc.md` ＝ 構音の構造のアナログモデリング（話者＝構音の構造＝回路、操作者の構音＝演奏、変換＝回路の差し替え）、正本論文は `current/paper_snv.md`。**フロンティア＝低レイテンシで実現可能なモデリング**。学習データは Irodori-TTS v4.1 の生成（入力側も出力側も）・zero-shot を維持・目標の声は VoiceDesign の架空の声と同意済みの実在の声。**リセット前の記録（results・RESEARCH・旧資料の測定と結論）は全て撤回、根拠にしない**（規則は規則として残る）。男声と女声の音響を 1 対 1 に対応させる対は使わない（演技は 1 対多）。
- **ノイズ耐性は必要**（オーナー 2026-11-09「例えば手をたたきながら笑うなど、ノイズ耐性は必要です」・定義は `current/paper_snv.md` §1）。声と重なる操作者の非音声（手拍子等）や伴奏の区間を、運用の外として判定から外さない。
- DAC latent から speaker-free content を学習する Stage1 依存設計は不採用。prosody は明示特徴、target voice/style/texture は別経路で扱う。
- GAN は信号・内容・制御が成立した後の texture fine-tune のみ。失敗した content 表現を GAN で救済しない。

## Shipping Gate (出荷前提) — 長時間学習を起動する前に必ず通す

**このゲートが無かったため、ストリーミングできない front-end の上で 55 時間（gvoc f0版 27.5h ＋ NHV版 27.5h）を消費した。** 二度とやらない。

- **1時間を超える学習を起動する前に `training/ship_check.py`（段 P の型は `training/snv_p1_ship.py`・`training/snv_p3_ship.py`）相当の台帳を出し、PASS を確認してから起動する。** FAIL のまま起動しない。ログ先頭に台帳を残す。Rust の製品経路は `cargo test --release -p lightvc-core --features model-tests ship_gate` で確かめる。
- **合否は 2 条件**:
  1. **未来不変性** — 入力の `t` 以降を書き換えたとき、`t` 以前を担当する出力フレームが変化しないこと。**実音声プローブで測る**（白色雑音は argmax/閾値の離散性で偽 PASS を出す。実測済み: `harmonic_sum_f0` が雑音 1.81ms / 実音声 1882ms）。編集が出力を動かさない場合は INCONCLUSIVE ＝ PASS にしない。PyTorch では CPU・決定的なアルゴリズムで測る（CUDA 畳み込みの非決定性が偽の先読みを出す）。
  2. **静的遅延台帳** — framing 先読みの合計 ＋ アルゴリズム遅延（ブロック ＋ ボコーダ `DELAY`） ＋ content encoder 予備 10ms < 30ms。
- **推論経路に発話全体の統計を置かない**（禁止例: 発話中央値による正規化、`amax()` 包絡正規化、発話 RMS レベル決定、発話中央値 f0 からの倍音数決定）。因果な走行推定か固定定数にする。front-end を直すとネットの入力分布が変わり学習し直しになる。
- **centered framing は診断と教師側の分析専用**。製品経路は左寄せ（`training/snv_p0_front.py` ↔ `crates/lightvc-core/src/p_front.rs`・`training/causal_mel.py`）。**左寄せ窓は n_fft をいくつにしても先読み 0** — 窓長は遅延ではなく過渡のにじみのツマミ。
- **ゲートを通らない構成で学習してよいのは、tag に `diag_` を付けて診断と明示した場合のみ。** その結果を製品判断・昇格の根拠に引用しない。
- **「あとで front-end だけ差し替える」は不可**と前提する。差し替えられるのは、ネットが見る値が変わらないと示せた場合だけ。

## Design Laws (設計三層則) — 学習腕の起動前必須

**配線は「何が表現可能か」を決めるだけ。学習の収束先は、損失の解析的最適解と学習データ分布上の識別可能性が決める。設計意図は回路図ではなく損失とデータ分布に書く。**（F0権威7負例・s8/s9容量無効・CFM平均退化の実測から制定）

- 全ての学習腕は起動前に `current/design_laws.md` の検査 **L**（損失最適解の書き下ろし）・**I**（識別可能性監査）・**C**（チート列挙と最ラク性）を記入し、`results/<tag>/prereg.yaml` に保存、学習ログ先頭にパスを残す。**記入できない・FAIL が埋まらない場合は起動せず停止報告**（「走らせて様子を見る」禁止）。
- 権威を持たせたい条件と同じ情報を他入力が運ぶ場合、遮断は**分布レベル**（入力摂動・ペア構成・条件ドロップ・不変性正則化）で行う。配線レベルの遮断だけでは停滞する。
- 検査 I の FAIL に容量・幅・条件 concat・loss 重み調整を対処として使わない。
- 摂動・シフト等の人工データを target 側に置かない。
  - **例外1（オーナー判断 2026-11-01）**: `synth_nam_vc` の教師は Irodori-TTS v4.1 の生成音声。その場合も符号・回路の target 側に摂動・シフト・時間伸縮を置かない（加工は入力側だけ）。
  - **例外2（オーナー判断 2026-11-26・`current/paper_snv.md` §9 O44）**: 描画器（出力部）の学習に限り、合成の残響・雑音・符号化を当てた合成音声を条件にも教師にも置いてよい（描画器は音を直さずそのまま描く）。本物の音声・本物の部屋応答の録音を使わないのは不変。
- fork（PASS→次の行動 / FAIL→次の行動）が同一になる腕は情報価値ゼロなので起動しない。
- 本ゲートは Shipping Gate への**追加**であり代替ではない。

## Data (学習データ) — 必ずフルデータでフル学習

- **学習データ（2026-11-01〜）は Irodori-TTS-v4.1-Small（MIT）の生成**（入力側も出力側も・VoiceDesign の架空の声 `t00` 参照・実在の人の音声を参照にしない）:
  - `data/snv_e1` — 600 声（女 480・男 120）× 普通の発話 10 本（`training/snv_e1_gen.py`）
  - `data/snv_e1nv` — 同じ 600 声の音素の無い発声 7 種（喘ぎ・悲鳴・吐息・息切れ・泣き声・笑い・混ざり・`training/snv_e1nv_gen.py`）
  - `data/snv_d` — 同じ声が同じ文を読み合う内容共有コーパス（`training/snv_d_gen.py`）
  - 特徴ディレクトリ: 前段 v3 は `data/snv_p_feat`（`snv_p_feat.py`）、前段 v4（現行製品）は `data/snv_p_feat_v4`・`x16_v4`・`st_v4`・`st_x16_v4`（`snv_p125_feat.py`）。データの由来一覧は `data/README.md`。
- **別の合成生成器・合成データセットの追加は可**（オーナー判断 2026-11-16・`paper_snv.md` §9 O28）。許諾を確認して `NOTICE` と台帳に残す。本物の音声から学ばない・実在人物を参照にしない・`female-dataset` の文を外部サービスへ送らない原則は不変。
- **入力側の乱しには本物の非音声の音源を使ってよい**（オーナー判断 2026-11-08）:
  - 音声を含まず許諾が明らかなもののみ（CC0 / CC BY / Public Domain。NC・ND・SA 不可）。人の声は AudioSet 分類器（AST 閾値 0.02）で除去。
  - 音源: MUSAN（`musan_sources.py`）・FMA（`fma_sources.py`）・FSD50K（`fsd50k_sources.py`、eval split は評価専用）。帰属は `ATTRIBUTION_SOURCES.md` と台帳に残す。本物の音声と本物の室内インパルス応答録音は含めない。
- **評価は本物の男声の入力**（学習・目標には一切使わない）:
  - 開発の組: Common Voice 22.0 ja（CC0）男声 239 人（`data/cv22_ja`・`snv_dev_set.py`・`devA` は設計と途中信号、`devB` は候補確認専用。話者特定を試みない）。JVNV / JNV（CC BY-SA 4.0）は物差し較正と評価入力専用。
  - 試験の 12 区間（外郎売・namikawa）は最終判定専用（`paper_snv.md` §7）。成績表は `results/LEADERBOARD.md`。
- **`female-dataset` は公開しない**（オーナー判断 2026-11-10）:
  - 音声・書き起こし・話者一覧をコミット・配布物・公開資料・外部計算機（Colab 等）に含めない。
  - `NOTICE` や GUI クレジットでは名前を出さず「非公開のデータ（a non-public dataset）」と書く（基盤ボコーダ `rvoc3hi` の学習データおよび台詞テキストの出典）。
- **Irodori-TTS-v4-Large は学習データに使わない**（Gemma 規約制限。v4.1-Small は MIT）。
- **本番学習は必ずフル規模（全話者）で行う**。少数話者・部分集合での学習で結論を出さない（切り分け用 overfit gate と `diag_` を除く）。

## Environment & Compute (uv / Local GPU / Google Colab)

- **Python 環境分離は必ず `uv` で行う**（`cd training && uv run python ...`）。**`conda` は一切禁止**。
- **計算資源と使い分け**:
  - **Linux 開発機（`ch160`）**: NVIDIA GPU（`cuda`, RTX 3080）。研究・学習・評価・Rust コア検証を担当。ヘッドレス環境で X11 開発ライブラリ（`-lxkbcommon`）が無いため、GUI バイナリ `lightvc-app` のリンクを伴うビルド/テストは Windows 実機で行い、Linux では `cargo check --workspace --all-targets`・`cargo clippy --workspace --all-targets`・`cargo test --workspace --exclude lightvc-app --release` を使う。
  - **Google Colab 利用可（オーナー判断 2026-11-28）**: `~/.local/bin/colab` で GPU（A100 等）を学習に使ってよい。運用上の鉄則:
    1. **`female-dataset` を絶対にアップロードしない**（送るのは Irodori v4.1 合成特徴・合成音声と学習コードのみ）。
    2. 大容量 tar の分割転送中はカーネルがアイドル切断されないよう待機スクリプト（例: `training/snv_diag/p125_wait_run.sh`）を走らせ、`--mirror` でチェックポイントを随時ローカルへミラーしてプリエンプションに備える。
    3. **ジョブ完了・中断後は必ずセッションを終了し、稼働セッション 0 件（`Usage rate: 0.00/hr`）を確認する。**
  - **Windows 実機**: アプリ GUI（GPUI）・ネイティブ COM ASIO・CLAP/VST3 DAW 実機検証・Windows 配布パッケージ生成を担当。`results/` は `.gitignore` 対象のため `ch160` から `scp` で取得する（手順は `current/windows_handoff.md`・`handover.md`）。
  - **Intel GPU を使う場合**: IPEX (CPU) ではなく `xpu`（torch >= 2.6 本体統合済み）を使う。
- **Rust の試験**: `cargo test --workspace --release` は速い単体の試験だけを回す（アプリの開発ではこれで足りる）。
  - **モデルの試験**（PyTorch との一致と出荷の関門・重い・`results/` の書き出しと fixture が無いと読み飛ばす）は、**推論コード（`lightvc-core`）か書き出しを替えたときだけ** `cargo test --release -p lightvc-core --features model-tests` で回す（オーナー判断 2026-11-28「一度モデルを fix したらクライアントアプリ開発時にそれを流す必要がない」）。
  - **試験は今の製品の経路だけを確かめる**（オーナー判断 2026-11-28「古いロジックの検証とかは要らない」）。製品を替えたら、試験を新しい書き出しへ向け直し、旧経路の試験は消す（中身は git の履歴に残る）。

## Build Commands

```bash
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets
cargo test --workspace --release                      # 速い単体の試験だけ。Linux では --exclude lightvc-app を付ける
cargo test --release -p lightvc-core --features model-tests -- --test-threads=1   # モデルの一致と出荷の関門(推論コードか書き出しを替えたときだけ)
cargo build --release -p lightvc-app                   # Windows 標準ビルドで WASAPI + ネイティブ COM ASIO 対応
cargo run -p lightvc-xtask -- bundle                   # CLAP / VST3 の bundle
cargo run -p lightvc-xtask -- install                  # ユーザープラグインフォルダへ導入
cargo run -p lightvc-xtask -- package                  # bundle + CLI + 3重み + voices/ + NOTICE + ATTRIBUTION_SOURCES.md + README の zip（欠品は失敗）
```

## Architecture Rules

- **推論は全て Rust**（自作実装・外部推論ランタイム不要・Candle は 2026-11-10 に撤去）。**学習は全て PyTorch (`uv`)**。
- **製品の経路 ＝ `paper_snv.md` 段 P126 の `K3G-v4`**（2026-11-28 採用・耳の関門「うーん、ええんちゃう」通過）:
  1. **前段＋瞬時マスク門**（`p_gate.rs`・`p_front.rs`・`results/snv_p126_gate_export`）: 前段 `v4_dual`（`W_YIN = 600`・`ENV_CAP_MS = 7.5`・`TAU_UV_HZ = 500`・`F0_MODE = "v4_dual"`）＋ `inst_mask = true`（`thr = 0.8`・錨切替は `hyst = [40, 100]`）。
  2. **変換網 `K3G-v4`**（`k3g_model.rs`・`p30_model.rs`・`p_fixenv.rs`・`results/snv_p126_k3g16kv_export` = `snv_p125{a,b,e}_c`）: 符号と回路 3 組の平均＋錨フォールバック、重み・注意 KV ともに f16、`target_v4 = true`。
  3. **出力部 `snv_p123_s` ＋ 出力の番**（`rvoc.rs`・`out_guard.rs`・`results/snv_p123_s/export_20k`）: `delay = 480`・因果 3 点包絡平滑 `env_smooth = 0.25`・尖頭値保護 `out_guard`。
  - `lightvc_core::snv_stream::SnvStream` が 48 kHz・240 サンプル（5 ms）単位・**4 スレッド並列**（関門 S 最悪 `p99 = 4.89 / 4.67 ms < 5.0 ms`）で流し、アルゴリズム遅延は `HOP + DELAY = 720` サンプル（`15.0 ms`）。GUI・CLAP/VST3・CLI 全て共通。
  - `training/snv_p0_front.py` の既定環境変数は旧 v3 互換のままなので、PyTorch 側で現行製品を測る・書き出すときは必ず `SNV_F0V4=1 SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500 SNV_FEAT=data/snv_p_feat_v4` を付ける。
- **PyTorch と Rust の完全一致**: `manifest.json` のテンソル名 ＝ PyTorch `state_dict` 名 ＝ Rust 読み込み名。書き出しスクリプトは `training/export_snv_gate.py`・`export_snv_k3g.py`・`export_rvoc.py`・`export_pfront.py`・`export_snv_model.py`・`export_snv_p30.py`・`export_snv_stream.py`。変更時は両方更新し一致試験（`cargo test --release -p lightvc-core --features model-tests`）を通す。
- **VC teacher 蒸留は禁止**（別 VC の変換音声を target にしない）。
- **補助モデルは評価の物差し・参考値まで**（オーナー判断 2026-11-01「既存アルゴリズムはベンチマークの参考値にしても良いが、直接教師にしない」）。WavLM-SV / ECAPA / ASR / PESQ / UTMOS 等を学習損失・蒸留・表現監督に直接使わない（使う案は事前にオーナー確認必須）。推論時依存にもしない。
- **Irodori-TTS は学習データ生成とプリセット作成専用**。推論・登録経路に入れない。実在人物の声は本人同意がある場合のみ目標にする。

## Engineering Lessons (実装・学習で踏んだ急所 — 再発防止)

- **描画器（`rvoc`）は必ず「教師の綺麗な `f0`（`harvest`）」＋「包絡重心との厳密時刻同期」で学習する**:
  - 因果追跡器の誤りを含む `f0` で描画器を学習させると、倍音間ノイズを重みが記憶して写し合成自体が壊れる（段 P97/P114 負例: PESQ `3.67 → 3.09`）。
  - 包絡に因果 3 点平滑（`[0.25, 0.5, 0.25]`, 群遅延 1 フレーム = `240` サンプル）を掛ける場合、`f0[t-1]`・`per[t-1]`・目標波形切り出し（`delay = 480`）をすべて時刻 `(t-1)·HOP` に揃える。1 フレームでもずれると 20〜100 Hz の声門うなり・雑音が出る（段 P121〜P123）。出力部を呼ぶ全経路でこの平滑を通す（段 P95）。
- **「原音 → 因果写し合成（T）」が合格水準（錨 PESQ ≥ 3.50）に届く前に変換音を耳に出さない**:
  - 写し合成が劣化していれば変換の耳出しは無意味（段 P121）。また **UTMOS は倍音間雑音・声門うなりに盲目で耳判定と逆転しうる**（段 P119 実測）ため、描画器と `c0` 揺れの検出力には最適ラグ整合 PESQ・平滑後 `c0` の 2 階差比（`d2 ≤ 1.25`）・`probe4` の `zz95` を必ず併用する。
- **因果ピッチ追跡の役割分離（`v4_dual`）と特徴変更時のフル再学習**:
  - `K3G`（`c0` 非入力）は無音・破裂音で `v=1` が立つと会話音量の声を幻覚するため厳格な `f0_net`（8 次 LPC 残差）が要り、包絡抽出・写し描画（`c0` 権威あり）は小声母音の無声化を防ぐ寛容な `f0_voc` が要る（段 P124）。
  - 前段の有声率や特徴分布を変更すると、受容野 315 ms の CPC と回路 `C`（特に `+12` 半音の着地 `M`）が分布ずれを起こすため、全コーパス特徴を抽出し直して `K3G` 全系を一から学習し直す（段 P124 → P125）。
- **門の持続ヒステリシスと STFT マスクの分離（`inst_mask`）**:
  - 錨切替 `op` の 200 ms ヒステリシス待ち中や単発手拍子（< 200 ms）でも `gate_p >= thr` なら STFT マスクと `myin` を瞬時適用する（`mask_on = op || (p >= thr)`）。これによりクリーン音声と同一 bit（`p < thr`）を保ったまま伴奏出だし・単発拍手の漏れを 15 dB 以上抑えられる（段 P126）。
- **アルゴリズム遅延の動的一貫性**:
  - ボコーダごとに `manifest.json` の `DELAY` が異なる（`rvoc3hi` = 240 / `snv_p123_s` = 480）。インスタンスがある箇所（CLI・オフライン変換・リアルタイム表示）は `stream.latency_samples()` を参照し、DAW 初期化時定数 `SnvStream::LATENCY_SAMPLES` は常に既定ボコーダの `HOP + DELAY` に一致させる（不一致は CLAP/VST3 の dry/wet 混合でコムフィルタを起こす）。

## Licensing

- **プロジェクト全体は MIT**。**GPLv3 依存は禁止**。依存追加時は `cargo tree --workspace -e normal --format '{p} {l}'` で確認する（`self_cell` は Apache-2.0 を選択して使用）。
- VST3 出力は `clap-wrapper` 0.3（MIT・同梱 VST3 SDK 3.8 も MIT）経由。
- Windows の ASIO は `crates/lightvc-audio/src/asio_win.rs` のネイティブ COM (`IASIO`) 実装により Steinberg ASIO SDK 不要で動作する（オプションの `cpal/asio` 用を含め、プロプライエタリな Steinberg ASIO SDK は絶対にリポジトリにコミット・同梱しない）。
- **配布物の重みの学習データ・ボイスプリセット・第三者許諾は `NOTICE` と GUI クレジット（`crates/lightvc-app/src/ui.rs` の `render_credits`）の両方に書く**。重み・データ・プリセット・依存を替えたら両方同時に更新する。

## Bundled Voice Presets (同梱のボイスプリセット) — 2026-11-24〜

- **`voices/` に目標の声のプリセット 30 人（女性 15・男性 15）を同梱する。** Irodori-TTS で作った架空の声であり、実在人物の録音は使わない。
- 構成は `voices/manifest.json` と `voices/<id>.flac`（48 kHz・モノラル・各 20 秒以上）。生成定義は `tools/voice_presets.json` と `tools/make_voice_presets.ps1`（要 `sttts-say` + `ffmpeg`）。
- `.gitignore` は `*.flac` を除外するため `voices/*.flac` のみ例外追跡する。他の音声ファイルをリポジトリに入れない。
- プリセットは**推論時の目標参照専用**（学習データ・評価目標には使わない）。`cargo run -p lightvc-xtask -- package` と `cargo test -p lightvc-app` が 30 声の揃いを検査する。

## Code Style

- **Rust**: コメントは最小限
- **Python**: コメントなし（`#` 行コメントを書かない）、型ヒント推奨
- **コミットメッセージ**: 英語、`feat:` / `fix:` / `docs:` / `refactor:` プレフィックス

## Results Directory Convention

- **命名**: `snv_<段>`＝`paper_snv.md` の段の走行／`diag_`＝診断専用（製品判定に引用しない）／`*_smoke`＝動作確認のみ。
- **寿命と削除台帳**: smoke は当日中に削除。負け腕・採用確定後の勝ち腕の中間 `snap/` は整理してよいが、**削除前に必ず `pgrep -f train` で走行中ジョブを確認し、`results/DELETED_<date>.txt` に一覧を追記する**（不可逆のため）。`RESEARCH.md`・`paper_snv.md` が参照する証拠（耳軸測定・帰属 json/wav）は消さない。
- **消してはならない書き出しと元チェックポイント**:
  - 現行製品既定（モデルの試験もこの fixture を読む）: `results/snv_p126_gate_export`・`results/snv_p126_k3g16kv_export`・`results/snv_p123_s/export_20k`・`results/pfront_export_p124`（前段 v4_dual の一致）
  - 旧製品・旧経路（2026-11-28 に試験から外した。消すかはオーナーの判断待ちなので、それまで消さない）: `results/snv_p45_gate_t80_export`・`results/snv_p45_gate_t80_export_v3`・`results/snv_k3g_export`・`results/snv_k3g16_export`・`results/snv_k3g16kv_export`・`results/rvoc3hi/export_330k`・`results/pfront_export`・`results/pfront_export_p100`・`results/p_model_export`・`results/snv_p30_export`・`results/snv_stream_export`
  - 書き出し元の重み: `results/snv_p126_gate_i80/last.pt`・`results/snv_p45_mask/last.pt`・`results/snv_p125{a,b,e}_c/last.pt`・`results/snv_p125{a,b,e}_cpc/{last.pt,pca_c.npy,codebook_k128.npz}`・`results/snv_p123_s/snap/ema_20k.pt`・`results/snv_p11_r/last.pt`・`results/rvoc3hi/snap/ema_330k.pt`

## Known Issues

- Windows の WASAPI 共有モードでは `BufferSize::Fixed` が効かず 30ms を切れない → 30ms 級には GUI / CLI で ASIO（標準搭載のネイティブ COM `IASIO` バックエンド、入出力 WASAPI/ASIO 混在可）を選ぶ。
- Intel XPU 使用時: backward で depthwise conv (`groups=in_ch`) が失敗するため標準 conv (`groups=1`) を使う。学習中ハング時はバッチサイズ・フレーム長を調整する。

## GUI Review Protocol

GUI は GPUI (`gpui 0.2.2`・デザイン規約は `DESIGN.md` の Material 3 Expressive 準拠・意味のない装飾要素禁止・文言は日本語統一) で実装されている（VST3/CLAP プラグイン画面のみ `nice-plug-egui`）。Windows でのビルドには `fxc.exe`（Windows SDK・`dev.ps1` が自動で `PATH` に追加）が要る。通常開発は `[profile.dev-release]`（差分ビルド）で高速に回し、本番配布は `-Release` / `xtask package` を使う。

```powershell
.\dev.ps1 -Watch
powershell -NoProfile -ExecutionPolicy Bypass -File tools\snap.ps1 -Out docs/screenshots/live.png
```
