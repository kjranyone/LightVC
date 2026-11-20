# AGENTS.md - Project Rules

## Communication

- **AIエージェントは常に日本語で応答すること**

## Active Research Document

- 研究・学習・評価・設計作業では、必ず最初に `current/README.md`（製品要件＋設計索引）と `current/RESEARCH.md`（研究中）を読むこと。
- **研究の正本は `current/paper_snv.md`**（2026-11-06〜）。オーナー方針(11-06)「では論文からのみ学習／評価を行うこと」: 学習・評価は論文の段と物差しとしてだけ走らせる。論文に無い腕は起動しない。
- 現役資料は `current/` 配下: `README.md`＝正典（§1目的と採用状況／§2設計と現役資料の索引, 安定）、`RESEARCH.md`＝研究中の仮説・反証実験・negative results（揮発）、`paper_snv.md`＝論文、`current/<network>.md`＝フル設計ネットワーク（1ネットワーク=1ファイル, status: PROPOSED/ADOPTED）。例: `current/renderer.md`（出力部 rvoc）。
- 昇格フロー: RESEARCH（仮説・PROPOSED 設計）→ overfit gate / 人間の耳ゲート → 勝てば設計を ADOPTED 化し README §1採用が指す。RESEARCH→採用の直行禁止。採用は耳で勝ったものだけ（proxy 単独昇格禁止）。証拠=`results/`、横断=`memory/`（重複させずポインタ）。
- 削除した旧資料・旧checkpoint・旧training script（一覧は `results/DELETED_*.txt`、本文は git の履歴）は歴史的資料・失敗 artifact として扱い、現行設計の根拠にしない。
- 特に旧B1/B3 staged系（`train_stage1_content.py`・`train_stage2_generator.py`・`train_stage2_adv.py`・`train_b3.py`、削除済み）を履歴から戻して現行経路として開始・継続しない。

## Current Direction

- 目標は ASMR・官能バ美肉向けリアルタイムVC、Human-in-the-Loop Kansei評価。
- **E2E レイテンシ（`current/README.md` が正本）: 設計目標 p95 &lt; 30ms / 理想 20ms級 / p95 ≥ 50ms は失格上限**。50ms は合格ラインではない。「50ms未満ならよい」と読まない。
- **出力の高さは利用者が決める固定の半音**（オーナー判断 2026-10-25・評価は +0 / +5 / +7 / +12）。目標の声域への自動のずらしは運用に無い。
- **2026-11-01 研究リセット(オーナー判断)**: 本線は `current/synth_nam_vc.md` = 構音の構造のアナログモデリング(話者 = 構音の構造 = 回路、操作者の構音 = 演奏、変換 = 回路の差し替え)。その論文が `current/paper_snv.md`（上の正本）。**フロンティア = 低レイテンシで実現可能なモデリング**。学習データは Irodori-TTS v4.1 の生成(入力側も出力側も)・zero-shot を維持・目標の声は VoiceDesign の架空の声と同意済みの実在の声。**リセット前の記録(results・RESEARCH・current の旧資料・memory の測定と結論)は全て撤回、根拠にしない**(規則は規則として残る)。旧構音 VC 系(B4 下の artic_* / paper_mf)の資料も撤回された記録。男声と女声の音響を 1 対 1 に対応させる対は使わない(演技は 1 対多)。
- **ノイズ耐性は必要**（オーナー 2026-11-09「例えば手をたたきながら笑うなど、ノイズ耐性は必要です」・定義は `current/paper_snv.md` §1）。声と重なる操作者の音や伴奏の区間を、運用の外として判定から外さない。
- リセット前の現行設計 `B4 Dual-Path Kansei VC` の資料は撤回・削除済み。目的・製品要件(上の目標と遅延)は維持。
- DAC latentからspeaker-free contentを学習するStage1依存設計は不採用。
- prosodyは明示特徴、target voice/style/textureは別経路で扱う。content を SSL/ASR の表現で作る旧方針は、オーナー判断 2026-11-01(下の Architecture Rules の補助モデル)で置き換えた。
- GANは最後のtexture fine-tuneのみ。失敗したcontent表現をGANで救済しない。

## Shipping Gate (出荷前提) — 長時間学習を起動する前に必ず通す

**このゲートが無かったため、ストリーミングできない front-end の上で 55 時間（gvoc f0版 27.5h ＋ NHV版 27.5h）を消費した。** 二度とやらない。

- **1時間を超える学習を起動する前に `training/ship_check.py`（`cd training && uv run python ship_check.py`）相当の台帳を出し、PASS を確認してからにする。** FAIL のまま起動しない。ログの先頭に台帳を残す。段 P の型は `training/snv_p1_ship.py`・`training/snv_p3_ship.py`。Rust の製品の経路は `cargo test --release -p lightvc-core ship_gate` で確かめる。
- **合否は 2 条件**:
  1. **未来不変性** — 入力の t 以降を書き換えたとき、t 以前を担当する出力フレームが変化しないこと。**実音声プローブで測る**（白色雑音は argmax/閾値の離散性で偽 PASS を出す。実測済み: `harmonic_sum_f0` が雑音 1.81ms / 実音声 1882ms）。編集が出力を動かさない場合は INCONCLUSIVE ＝ PASS にしない。PyTorch では CPU・決定的なアルゴリズムで測る（CUDA の畳み込みの非決定性が偽の先読みを出す）。
  2. **静的遅延台帳** — framing 先読みの合計 ＋ content encoder 予備 10ms &lt; 30ms。
- **推論経路に発話全体の統計を置かない**（禁止例: 発話中央値による正規化、`amax()` による包絡正規化、発話 RMS によるレベル決定、発話中央値 f0 から倍音数を決める）。因果な走行推定か固定定数にする。これらは **front-end を直すとネットの入力分布が変わるので、重みの流用ができず学習し直しになる**。
- **centered framing は診断専用**。製品経路は左寄せ（今の前段 v3 = `training/snv_p0_front.py` ↔ `crates/lightvc-core/src/p_front.rs`・`training/causal_mel.py` も左寄せ）。**左寄せ窓は n_fft をいくつにしても先読み 0** — 窓長は遅延ではなく過渡のにじみのツマミ（実測確認済み）。
- **ゲートを通らない構成で学習してよいのは、tag に `diag_` を付けて診断と明示した場合のみ。** その結果を製品判断・昇格の根拠に引用しない。
- **「あとで front-end だけ差し替える」は不可**と前提する。差し替えられるのは、ネットが見る値が変わらないと示せた場合だけ。

## Design Laws (設計三層則) — 学習腕の起動前必須

**配線は「何が表現可能か」を決めるだけ。学習の収束先は、損失の解析的最適解と学習データ分布上の識別可能性が決める。設計意図は回路図ではなく損失とデータ分布に書く。**（F0権威7負例・s8/s9容量無効・CFM平均退化の実測から制定）

- 全ての学習腕は起動前に `current/design_laws.md` の検査 **L**（損失最適解の書き下ろし）・**I**（識別可能性監査）・**C**（チート列挙と最ラク性）を記入し、`results/<tag>/prereg.yaml` に保存、学習ログ先頭にパスを残す。**記入できない・FAILが埋まらない場合は起動せず停止報告**（「走らせて様子を見る」禁止）。
- 権威を持たせたい条件と同じ情報を他入力が運ぶ場合、遮断は**分布レベル**（入力摂動・ペア構成・条件ドロップ・不変性正則化）で行う。配線レベルの遮断だけでは停滞する（NSF v2実測）。
- 検査IのFAILに容量・幅・条件concat・loss重み調整を対処として使わない（s8/s9/f0h実測で無効）。
- 摂動・シフト等の人工データをtarget側に置かない。教師は実音声を維持する（s12負例）。**例外(オーナー判断 2026-11-01)**: synth_nam_vc の教師は Irodori-TTS v4.1 の生成音声。その場合も target 側に摂動・シフト・時間伸縮を置かない（加工は入力側だけ）。
- fork（PASS→次の行動/FAIL→次の行動）が同一になる腕は情報価値ゼロなので起動しない。
- 本ゲートは Shipping Gate（未来不変性・遅延台帳・RTF）への**追加**であり代替ではない。

## Data (学習データ) — 必ずフルデータでフル学習

- **学習データ（2026-11-01〜）は Irodori-TTS-v4.1-Small（MIT）の生成**（入力側も出力側も・オーナー判断 2026-11-01）。声は VoiceDesign の架空の声（説明文だけで作った t00 を、その声の参照にする。実在の人の音声を参照にしない）。今のコーパス:
  - `data/snv_e1` — 600 声（女 480・男 120）× 普通の発話 10 本（`training/snv_e1_gen.py`・台帳 `data/snv_e1/manifest.json`）
  - `data/snv_e1nv` — 同じ 600 声の音素の無い発声 7 種（喘ぎ・悲鳴・吐息・息切れ・泣き声・笑い・混ざり・`training/snv_e1nv_gen.py`）
  - `data/snv_d` — 同じ声が同じ文を読み合う、内容を共有するコーパス（paper_snv 段 D・`training/snv_d_gen.py`）
  - 特徴は前段 v3 で `data/snv_p_feat`（`training/snv_p_feat.py`）。台詞の文は female-dataset の書き起こしから採る（露骨な性的表現を除く・`snv_e1_gen.text_pool`）。
- **別の合成の生成器・合成のデータセットを学習データに足してよい**（オーナー判断 2026-11-16「別の合成データセットを使っても構いません」・`current/paper_snv.md` §9 O28）。許諾（コード・重み・生成物の利用）を確かめ、`NOTICE` と台帳に残す。本物の音声から学ばない・実在の人の音声を参照にしない・female-dataset の文を外のサービスへ送らない、は変わらない。足す生成器は論文の段で決める（今は未定）。
- **入力側の乱しには本物の非音声の音源を使ってよい**（オーナー判断 2026-11-08「本物の非音声の音源を許可」）。条件は次のとおり。
  - 音声を含まず、許諾が明らかなものだけ（CC0 / CC BY / パブリックドメイン。NC・ND・SA は除く）。
  - 人の声は AudioSet の分類器（AST）で選別する（閾 0.02）。
  - 音源と選び方: MUSAN（`training/musan_sources.py`）・FMA（`training/fma_sources.py`）・FSD50K（`training/fsd50k_sources.py`。評価の組は評価だけ）。帰属は台帳に残す。
  - 当てる先は Irodori の元の音のまま。本物の音声と本物の部屋の応答の録音は、この例外に含めない。
- **評価は本物の男声の入力**（外郎売・namikawa・VCTK の男声）。JVNV / JNV（CC BY-SA 4.0）は物差しの較正と評価の入力の側だけに使い、学習・目標にしない。
  - 開発の組（2026-11-10〜）= Common Voice 22.0 ja（CC0）の男声 239 人（`data/cv22_ja`・`training/snv_dev_set.py`・devA は設計と途中信号・devB は候補の確認だけ）。評価の入力だけに使い、学習・目標にしない。話者の特定を試みない。
  - 試験の 12 区間（外郎売・namikawa）は最終判定だけに使う（`current/paper_snv.md` §7）。成績表は `results/LEADERBOARD.md`（`training/snv_dev_eval.py`・`training/snv_leaderboard.py`）。
- **female-dataset は公開しない**（オーナー判断 2026-11-10「female-datasetは公開しない」）。
  - 音声・書き起こし・話者の一覧を、コミット・配布物・公開の資料に含めない。
  - `NOTICE` では名前を出さず「非公開のデータ」と書く。
  - 出力部 rvoc3hi（リセット前の学習）の学習データで、今の台詞の文の出典でもある。rvoc3hi は本物の録音と、それを参照にした Irodori-TTS v3 の複製で学習している。
- **Irodori-TTS-v4-Large は、この用途の学習データに使わない**。Gemma の規約が性的な満足を目的とする生成を禁じる（2026-11-02 の調べ・v4.1-Small は MIT）。`data/pair_tts_v4l` はリセット前の v4-Large の生成。
- **本番学習は必ずフルの規模で行う**。本番・比較・昇格の学習は、コーパスの全ての声を使う。**少数話者・部分集合での学習は誤り**（切り分け用の overfit gate と `diag_` を除く）。
- 新しいコーパスは学習の前に前段 v3 で全量を特徴にし（`data/snv_p_feat` の型）、台帳（manifest）を揃える。「まず小さく回す」で結論を出さない。
- リセット前のコーパスと特徴（`data/female_real_feat`・`data/female_tts_corpus`・`data/male_feat` など）は 2026-11-09 に削除した（`results/DELETED_2026-11-09.txt`）。
- データの置き場と各ディレクトリの由来は `data/README.md`。

## Environment

- **Python環境分離は必ず uv で行う**
- **conda は禁止**。`conda install` / `conda create` / `conda search` は一切使わない
- 今の学習機は NVIDIA GPU（device `cuda`）。Intel加速は **IPEX (CPU) ではなく XPU (Intel GPU)** を使う
  - device は `xpu`
  - torch >= 2.6 は XPU が本体統合済み（別途IPEX不要）
- Rust の lint/typecheck: `cargo check --workspace --all-targets` / `cargo clippy --workspace --all-targets`
- Rust の試験: `cargo test --workspace --release`（PyTorch との一致と出荷の関門。`results/` の書き出しと fixture が無いと読み飛ばす）

## Build Commands

```bash
cargo build --release -p lightvc-app
cargo build --release --features asio -p lightvc-app   # ASIO SDK は各自で用意する
cargo run -p lightvc-xtask -- bundle    # CLAP / VST3 の bundle
cargo run -p lightvc-xtask -- install
cargo run -p lightvc-xtask -- package   # bundle + CLI + 3 つの重みの書き出し + NOTICE + ATTRIBUTION_SOURCES.md + README の zip（欠品なら失敗）
```

## Architecture Rules

- **推論は全て Rust**（自作の実装・外部の推論ライブラリなし。Candle は 2026-11-10 に外した）— Pythonランタイム不要
- **製品の経路 = paper_snv の段 P93 の K3G**(2026-11-19・数値の関門・耳の関門・Rust の一致と実時間を通った)。門と重みつきの前段（`p_gate.rs`・`p_front.rs`・`results/snv_p45_gate_t80_export`）→ 符号と回路の 3 組の平均と錨（`k3g_model.rs`・`p30_model.rs`・`p_fixenv.rs`・`results/snv_k3g16kv_export` = 重みと注意の鍵と値を f16 で持つ）→ 出力部 rvoc3hi（`rvoc.rs`）。実時間には 4 スレッドが要る。それまでの製品の経路 = 前段 v3 → E / C（`p_model.rs`・`results/snv_p11_r`）。
  - これを `lightvc_core::snv_stream::SnvStream` が 48 kHz・240 サンプル（5 ms）単位で流す。GUI・CLAP / VST3・CLI は全てこれを使う。
  - 研究の候補（RESEARCH.md・成績表 `results/LEADERBOARD.md`・段 I の型）は、K3G を除いて Rust に移していない。
- **学習は全て PyTorch (uv環境)** — Rustコードに依存しない
- **PyTorch と Rust の名前は完全一致** — 書き出しの `manifest.json` の名前 = PyTorch の state_dict の名前 = Rust の読み込みの名前。
  - 書き出しのスクリプト: `training/export_snv_gate.py`(門と重みつきの前段)・`export_snv_k3g.py`(K3G・`SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 … --out ../results/snv_k3g16kv_export` で製品の書き出し)・`export_rvoc.py`・`export_pfront.py`・`export_snv_model.py`・`export_snv_p30.py`・`export_snv_stream.py`（端から端の一致の fixture）。
  - 変更時は両方更新し、一致の試験（`cargo test`）を通す。
- **VC teacher蒸留は禁止** — 別VCモデルの変換音声を target とする synthetic parallel distillation は使わない
- **補助モデルは評価の物差し・参考値まで** — WavLM-SV/ECAPA/ASR 等は評価に使ってよい。推論時依存にしない。
  - **学習の目標・蒸留・表現の監督として直接使わない**（オーナー判断 2026-11-01「既存アルゴリズムはベンチマークの参考値にしても良いが、直接教師にしない」）。旧来の「loss・蒸留に使ってよい」より、この判断が優先する。
  - 損失の中に既存モデルを置く案は、使う前にオーナーに確認する。
- **VC teacher不要** — targetは実音声、同一内容ペア、または信号処理由来に限定
- **TTS の生成音声(オーナー判断 2026-11-01)** — Irodori-TTS v4.1 の生成音声を学習データに使ってよい。Irodori は学習データの生成だけに使い、登録・推論に使わない。Irodori のモデルカードに従い、実在の人の声は本人の同意がある場合だけ目標にする

## Licensing

- **プロジェクト全体は MIT**
- **GPLv3依存は禁止**。依存を足したら `cargo tree --workspace -e normal --format '{p} {l}'` で確かめる（`self_cell` は Apache-2.0 / GPL-2.0-only の選択で、Apache-2.0 で使う）
- VST3出力は `clap-wrapper`（MIT）経由（crate `clap-wrapper` 0.3 が同梱する VST3 SDK 3.8 も MIT）
- ASIO SDK はプロプライエタリ（再配布禁止）— リポジトリにコミットしない
- **配布物の重みの学習データと第三者の許諾は `NOTICE` に書く**。重み・学習データ・依存を替えたら NOTICE も直す。

## Code Style

- **Rust**: コメントは最小限
- **Python**: コメントなし、型ヒント推奨
- **コミットメッセージ**: 英語、`feat:` / `fix:` / `docs:` / `refactor:` プレフィックス

## Results Directory Convention

- **命名**: `snv_<段>`=paper_snv の段の走行（例 `snv_p11_r`・`snv_i10_c_pr`）／`diag_`=診断専用（部分集合・ゲート外構成。製品判定に引用しない）／`*_smoke` は動作確認のみ。
- **寿命**: smoke は当日中に削除。負け腕は判定記録後に best（+last）のみ残し中間 snap は削除。勝ち腕も採用確定後に snap を削除。
- **削除台帳**: 削除前に `results/DELETED_<date>.txt` に一覧を追記する（不可逆のため）。
- **走行中の run が参照するパス（ckpt・キャッシュ・データ）を消さない**。整理前に `pgrep -f train` で走行中ジョブを確認する。
- **例外**: RESEARCH.md・paper_snv.md が数値・パスで参照する証拠（耳軸測定・帰属 arm の wav/json）は削除しない。
- **製品の書き出しを消さない**:
  - `results/snv_p45_gate_t80_export`・`results/snv_k3g16kv_export`・`results/rvoc3hi/export_330k` は、配布物と GUI・CLI の既定が読む。`results/pfront_export`・`results/p_model_export`(それまでの製品)と `results/snv_k3g_export`・`results/snv_k3g16_export`(K3G の f32 と重みだけ f16・一致の試験)も消さない。
  - `results/snv_stream_export` は Rust の端から端の一致の試験が読む。
  - `results/snv_p11_r/last.pt` と `results/rvoc3hi/snap/ema_330k.pt` は書き出しの元。

## Known Issues

- XPU backwardでdepthwise conv (groups=in_ch) が失敗する → 標準conv (groups=1) を使用
- XPU学習中にPCハングする場合あり → CPU学習またはバッチサイズ/フレーム長調整
- Windows の WASAPI 共有モードでは cpal が `BufferSize::Fixed` を効かせず、30ms を切れない → 30ms 級には ASIO（`--features asio`）

## GUI Review Protocol

GUIレビューが必要な場合のみ、実コードとスクショを確認する（`lightvc-app gui --demo` は模擬の値で描く）。

```powershell
.\dev.ps1 -Watch
powershell -NoProfile -ExecutionPolicy Bypass -File tools\snap.ps1 -Out docs/screenshots/live.png
```

`dev.ps1` は `-Watch`・`-Demo`・`-Snap`・`-BuildOnly` が今のアプリで動く。DAC の重みの取得（`-Demo` 以外の起動で走る）・`-Roundtrip`・`-Cuda` / `-Metal` は旧経路の名残で、今のアプリには無い。
