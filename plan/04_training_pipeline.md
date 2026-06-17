# 04: 学習パイプラインの乖離

> カテゴリ: D
> 関連資料: MODEL_TRAINING.md, training/README.md, MANUAL.md §6

## 概要

学習パイプラインは設計（Phase B 50K step / Phase C 200K step on LibriTTS/VCTK / bf16 / 外部評価指標）に対して、全体的に「smoke test 用に縮小」状態。現在の実装で VC として発音することは確認できるが、SOTA 到達には程遠い。設計を本来仕様に戻すか、現状を「スモークテスト」として文書化するかの判断が必要。

## 現状の乖離

| 項目 | 設計（MODEL_TRAINING.md） | 実装（configs / コード） |
|---|---|---|
| Phase B `max_steps` | 50,000（~2h） | 50,000（本番）/ 10,000（smoke）[04-1] 解消 |
| Phase C `max_steps` | 200,000（~5-7 day） | 200,000（本番）/ 30,000（smoke）[04-1] 解消 |
| Phase C `mixed_precision` | bf16 | bf16（実機検証 pending）[04-2] 🚧 |
| `batch_size` (Phase C) | **4** | **8**（本番）⚠️ [04-13] |
| `learning_rate` (Phase C) | **1.0e-4** | **1.5e-4** ⚠️ [04-13] |
| `max_utterance_frames` (Phase C) | **400** | **600** ⚠️ [04-13] |
| `min_utterance_frames` | **50** | **30**（かつコードは無視して硬编码30）⚠️ [04-13] |
| コーパス | LibriTTS / VCTK | download_corpus.py 実装済み [04-3] |
| Phase B `role_assignment` | reconstruction:0.6 / cross_speaker:0.4 | **未実装**（硬编码 50/50、cross-speaker は no-op）⚠️ [04-10] |
| Phase B `speaker_consistency` | target=参照話者 | **target=source話者**（VCに参照無視を学習させる逆監督）⚠️ [04-8] |
| Phase B `content_preservation` | content_code の話者不変性 | **恒等的にゼロ**（src vs src）⚠️ [04-9] |
| Phase B `speaker_classify` 損失 | 設計・設定に不在 | **暗黙デフォルト 0.5 で追加** ⚠️ [04-11] |
| content loss | VQMIVC-style MI（gradient reversal） | GRL実装済み [04-4] |
| speaker SECS | ECAPA-TDNN / WavLM 外部 | evaluate.py 実装済み [04-5] |
| Validation | UTMOS / Whisper WER / VCTK parallel | 実装済み [04-5][04-6] |
| **mean-flow 定式化** | MeanVC2 の mean-flow | **実装は rectified/linear FM**（誤帰属）⚠️ [04-7] |
| Quick Start の config 名 | `phase_b_warmstart.yaml` / `phase_c_flow.yaml` | **存在しない** ⚠️ [04-13] |

## タスクリスト

### [04-1] (P1) ✅ 学習ステップ数の整合（smoke / 本番の分離）
- **現状**: 設計は大規模、実装は smoke test 規模。両者が同じファイル名 (`phase_b.yaml` / `phase_c.yaml`) で混在し、利用者がどちらを参照しているか判然としない。
- **作業**:
  - 設定ファイルを明示的に分離
    - `phase_b_smoke.yaml`（10K step, TTS corpus）: 現在の `phase_b.yaml` をリネーム
    - `phase_b.yaml`（50K step, LibriTTS/VCTK）: 設計通り新規作成
    - `phase_c_smoke.yaml`（30K step）/ `phase_c.yaml`（200K step）: 同様
  - `training/README.md` に両者の使い分けを明記
  - `MODEL_TRAINING.md` の Quick Start（407-444 行）を見直し
- **受け入れ基準**: smoke 用 / 本番用の設定が明示的に分かれ、README で位置づけが説明されていること。
- **関連**: `training/configs/phase_b.yaml`, `training/configs/phase_c.yaml`, `MODEL_TRAINING.md:166, 301, 407-444`

### [04-2] (P✅) mixed_precision=bf16 の検証と再有効化
- **現状**: `phase_c.yaml:38` は `mixed_precision: none`。MODEL_TRAINING.md C.5（324 行）は bf16 を指定。AGENTS.md は XPU を前提とし、B580 は BF16 対応。
- **影響**: XPU 学習速度が fp32 では大幅に低下。Phase C を 200K step まで回す場合、実用的でなくなる。
- **作業**:
  1. bf16 有効版で 1000 step 動作確認（`phase_c_smoke.yaml` を一時的に変更）
  2. 数値安定性確認（NaN / Inf 発生率、loss 曲線の fp32 との一致）
  3. 安定すれば `phase_c.yaml` を bf16 に更新
  4. 必要なら `torch.autocast` の対象箇所を見直し（現在は `forward_velocity` と `loss_fm` のみ autocast 内、他の loss は外）
- **受け入れ基準**: bf16 で 1000 step 完走、loss 曲線が fp32 と同等（100 step 移動平均で ±10% 以内）。
- **関連**: `training/configs/phase_c.yaml:38`, `training/train_flow.py:197, 227-233`, `MODEL_TRAINING.md:324`

### [04-3] (P1) ✅ LibriTTS / VCTK 本格学習手順の整備
- **現状**: MANUAL.md §6.6（288-295 行）は Edge TTS コーパス生成を「クイックスタート」として紹介。MODEL_TRAINING.md A.2（69-78 行）は LibriTTS/VCTK を前提。両者の位置づけが曖昧。
- **作業**:
  1. `training/README.md` に「smoke test（Edge TTS）」「本格学習（LibriTTS/VCTK）」の 2 パスを明記
  2. `download_corpus.py` を新設（HuggingFace datasets 経由で LibriTTS / VCTK ダウンロード）
  3. `MANUAL.md §6.4`（269-278 行）の「100 話者以上推奨」を強調、smoke test はあくまで動作確認と明記
  4. `encode_corpus.py` が LibriTTS / VCTK のディレクトリ構造を正しく扱えるか検証（既に speaker_of があるので概ね OK）
- **受け入れ基準**: 公開コーパスで 100+ 話者のエンコード→学習が再現できること。
- **関連**: `training/generate_tts_corpus.py`, `training/encode_corpus.py:39-66`, `training/README.md`, `MANUAL.md:269-296`, `MODEL_TRAINING.md:69-78`

### [04-4] (P1) ✅ content MI loss（gradient reversal）の実装
- **現状**: `train_flow.py:247-252` は単なる L1 (`content_inv`)。MODEL_TRAINING.md C.4 #5（273-275 行）は「VQMIVC-style mutual information regularization via gradient reversal on speaker classification of content_code」を想定。
- **影響**: content / speaker の disentanglement が弱く、ゼロショット時に話者漏れ（source 声質が残る）が起きやすい。
- **作業**:
  1. 軽量 speaker classifier（`content_code [B, 256, T]` → `speaker logits [B, n_speakers]`）を追加
  2. GradientReversalLayer（GRL）で content_code の speaker 情報を削る
  3. または infoNCE による contrastive content loss を実装
  4. `phase_c.yaml` の `losses` に `content_mi` を追加
- **受け入れ基準**: 別話者への変換時に content 保存性（WER、[04-5] が必要）が改善すること。
- **関連**: `training/train_flow.py:247-252`, `MODEL_TRAINING.md:273-275`, `RESEARCH.md:218`

### [04-5] (P1) ✅ 外部指標（SECS / UTMOS / WER）評価パイプライン
- **現状**: `infer_flow.py` は推論するだけ。MODEL_TRAINING.md「Validation Protocol」（380-403 行）が未実装。
- **影響**: モデル品質が定量的に測定できず、改善・回帰の判断が主観的。
- **作業**:
  1. `evaluate.py` を新設
  2. **SECS**: `speechbrain/spkrec-ecapa-voxceleb` または `microsoft/wavlm-large-superb-sv`
  3. **UTMOS**: `sarulab-speech/utmos-strong` 或いは UTMOS predictor
  4. **WER**: `openai/whisper-large-v3` で src / converted の文字起こし比較
  5. VCTK parallel セットでの content preservation 測定（[04-6]）
  6. `pyproject.toml` に必要パッケージ（speechbrain / evaluate 等）を追加
- **受け入れ基準**: 学習済みモデルに対して 4 指標が算出でき、目標（SECS > 0.70, UTMOS > 3.5, WER < 5%, WER degradation < 2%）を確認できること。
- **関連**: `training/infer_flow.py`, `MODEL_TRAINING.md:380-403`, `training/pyproject.toml`

### [04-6] (P2) ✅ VCTK parallel validation の整備
- **現状**: MODEL_TRAINING.md「VCTK Parallel Validation」（391-403 行）が未実装。
- **作業**:
  1. VCTK の同テキスト他話者ペアのホールドアウトリスト作成（スクリプトで生成）
  2. [04-5] の `evaluate.py` に組み込み
  3. content preservation: `WER(src) vs WER(converted)` を算出
- **受け入れ基準**: VCTK parallel セットで WER degradation が算出できる。
- **関連**: `MODEL_TRAINING.md:391-403`

### [04-7] (P0) ✅ mean-flow 定式化の誤帰属 + MODEL_TRAINING.md 内部矛盾
- **現状**: 3つの問題が重畳:
  1. **MODEL_TRAINING.md 内部矛盾**: §C.2（213,216行）は `z_t = (1-t)*z_noise + t*z_tgt`（**ノイズ起点**）、`v_target = z_tgt - z_noise`。一方 §C.2（228行）と §C.3（242行）は `z_0 = z_src`（**ソース起点**）。両立不能。
  2. **コードは §C.3 に従う**: `train_flow.py:281-292`
     ```python
     z_0 = z_src                              # ソース起点
     z_t = (1-t)*z_0 + t*z_tgt
     v_target = z_tgt - z_0                   # 定数（t非依存）
     ```
  3. **「mean-flow」呼称は誤帰属**: MeanVC2 の mean-flow は**平均速度** `v̄(t) = (1/t)∫₀ᵗ v(z_s,s)ds` を予測し、mean-flow 時間変数で条件付ける。本実装は `t ~ U[0,1]` をサンプルし `v_target = z_tgt - z_0`（線形流の一定速度）を全 t で学習 → これは **rectified flow / linear flow matching**（Lipman 2023, Liu 2022）。1-NFE 性は流の線形性から生じ、MeanVC2 の定式化ではない。
- **影響**:
  - 1-NFE 推論自体は正しく動作（線形 FM なので `t=1` で `z_0 + v ≈ z_tgt`）
  - ただし DESIGN.md（9,59-63,126,138,159,172行）、MODEL_TRAINING.md、converter.py:9,319-332、train_flow.py:1-17、README.md:4、MODEL_TRAINING.md:473 の参考文献表が「MeanVC2 mean-flow」と**誤って帰属**
  - 論文新規性を主張する際の足元を揺るがす
- **作業**:
  1. MODEL_TRAINING.md §C.2 を §C.3 と整合するよう修正（ノイズ起点の記述を削除）
  2. DESIGN.md / converter.py / train_flow.py / README.md の「mean-flow」を「rectified flow matching (linear FM, 1-NFE)」に修正。MeanVC2 への言及は「FRC と UTTE を参考」部分のみ残し、定式化の帰属を是正
  3. MODEL_TRAINING.md:473 の参考文献表の MeanVC2 行の Paradigm を「mean-flow」→「FRC+UTTE（定式化は本プロジェクトが linear FM 採用）」に修正
- **受け入れ基準**: 全ドキュメントで定式化の記述がコードと一致し、MeanVC2 への誤帰属が解消されていること。
- **関連**: `MODEL_TRAINING.md:202-250,473`, `DESIGN.md:9,59-63,126,138,159,172`, `training/train_flow.py:1-17,281-292`, `training/converter.py:9,318-332`, `README.md:4`

### [04-8] (P0) ✅ Phase B `speaker_consistency` が source 話者を target にしている（逆監督）
- **現状**: `train_warmstart.py:213-225`
  ```python
  tgt_embed = model.speaker_embedding(tgt)   # tgt == src（104行で tgt_list.append(src)）
  ...
  loss_spk = (1.0 - cos(pred_embed, tgt_embed)) * loss_cfg["speaker_consistency"]
  ```
  MODEL_TRAINING.md:171 は `# pred speaker ≈ ref speaker` を想定。コードは pred を **source 話者**へ引っ張る。cross-speaker role でも target は src 自身なので、モデルは「参照を無視して source を維持する」よう学習する = VC の目的と逆。
- **影響**: warm-start モデルが話者変換を学習しない可能性。Phase C の init_from がこの重みを使うため、初期化品質にも影響。
- **作業**: `train_warmstart.py:213` の `tgt` を `ref` に変更。role が reconstruction のときは ref=src で従来通り、cross_speaker のときは ref=B で正しく参照話者へ引っ張る。ただし [04-10]（cross-speaker role の no-op）と併せて修正が必要。
- **受け入れ基準**: cross_speaker role で pred_embed が ref_embed（参照話者）へ近づく。損失曲線が下降。
- **関連**: `training/train_warmstart.py:104,213-225`, `MODEL_TRAINING.md:171`

### [04-9] (P0) ✅ Phase B `content_preservation` が恒等的にゼロ
- **現状**: `train_warmstart.py:228-232`
  ```python
  content_src = model.content_code(src)
  content_tgt = model.content_code(tgt)      # tgt == src ⇒ 同じ入力
  loss_content = F.l1_loss(content_src, content_tgt)
  ```
  `src` と `tgt` は**同じテンソル**。`BottleneckEncoder` は決定論的（dropout なし）なので `loss_content ≡ 0`。何も寄与しない。MODEL_TRAINING.md:172 の意図「content_code は話者不変」を達成するには**異話者・同テキスト**の content code 比較が必要だが、非並列データでは不可能。
- **影響**: warm-start の disentanglement 圧力が `speaker_consistency`（[04-8] も壊れている）のみとなり、content/speaker 分離が学習されない。
- **作業**:
  - **(a)** cross_speaker role で `content_code(z_src_with_B_timbre)` vs `content_code(z_src)` を比較する（[04-10] の cross-speaker 実装と整合）。ただし teacher なしで z_src_with_B_timbre を作る必要があり、timbre_shifter で代用する案
  - **(b)** Phase B から content_preservation を削除し、Phase C の GRL（[04-4]）に一任
  - **(c)** 同一話者の別発話で content code の時間平均比較（弱い監督）
- **受け入れ基準**: loss_content が意味のある値を持ち、学習で変動すること。
- **関連**: `training/train_warmstart.py:228-232`, `MODEL_TRAINING.md:172`

### [04-10] (P1) ✅ Phase B `role_assignment` 未実装 + cross-speaker role が no-op
- **現状**:
  - MODEL_TRAINING.md:174-177 は `role_assignment: { reconstruction: 0.6, cross_speaker: 0.4 }` を指定
  - **コードにも設定ファイルにも `role_assignment` は存在しない**（リポジトリ全体 grep で doc のみヒット）
  - `train_warmstart.py:91` は `np.random.random() < 0.5` の**固定 50/50**
  - cross_speaker role（`train_warmstart.py:104`）は `tgt_list.append(src)` で target=source。teacher がないため「z_src(A) with B's timbre」を作れず、cross_speaker は reconstruction と**同一の監督**になる（no-op）
- **影響**: 設計の「role assignment による学習時/推論時ミスマッチ低減」が未達成。[04-8][04-9] の根源原因。
- **作業**:
  1. `phase_b.yaml` に `role_assignment` ブロック追加、`train_warmstart.py` で読み込み
  2. cross_speaker role を有意にする設計:
     - 案1: timbre_shifter で z_src を B 風に変形した疑似 target を使う
     - 案2: teacher 不要の content-preserving 変換（例: 別話者の ref で FiLM 条件付けした自己再生）
  3. `train_warmstart.py:104` の target 設定を role に応じて分岐
- **受け入れ基準**: cross_speaker role が reconstruction と異なる監督を与えること。
- **関連**: `training/train_warmstart.py:91,104`, `MODEL_TRAINING.md:174-177`

### [04-11] (P1) ✅ ドキュメント・設定にない `speaker_classify` 補助損失が暗黙追加
- **現状**: `train_warmstart.py:234-237`
  ```python
  loss_cls = F.cross_entropy(logits, ref_idx) * loss_cfg.get("speaker_classify", 0.5)
  ```
  `speaker_classify` は**設定ファイルにも MODEL_TRAINING.md §B.3 にも不在**。`loss_cfg.get(..., 0.5)` で暗黙デフォルト 0.5 が効く。ref_embed に対する話者分類 CE 損失。
- **影響**: ドキュメントにない損失が効くため、再現性・解釈性が損なわれる。意図的なら doc/config へ記載、不要なら削除。
- **作業**:
  - **(a)** 意図的: `phase_b.yaml` に `speaker_classify: 0.5` を追記し MODEL_TRAINING.md §B.3 へ説明追加
  - **(b)** 不要: `train_warmstart.py:234-237` を削除
- **受け入れ基準**: 設定・ドキュメント・コードの損失項が完全一致。
- **関連**: `training/train_warmstart.py:234-237`, `MODEL_TRAINING.md:166-172`, `training/configs/phase_b.yaml`

### [04-12] (P2) ✅ timbre-shift の apply_prob 希釈 + content_inv 弱化 + GRL lambda 非設定化
- **現状**: 3つの関連問題:
  1. **apply_prob 希釈**: `encode_corpus.py:184` が `_ts(wav, 44100)` を `apply_prob` 指定なしで呼出 → `timbre_shifter.py:92` デフォルト `0.5` が効き、各 `_ts.npy` の**50% が未変換オリジナル**になる。`_ts` 接尾辞なのに shift されていないファイルが混入。
  2. **Phase C content_inv が弱い**: `train_flow.py:318-322` で `content_src = bottleneck(z_src)`, `content_pred = bottleneck(z_src + v_pred.detach())`。`z_src` と `z_src+v_pred` は大部分が共有 → 自明に満たせる。disentanglement 圧力は GRL（[04-4]）のみ。
  3. **GRL lambda 非設定化**: `train_flow.py:333` が `grad_reverse(content_src)` をデフォルト `lambda_=1.0` で呼出。実効反転強度 = `content_mi_weight`（0.1）のみ。`lambda_` は config から触れない。
  4. **DisentangledConverter.forward_velocity がデッドコード**: `converter.py:520-531` は `(v, spk_logits)` を返すが、`train_flow.py` は `model.forward_velocity` と `disentangled.adversary` を別々に呼ぶ。wrapper 未使用。
- **作業**:
  1. `encode_corpus.py:184` で `apply_prob=1.0` を明示指定（encode 時は必ず shift）
  2. content_inv は残すが weight を 0.1 に下げるか削除（GRL が主）
  3. `phase_c.yaml` に `grl_lambda: 1.0` を追加し `train_flow.py:333` で読み込み
  4. `DisentangledConverter.forward_velocity` を削除（デッドコード）
- **受け入れ基準**: `_ts.npy` が全件 shift 済み。GRL 強度が config 制御可能。デッドコード除去。
- **関連**: `training/encode_corpus.py:184`, `training/timbre_shifter.py:92,105-106`, `training/train_flow.py:318-333`, `training/converter.py:506-534`

### [04-13] (P1) ✅ 設定ファイルとドキュメントの数値不一致（7件）
- **現状**:

  | 項目 | MODEL_TRAINING.md | 実際の config / コード |
  |---|---|---|
  | Phase C `learning_rate` | `1.0e-4`（301行）| `1.5e-4`（phase_c.yaml:20, phase_c_smoke.yaml:14）|
  | Phase C `batch_size` | `4`（300行「flow matching needs more memory」）| `8`（phase_c.yaml:19）/ `4`（smoke）|
  | Phase C `max_utterance_frames` | `400`（319行）| `600`（phase_c.yaml:45）/ `400`（smoke）|
  | `init_from` パス | `checkpoints/phase_b_warmstart/best.pt`（299行）| `checkpoints/phase_b/best.pt`（phase_c.yaml:18）|
  | Quick Start config 名 | `phase_b_warmstart.yaml`（155,423行）/ `phase_c_flow.yaml`（288,429行）| **存在しない**（`.yaml` のみ）|
  | `min_utterance_frames` | `50`（320行）| `30`（全 config）。かつ `train_warmstart.py:61`/`train_flow.py:67` は**硬编码 `< 30`** で config を無視 |
  | `device` | `xpu`（323行）| `auto`（全 config、`train_flow.py:188-197` で xpu 含む auto 解決）|

- **影響**: doc の Quick Start が手順通りに実行できない（存在しない config 名）。パラメータ値の信頼性低下。
- **作業**:
  1. MODEL_TRAINING.md の数値を config 実値に合わせる（学習結果を変えない方針）
  2. `init_from` を `checkpoints/phase_b/best.pt` に、Quick Start の config 名を `phase_b.yaml`/`phase_c.yaml` に
  3. `min_utterance_frames` を config から読むよう `load_latent_corpus` のシグネチャ拡張。または硬编码を 50 に上げ config と整合
  4. `device: xpu` を `device: auto` に（機能等価だが doc を実態に合わせる）
- **受け入れ基準**: MODEL_TRAINING.md の Quick Start がコピペで実行できる。数値が config と一致。
- **関連**: `MODEL_TRAINING.md:155,288,299-320,423,429`, `training/configs/phase_c.yaml`, `training/configs/phase_c_smoke.yaml`, `training/train_flow.py:67,188-197`, `training/train_warmstart.py:61`

## 関連文書
- [03_converter_model.md](03_converter_model.md)
- [07_unimplemented_phases.md](07_unimplemented_phases.md)
