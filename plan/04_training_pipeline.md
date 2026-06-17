# 04: 学習パイプラインの乖離

> カテゴリ: D
> 関連資料: MODEL_TRAINING.md, training/README.md, MANUAL.md §6

## 概要

学習パイプラインは設計（Phase B 50K step / Phase C 200K step on LibriTTS/VCTK / bf16 / 外部評価指標）に対して、全体的に「smoke test 用に縮小」状態。現在の実装で VC として発音することは確認できるが、SOTA 到達には程遠い。設計を本来仕様に戻すか、現状を「スモークテスト」として文書化するかの判断が必要。

## 現状の乖離

| 項目 | 設計（MODEL_TRAINING.md） | 実装（configs） |
|---|---|---|
| Phase B `max_steps` | 50,000（~2h） | 10,000 |
| Phase C `max_steps` | 200,000（~5-7 day） | 30,000 |
| Phase C `mixed_precision` | bf16 | none |
| `batch_size` (warmstart) | 8 | 4 |
| コーパス | LibriTTS / VCTK 100+ 話者 | Edge TTS 17 話者×170 発話 |
| content loss | VQMIVC-style MI（gradient reversal） | L1（`content_inv`） |
| speaker SECS | ECAPA-TDNN / WavLM 外部 | 内部 `speaker_encoder` cos sim |
| Validation | UTMOS / Whisper WER / VCTK parallel | `infer_flow.py` のみ、metric なし |

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

### [04-2] (P0) 🚧 mixed_precision=bf16 の検証と再有効化
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

## 関連文書
- [03_converter_model.md](03_converter_model.md)
- [07_unimplemented_phases.md](07_unimplemented_phases.md)
