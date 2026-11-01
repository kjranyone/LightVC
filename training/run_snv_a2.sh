#!/usr/bin/env bash
# A2(diag)の流れ: 生成の終了(PID で待つ)→ SPARC の特徴 → mel → g と x の学習(並行)→ 評価。事前登録 = results/diag_snv_a2_g/prereg.yaml
set -u
cd "$(dirname "$0")/.."
GEN_PID=$(cat data/snv_e1/gen.pids)
while kill -0 "$GEN_PID" 2>/dev/null; do sleep 30; done
echo "gen done $(date -u +%FT%TZ) wavs=$(find data/snv_e1 -name 't*.wav' | wc -l)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ../sparc_env/bin/python training/snv_a2_sparc_feats.py > data/snv_e1_feat_sparc_pass2.log 2>&1
RC=$?
echo "sparc done $(date -u +%FT%TZ) rc=$RC npz=$(find data/snv_e1_feat/sparc -name '*.npz' | wc -l)"
(cd training && uv run --no-sync python snv_a2_mel.py > ../data/snv_e1_feat_mel_pass2.log 2>&1)
RC=$?
echo "mel done $(date -u +%FT%TZ) rc=$RC npy=$(find data/snv_e1_feat/mel -name '*.npy' | wc -l)"
cd training
uv run --no-sync python train_snv_a2.py --mode g --tag diag_snv_a2_g > ../results/diag_snv_a2_g/train.log 2>&1 &
PG=$!
uv run --no-sync python train_snv_a2.py --mode x --tag diag_snv_a2_x > ../results/diag_snv_a2_x/train.log 2>&1 &
PX=$!
wait $PG; RG=$?
wait $PX; RX=$?
echo "train done $(date -u +%FT%TZ) g=$RG x=$RX"
if [ "$RG" -eq 0 ] && [ "$RX" -eq 0 ]; then
  uv run --no-sync python snv_a2_eval.py --tags diag_snv_a2_g diag_snv_a2_x > ../results/diag_snv_a2_g/eval.log 2>&1
  RC=$?
  echo "eval done $(date -u +%FT%TZ) rc=$RC"
fi
