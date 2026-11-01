#!/usr/bin/env bash
# A3(diag)の流れ: p と n の学習(並行)→ 評価。事前登録 = results/diag_snv_a3_p/prereg.yaml
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
uv run --no-sync python train_snv_a3.py --perturb p --tag diag_snv_a3_p > ../results/diag_snv_a3_p/train.log 2>&1 &
PP=$!
uv run --no-sync python train_snv_a3.py --perturb n --tag diag_snv_a3_n > ../results/diag_snv_a3_n/train.log 2>&1 &
PN=$!
wait $PP; RP=$?
wait $PN; RN=$?
echo "train done $(date -u +%FT%TZ) p=$RP n=$RN"
if [ "$RP" -eq 0 ] && [ "$RN" -eq 0 ]; then
  uv run --no-sync python snv_a3_eval.py --tags diag_snv_a3_p diag_snv_a3_n > ../results/diag_snv_a3_p/eval.log 2>&1
  echo "eval done $(date -u +%FT%TZ) rc=$?"
fi
