#!/usr/bin/env bash
# A3c(diag)の流れ: 話者の統計の差し替えつきの学習 → 評価。事前登録 = results/diag_snv_a3c_w/prereg.yaml
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4
uv run --no-sync python train_snv_a3.py --perturb n --swap --leak_trip 0.30 --tag diag_snv_a3c_w > ../results/diag_snv_a3c_w/train.log 2>&1
R=$?
echo "train done $(date -u +%FT%TZ) rc=$R"
if [ "$R" -eq 0 ]; then
  uv run --no-sync python snv_a3_eval.py --tags diag_snv_a3c_w > ../results/diag_snv_a3c_w/eval.log 2>&1
  echo "eval done $(date -u +%FT%TZ) rc=$?"
fi
