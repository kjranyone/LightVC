#!/bin/bash
# 段 P101 の測り(training/ の中で実行・前段 v3.1・SNV_FEAT=data/snv_p_feat_v31): devA の行 → 製品と v31_tgt との対の差 → 女声 → 女声の写し → 遅れの probe。
#   bash snv_diag/p101_eval.sh <名前> '<系の指定>' [行の枠]
set -euo pipefail
NAME=$1
SYS=$2
SLOT=${3:-1}
LOCK=../results/snv_p97/gpu_row.lock
[ "$SLOT" = 1 ] || LOCK=../results/snv_p97/gpu_row$SLOT.lock
O=results/snv_p101
ROW=$O/dev_rows_devA_s0_$NAME.json
export SNV_FEAT=data/snv_p_feat_v31 SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500 SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda
exec 9> $LOCK
flock 9
SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$SYS" set=devA n=64 out=$ROW prereg=results/snv_p101/prereg.yaml > ../$O/dev_$NAME.log 2>&1
uv run --no-sync python snv_diag/p96_pair_diff.py ../results/snv_p95/dev_rows_devA_s0_k3gkv_guard_esm.json ../$ROW ../$O/pair_$NAME.json
uv run --no-sync python snv_diag/p96_pair_diff.py ../results/snv_p100/dev_rows_devA_s0_v31_tgt.json ../$ROW ../$O/pair_${NAME}_vs_v31tgt.json
uv run --no-sync python snv_diag/p3_ff_probe.py "$SYS" js=3,4,5 out=$O/ff216_$NAME.json > ../$O/ff216_$NAME.log 2>&1
SNV_PROBE_SYS="$SYS" uv run --no-sync python snv_diag/p98_probe.py p101_$NAME > ../$O/probe_$NAME.log 2>&1
