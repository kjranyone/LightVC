#!/bin/bash
# 段 P102 の測り(training/ の中で実行): 操作者の側 = 前段 v3.1 のスイッチ・目標の側 = SNV_FEAT の既定(v3)。devA の行 → 製品・段 P101 の K3G' との対の差 → 女声 → 女声の写し → 遅れの probe(目標は v3 で読む)。
#   bash snv_diag/p102_eval.sh m0|mh|m1 [行の枠]
set -euo pipefail
ARM=$1
SLOT=${2:-1}
G='+mask:snv_p45_gate_t80+fb:anch'
SYS="ens:snv_p102${ARM}_a_c$G|snv_p102${ARM}_b_c$G|snv_p102${ARM}_e_c$G"
LOCK=../results/snv_p97/gpu_row.lock
[ "$SLOT" = 1 ] || LOCK=../results/snv_p97/gpu_row$SLOT.lock
O=results/snv_p102
ROW=$O/dev_rows_devA_s0_$ARM.json
unset SNV_FEAT
export SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500 SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda
exec 9> $LOCK
flock 9
SNV_PROBE_TGV3=1 SNV_PROBE_SYS="$SYS" uv run --no-sync python snv_diag/p98_probe.py p102_$ARM > ../$O/probe_$ARM.log 2>&1
SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$SYS" set=devA n=64 out=$ROW prereg=$O/prereg.yaml > ../$O/dev_$ARM.log 2>&1
uv run --no-sync python snv_diag/p96_pair_diff.py ../results/snv_p95/dev_rows_devA_s0_k3gkv_guard_esm.json ../$ROW ../$O/pair_$ARM.json
uv run --no-sync python snv_diag/p96_pair_diff.py ../results/snv_p101/dev_rows_devA_s0_k3g.json ../$ROW ../$O/pair_${ARM}_vs_p101.json
uv run --no-sync python snv_diag/p3_ff_probe.py "$SYS" js=3,4,5 out=$O/ff216_$ARM.json > ../$O/ff216_$ARM.log 2>&1
