#!/usr/bin/env bash
# 段 P103 の測り(training/ の中で実行)。前段は製品の v3(スイッチなし・SNV_FEAT 既定)。
#   bash snv_diag/p103_eval.sh [行の枠]
set -euo pipefail
SLOT=${1:-1}
LOCK=../results/snv_p97/gpu_row.lock
[ "$SLOT" = 1 ] || LOCK=../results/snv_p97/gpu_row$SLOT.lock
G='+mask:snv_p45_gate_t80+fb:anch'
SYS="ens:snv_p103a_c$G|snv_p103b_c$G|snv_p103e_c$G"
O=results/snv_p103; ROW=$O/dev_rows_devA_s0_aa.json
unset SNV_FEAT SNV_WIN_CAP_MS SNV_W_YIN SNV_TAU_UV
export SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda
exec 9> $LOCK
flock 9
SNV_PROBE_SYS="$SYS" uv run --no-sync python snv_diag/p98_probe.py p103_aa > ../$O/probe_aa.log 2>&1
SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$SYS" set=devA n=64 out=$ROW prereg=$O/prereg.yaml > ../$O/dev_aa.log 2>&1
uv run --no-sync python snv_diag/p96_pair_diff.py ../results/snv_p95/dev_rows_devA_s0_k3gkv_guard_esm.json ../$ROW ../$O/pair_aa.json
uv run --no-sync python snv_diag/p96_pair_diff.py ../results/snv_p101/dev_rows_devA_s0_k3g.json ../$ROW ../$O/pair_aa_vs_p101.json
uv run --no-sync python snv_diag/p3_ff_probe.py "$SYS" js=3,4,5 out=$O/ff216_aa.json > ../$O/ff216_aa.log 2>&1
