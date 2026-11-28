#!/bin/bash
# 段 P100 G3(training/ の中で実行): 族の中の線。ff = 女声 → 女声の写し(216 本)・iro = 検証の Irodori の声の行。前段のスイッチは環境変数で渡す。
set -euo pipefail
KIND=$1
NAME=$2
SLOT=${3:-1}
K3G='ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch'
LOCK=../results/snv_p97/gpu_row.lock
[ "$SLOT" = 1 ] || LOCK=../results/snv_p97/gpu_row$SLOT.lock
exec 9> $LOCK
flock 9
if [ "$KIND" = ff ]; then
  SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p3_ff_probe.py "$K3G" js=3,4,5 out=results/snv_p100/ff216_$NAME.json > ../results/snv_p100/ff216_$NAME.log 2>&1
else
  SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/i13_bgm_dev_probe.py "$K3G" n=6 conds=mus10,clap0 out=results/snv_p100/irodori_$NAME.json > ../results/snv_p100/irodori_$NAME.log 2>&1
fi
