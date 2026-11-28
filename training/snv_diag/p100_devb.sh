#!/bin/bash
# 段 P100 の確認の組 devB の行(training/ の中で実行・学び直しなし): 前段のスイッチは環境変数で渡す(名前 product = スイッチなし)。
#   [SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 ...] bash snv_diag/p100_devb.sh <名前> [行の枠 1〜3]
set -euo pipefail
NAME=$1
SLOT=${2:-1}
K3G='ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch'
LOCK=../results/snv_p97/gpu_row.lock
[ "$SLOT" = 1 ] || LOCK=../results/snv_p97/gpu_row$SLOT.lock
ROW=results/snv_p100/dev_rows_devB_s0_$NAME.json
exec 9> $LOCK
flock 9
SNV_ESM=1 SNV_GUARD=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_dev_eval.py "$K3G" set=devB n=64 out=$ROW prereg=results/snv_p100/prereg.yaml > ../results/snv_p100/devB_$NAME.log 2>&1
