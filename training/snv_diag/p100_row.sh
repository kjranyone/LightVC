#!/bin/bash
# 段 P100 step a の devA の行(training/ の中で実行・学び直しなし・製品の描画器): 前段のスイッチ(SNV_WIN_CAP_MS など)は環境変数で渡す。
#   SNV_WIN_CAP_MS=10 bash snv_diag/p100_row.sh cap10 [行の枠 1〜3]
set -euo pipefail
NAME=$1
SLOT=${2:-1}
K3G='ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch'
LOCK=../results/snv_p97/gpu_row.lock
[ "$SLOT" = 1 ] || LOCK=../results/snv_p97/gpu_row$SLOT.lock
ROW=results/snv_p100/dev_rows_devA_s0_$NAME.json
exec 9> $LOCK
flock 9
SNV_ESM=1 SNV_GUARD=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_dev_eval.py "$K3G" set=devA n=64 out=$ROW prereg=results/snv_p100/prereg.yaml > ../results/snv_p100/dev_$NAME.log 2>&1
uv run --no-sync python snv_diag/p96_pair_diff.py ../results/snv_p95/dev_rows_devA_s0_k3gkv_guard_esm.json ../$ROW ../results/snv_p100/pair_$NAME.json
