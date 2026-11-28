#!/bin/bash
# 段 P98 の腕の測り(training/ の中で実行・2 本目の行の枠・学習が走っていないときだけ): 遅れと折れ(PyTorch の経路・p98_probe)→ devA の標準の行 → 製品の行との対の差。
#   bash snv_diag/p98_eval.sh <tag> <k(千歩)>
set -euo pipefail
TAG=$1
K=$2
CK=results/$TAG/snap/ema_${K}k.pt
NAME=${TAG}_${K}k
K3G='ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch'
until [ -f ../$CK ]; do sleep 60; done
sleep 20
ROW=results/snv_p98/dev_rows_devA_s0_$NAME.json
exec 9> ../results/snv_p97/gpu_row2.lock
flock 9
SNV_RCK=$CK SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p98_probe.py $NAME 2>&1 | grep -v -i "warn\\|WeightNorm" | tail -n 1
SNV_RCK=$CK SNV_ESM=1 SNV_GUARD=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_dev_eval.py "$K3G" set=devA n=64 out=$ROW prereg=results/snv_p98/prereg.yaml > ../results/snv_p98/dev_$NAME.log 2>&1
uv run --no-sync python snv_diag/p96_pair_diff.py ../results/snv_p95/dev_rows_devA_s0_k3gkv_guard_esm.json ../$ROW ../results/snv_p98/pair_$NAME.json
