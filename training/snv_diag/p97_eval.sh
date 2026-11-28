#!/bin/bash
# 段 P97 の腕の測り(training/ の中で実行): 書き出し → Rust の製品の経路の遅れと折れ(p97_probe)→ devA の標準の行 → 製品の行との対の差。
#   bash snv_diag/p97_eval.sh <tag> <k(千歩)> [probe だけなら 1]
set -euo pipefail
TAG=$1
K=$2
ONLY=${3:-0}
CK=results/$TAG/snap/ema_${K}k.pt
EX=results/$TAG/export_${K}k
NAME=${TAG}_${K}k
K3G='ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch'
test -f ../$EX/manifest.json || CUDA_VISIBLE_DEVICES= uv run --no-sync python export_rvoc.py --ckpt ../$CK --out ../$EX
uv run --no-sync python snv_diag/p97_probe.py $NAME $EX 2>&1 | grep -v Warn | tail -5
if [ "$ONLY" = "1" ]; then exit 0; fi
ROW=results/snv_p97/dev_rows_devA_s0_$NAME.json
SNV_RCK=$CK SNV_ESM=1 SNV_GUARD=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_dev_eval.py "$K3G" set=devA n=64 out=$ROW prereg=results/snv_p97/prereg.yaml > ../results/snv_p97/dev_$NAME.log 2>&1
uv run --no-sync python snv_diag/p96_pair_diff.py ../results/snv_p95/dev_rows_devA_s0_k3gkv_guard_esm.json ../$ROW ../results/snv_p97/pair_$NAME.json
