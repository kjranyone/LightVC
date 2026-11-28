#!/bin/bash
# 段 P97 step c(training/ の中で実行): 学び足した描画器の界面に、推論のときだけ因果の平滑の重み(t・t−1・t−2・t−3)を掛けた診断の行。
#   bash snv_diag/p97_eval_ek.sh <tag> <k(千歩)> <名前の後ろ> <重み 例 0.25,0.5,0.25,0>
set -euo pipefail
TAG=$1
K=$2
SUF=$3
EK=$4
CK=results/$TAG/snap/ema_${K}k.pt
EX=results/$TAG/export_${K}k
NAME=${TAG}_${K}k_$SUF
K3G='ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch'
test -f ../$EX/manifest.json || CUDA_VISIBLE_DEVICES= uv run --no-sync python export_rvoc.py --ckpt ../$CK --out ../$EX
LVC_EK=$EK uv run --no-sync python snv_diag/p97_probe.py $NAME $EX 2>&1 | grep -v Warn | tail -1
ROW=results/snv_p97/dev_rows_devA_s0_$NAME.json
SNV_EK=$EK SNV_RCK=$CK SNV_ESM=1 SNV_GUARD=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_dev_eval.py "$K3G" set=devA n=64 out=$ROW prereg=results/snv_p97/prereg.yaml > ../results/snv_p97/dev_$NAME.log 2>&1
uv run --no-sync python snv_diag/p96_pair_diff.py ../results/snv_p95/dev_rows_devA_s0_k3gkv_guard_esm.json ../$ROW ../results/snv_p97/pair_$NAME.json
