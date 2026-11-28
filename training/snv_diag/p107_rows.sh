#!/usr/bin/env bash
# 段 P107 step e(training/ の中で実行): 製品と直し 2(SNV_F0G=2)の devA の行・半音 +0 と +12。
set -euo pipefail
K3G='ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch'
O=results/snv_p107
exec 9> ../results/snv_p97/gpu_row.lock
flock 9
export SNV_KS=0,12 SNV_ESM=1 SNV_GUARD=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda
unset SNV_FEAT SNV_F0G
uv run --no-sync python snv_dev_eval.py "$K3G" set=devA n=64 out=$O/dev_rows_devA_k012_product.json prereg=$O/prereg.yaml > ../$O/dev_product.log 2>&1
SNV_F0G=2 uv run --no-sync python snv_dev_eval.py "$K3G" set=devA n=64 out=$O/dev_rows_devA_k012_g2.json prereg=$O/prereg.yaml > ../$O/dev_g2.log 2>&1
