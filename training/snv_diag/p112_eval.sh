#!/usr/bin/env bash
# 段 P112 の測り(training/ の中で実行): 製品の前段(v3.1)で、製品の K3G か K3G-b の devA の行(+0・+12)。
#   bash snv_diag/p112_eval.sh <product|b>
set -euo pipefail
W=$1; G='+mask:snv_p45_gate_t80+fb:anch'; O=results/snv_p112
case $W in
  product) SYS="ens:snv_p24_c$G|snv_p30_c$G|snv_p91e_c$G";;
  b) SYS="ens:snv_p112a_c$G|snv_p111b_c$G|snv_p112e_c$G";;
  *) exit 2;;
esac
exec 9> ../results/snv_p97/gpu_row.lock
flock 9
unset SNV_F0G
export SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500 SNV_FEAT=data/snv_p_feat_p100 SNV_ESM=1 SNV_ENVC=1 SNV_DEV=cuda
SNV_KS=0,12 SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$SYS" set=devA n=64 out=$O/dev_rows_devA_k012_$W.json prereg=$O/prereg.yaml > ../$O/dev_$W.log 2>&1
if [ "$W" = b ]; then
  uv run --no-sync python snv_diag/p3_ff_probe.py "$SYS" js=3,4,5 out=$O/ff216_b.json > ../$O/ff216_b.log 2>&1
  unset SNV_FEAT SNV_WIN_CAP_MS SNV_W_YIN SNV_TAU_UV
  uv run --no-sync python snv_diag/p110_nv.py p112b_speechrefs "sys=$SYS" > ../$O/nv_b_speechrefs.log 2>&1
  uv run --no-sync python snv_diag/p110_nv.py p112b_nvrefs "sys=$SYS" refs=nv > ../$O/nv_b_nvrefs.log 2>&1
fi
