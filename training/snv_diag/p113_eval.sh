#!/usr/bin/env bash
# 段 P113 の測り(training/ の中で実行): 1 系の腕(b|c)の、音素の無い発声の物差し 2 種と devA の行(+0・+12)。
set -euo pipefail
ARM=$1; TAG=snv_p113${ARM}_c; O=results/snv_p113
exec 9> ../results/snv_p97/gpu_row.lock
flock 9
unset SNV_FEAT SNV_WIN_CAP_MS SNV_W_YIN SNV_TAU_UV SNV_F0G SNV_ENVC
export SNV_ESM=1 SNV_DEV=cuda
uv run --no-sync python snv_diag/p110_nv2.py p113${ARM} sys=ens:$TAG > ../$O/nv2_$ARM.log 2>&1
uv run --no-sync python snv_diag/p110_nv.py p113${ARM}_speechrefs sys=ens:$TAG > ../$O/nv_${ARM}_speechrefs.log 2>&1
uv run --no-sync python snv_diag/p110_nv.py p113${ARM}_nvrefs sys=ens:$TAG refs=nv > ../$O/nv_${ARM}_nvrefs.log 2>&1
SNV_KS=0,12 SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$TAG+mask:snv_p45_gate_t80+fb:anch" set=devA n=64 out=$O/dev_rows_devA_k012_$ARM.json prereg=$O/prereg.yaml > ../$O/dev_$ARM.log 2>&1
