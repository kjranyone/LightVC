#!/bin/bash
# 段 P101 の関門(training/ の中で実行・R_cand のときだけ・前段 v3.1・SNV_FEAT=data/snv_p_feat_v31)。
#   bash snv_diag/p101_gate.sh devb|music|iro <名前> '<系の指定>' [行の枠]
# 対照(製品)は段 P100 の行: dev_rows_devB_s0_product.json・cv_music_product.json・irodori_product.json。
set -euo pipefail
KIND=$1
NAME=$2
SYS=$3
SLOT=${4:-1}
LOCK=../results/snv_p97/gpu_row.lock
[ "$SLOT" = 1 ] || LOCK=../results/snv_p97/gpu_row$SLOT.lock
O=results/snv_p101
export SNV_FEAT=data/snv_p_feat_v31 SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500 SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda
exec 9> $LOCK
flock 9
case $KIND in
  devb)
    ROW=$O/dev_rows_devB_s0_$NAME.json
    SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$SYS" set=devB n=64 out=$ROW prereg=$O/prereg.yaml > ../$O/devB_$NAME.log 2>&1
    uv run --no-sync python snv_diag/p96_pair_diff.py ../results/snv_p100/dev_rows_devB_s0_product.json ../$ROW ../$O/pair_devB_$NAME.json
    ;;
  music)
    OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p37_cv_music.py "$SYS" set=devA n=24 conds=mus10,mus0,clap0 out=$O/cv_music_$NAME.json prereg=$O/prereg.yaml > ../$O/cv_music_$NAME.log 2>&1
    ;;
  iro)
    OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/i13_bgm_dev_probe.py "$SYS" n=6 conds=mus10,clap0 out=$O/irodori_$NAME.json > ../$O/irodori_$NAME.log 2>&1
    ;;
esac
