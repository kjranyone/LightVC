#!/bin/bash
# 段 P100 G2(training/ の中で実行): 本物の男声 + 伴奏・手拍子の行。前段のスイッチは環境変数で渡す(名前 product = スイッチなし)。
set -euo pipefail
NAME=$1
SLOT=${2:-3}
K3G='ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch'
LOCK=../results/snv_p97/gpu_row.lock
[ "$SLOT" = 1 ] || LOCK=../results/snv_p97/gpu_row$SLOT.lock
exec 9> $LOCK
flock 9
SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p37_cv_music.py "$K3G" set=devA n=24 conds=mus10,mus0,clap0 out=results/snv_p100/cv_music_$NAME.json prereg=results/snv_p100/prereg.yaml > ../results/snv_p100/cv_music_$NAME.log 2>&1
