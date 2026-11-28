#!/usr/bin/env bash
set -euo pipefail
K3G_BASE='ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch'
K3G_C0BG='ens:snv_p24_c+mask:snv_p45_gate_t80+fb:c0bg0.45_0.15,anch|snv_p30_c+mask:snv_p45_gate_t80+fb:c0bg0.45_0.15,anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:c0bg0.45_0.15,anch'
O=results/snv_p124
RCK=results/snv_p123_s/snap/ema_20k.pt
B0_JSON=../results/snv_p100/dev_rows_devA_s0_v31_tgt.json
mkdir -p ../$O
exec 9> ../results/snv_p97/gpu_row.lock
flock 9
export OMP_NUM_THREADS=4 SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda
export SNV_FEAT=data/snv_p_feat_p100 SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500

echo "=== [1/8] ff216 R3 (js=3,4,5 & js=6,7,8) ==="
SNV_RCK=$RCK SNV_F0V4=1 uv run --no-sync python snv_diag/p3_ff_probe.py "$K3G_C0BG" js=3,4,5 out=$O/ff216a_R3.json
SNV_RCK=$RCK SNV_F0V4=1 uv run --no-sync python snv_diag/p3_ff_probe.py "$K3G_C0BG" js=6,7,8 out=$O/ff216b_R3.json

echo "=== [2/8] ff216 R2 (js=3,4,5 & js=6,7,8) ==="
SNV_RCK=$RCK SNV_F0V4=1 uv run --no-sync python snv_diag/p3_ff_probe.py "$K3G_BASE" js=3,4,5 out=$O/ff216a_R2.json
SNV_RCK=$RCK SNV_F0V4=1 uv run --no-sync python snv_diag/p3_ff_probe.py "$K3G_BASE" js=6,7,8 out=$O/ff216b_R2.json

echo "=== [3/8] bgm_dev_probe B0 vs R2 vs R3 (mus10,lead0,claplead15) ==="
unset SNV_RCK SNV_F0V4
uv run --no-sync python snv_diag/i13_bgm_dev_probe.py "$K3G_BASE" n=6 pool=dev mix=0 conds=mus10,lead0,claplead15 out=$O/bgm_B0.json > ../$O/bgm_B0.log 2>&1
tail -n 35 ../$O/bgm_B0.log
SNV_RCK=$RCK SNV_F0V4=1 uv run --no-sync python snv_diag/i13_bgm_dev_probe.py "$K3G_BASE" "$K3G_C0BG" n=6 pool=dev mix=0 conds=mus10,lead0,claplead15 out=$O/bgm_R2_R3.json > ../$O/bgm_R2_R3.log 2>&1
tail -n 55 ../$O/bgm_R2_R3.log

echo "=== [4/8] devA (k=0,7) R3 ==="
SNV_GUARD=1 SNV_RCK=$RCK SNV_F0V4=1 uv run --no-sync python snv_dev_eval.py "$K3G_C0BG" set=devA n=64 out=$O/dev_rows_devA_R3.json prereg=$O/prereg.yaml > ../$O/dev_R3.log 2>&1
cat ../$O/dev_R3.log
uv run --no-sync python snv_diag/p96_pair_diff.py $B0_JSON ../$O/dev_rows_devA_R3.json ../$O/pair_devA_R3_vs_B0.json

echo "=== [5/8] devA (k=0,7) R2 ==="
SNV_GUARD=1 SNV_RCK=$RCK SNV_F0V4=1 uv run --no-sync python snv_dev_eval.py "$K3G_BASE" set=devA n=64 out=$O/dev_rows_devA_R2.json prereg=$O/prereg.yaml > ../$O/dev_R2.log 2>&1
cat ../$O/dev_R2.log
uv run --no-sync python snv_diag/p96_pair_diff.py $B0_JSON ../$O/dev_rows_devA_R2.json ../$O/pair_devA_R2_vs_B0.json

echo "=== [6/8] devA (k=0,7) R1 ==="
unset SNV_F0V4
SNV_GUARD=1 SNV_RCK=$RCK uv run --no-sync python snv_dev_eval.py "$K3G_BASE" set=devA n=64 out=$O/dev_rows_devA_R1.json prereg=$O/prereg.yaml > ../$O/dev_R1.log 2>&1
cat ../$O/dev_R1.log
uv run --no-sync python snv_diag/p96_pair_diff.py $B0_JSON ../$O/dev_rows_devA_R1.json ../$O/pair_devA_R1_vs_B0.json

echo "=== [7/8] devA (k=12) B0, R3, R2 ==="
unset SNV_RCK SNV_F0V4
SNV_KS=12 SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$K3G_BASE" set=devA n=64 out=$O/dev_rows_devA_k12_B0.json prereg=$O/prereg.yaml > ../$O/dev_k12_B0.log 2>&1
cat ../$O/dev_k12_B0.log
SNV_KS=12 SNV_GUARD=1 SNV_RCK=$RCK SNV_F0V4=1 uv run --no-sync python snv_dev_eval.py "$K3G_C0BG" set=devA n=64 out=$O/dev_rows_devA_k12_R3.json prereg=$O/prereg.yaml > ../$O/dev_k12_R3.log 2>&1
cat ../$O/dev_k12_R3.log
uv run --no-sync python snv_diag/p96_pair_diff.py ../$O/dev_rows_devA_k12_B0.json ../$O/dev_rows_devA_k12_R3.json ../$O/pair_devA_k12_R3_vs_B0.json
SNV_KS=12 SNV_GUARD=1 SNV_RCK=$RCK SNV_F0V4=1 uv run --no-sync python snv_dev_eval.py "$K3G_BASE" set=devA n=64 out=$O/dev_rows_devA_k12_R2.json prereg=$O/prereg.yaml > ../$O/dev_k12_R2.log 2>&1
cat ../$O/dev_k12_R2.log
uv run --no-sync python snv_diag/p96_pair_diff.py ../$O/dev_rows_devA_k12_B0.json ../$O/dev_rows_devA_k12_R2.json ../$O/pair_devA_k12_R2_vs_B0.json

echo "=== [8/8] ALL STAGE P124 GATES DONE ==="
