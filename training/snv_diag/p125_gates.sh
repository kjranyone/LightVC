#!/usr/bin/env bash
set -euo pipefail
P125_C0BG='ens:snv_p125a_c+mask:snv_p45_gate_t80+fb:c0bg0.45_0.15,anch|snv_p125b_c+mask:snv_p45_gate_t80+fb:c0bg0.45_0.15,anch|snv_p125e_c+mask:snv_p45_gate_t80+fb:c0bg0.45_0.15,anch'
P125_ANCH='ens:snv_p125a_c+mask:snv_p45_gate_t80+fb:anch|snv_p125b_c+mask:snv_p45_gate_t80+fb:anch|snv_p125e_c+mask:snv_p45_gate_t80+fb:anch'
O=results/snv_p125
RCK=results/snv_p123_s/snap/ema_20k.pt
B0_JSON=../results/snv_p100/dev_rows_devA_s0_v31_tgt.json
B0_K12_JSON=../results/snv_p124/dev_rows_devA_k12_B0.json
R3_JSON=../results/snv_p124/dev_rows_devA_R3.json
R3_K12_JSON=../results/snv_p124/dev_rows_devA_k12_R3.json
mkdir -p ../$O
exec 9> ../results/snv_p97/gpu_row.lock
flock 9
export OMP_NUM_THREADS=4 SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda
export SNV_FEAT=data/snv_p_feat_v4 SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500
export SNV_RCK=$RCK SNV_F0V4=1

echo "=== [1/6] P120 & probe4 (P125_c0bg & P125_anch) ==="
uv run --no-sync python snv_diag/p125_eval.py

echo "=== [2/6] ff216 P125_c0bg & P125_anch (js=3,4,5 & js=6,7,8) ==="
uv run --no-sync python snv_diag/p3_ff_probe.py "$P125_C0BG" js=3,4,5 out=$O/ff216a_P125_c0bg.json
uv run --no-sync python snv_diag/p3_ff_probe.py "$P125_C0BG" js=6,7,8 out=$O/ff216b_P125_c0bg.json
uv run --no-sync python snv_diag/p3_ff_probe.py "$P125_ANCH" js=3,4,5 out=$O/ff216a_P125_anch.json
uv run --no-sync python snv_diag/p3_ff_probe.py "$P125_ANCH" js=6,7,8 out=$O/ff216b_P125_anch.json

echo "=== [3/6] bgm_dev_probe P125_c0bg & P125_anch (mus10,lead0,claplead15) ==="
uv run --no-sync python snv_diag/i13_bgm_dev_probe.py "$P125_ANCH" "$P125_C0BG" n=6 pool=dev mix=0 conds=mus10,lead0,claplead15 out=$O/bgm_P125.json > ../$O/bgm_P125.log 2>&1
tail -n 55 ../$O/bgm_P125.log

echo "=== [4/6] devA (k=0,7) P125_c0bg & P125_anch ==="
SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$P125_C0BG" set=devA n=64 out=$O/dev_rows_devA_P125_c0bg.json prereg=$O/prereg.yaml > ../$O/dev_P125_c0bg.log 2>&1
cat ../$O/dev_P125_c0bg.log
uv run --no-sync python snv_diag/p96_pair_diff.py $B0_JSON ../$O/dev_rows_devA_P125_c0bg.json ../$O/pair_devA_P125_c0bg_vs_B0.json
uv run --no-sync python snv_diag/p96_pair_diff.py $R3_JSON ../$O/dev_rows_devA_P125_c0bg.json ../$O/pair_devA_P125_c0bg_vs_R3.json

SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$P125_ANCH" set=devA n=64 out=$O/dev_rows_devA_P125_anch.json prereg=$O/prereg.yaml > ../$O/dev_P125_anch.log 2>&1
cat ../$O/dev_P125_anch.log
uv run --no-sync python snv_diag/p96_pair_diff.py $B0_JSON ../$O/dev_rows_devA_P125_anch.json ../$O/pair_devA_P125_anch_vs_B0.json

echo "=== [5/6] devA (k=12) P125_c0bg & P125_anch ==="
SNV_KS=12 SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$P125_C0BG" set=devA n=64 out=$O/dev_rows_devA_k12_P125_c0bg.json prereg=$O/prereg.yaml > ../$O/dev_k12_P125_c0bg.log 2>&1
cat ../$O/dev_k12_P125_c0bg.log
uv run --no-sync python snv_diag/p96_pair_diff.py $B0_K12_JSON ../$O/dev_rows_devA_k12_P125_c0bg.json ../$O/pair_devA_k12_P125_c0bg_vs_B0.json
uv run --no-sync python snv_diag/p96_pair_diff.py $R3_K12_JSON ../$O/dev_rows_devA_k12_P125_c0bg.json ../$O/pair_devA_k12_P125_c0bg_vs_R3.json

SNV_KS=12 SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$P125_ANCH" set=devA n=64 out=$O/dev_rows_devA_k12_P125_anch.json prereg=$O/prereg.yaml > ../$O/dev_k12_P125_anch.log 2>&1
cat ../$O/dev_k12_P125_anch.log
uv run --no-sync python snv_diag/p96_pair_diff.py $B0_K12_JSON ../$O/dev_rows_devA_k12_P125_anch.json ../$O/pair_devA_k12_P125_anch_vs_B0.json

echo "=== [6/6] ALL STAGE P125 GATES DONE ==="
