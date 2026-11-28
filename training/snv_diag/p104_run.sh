#!/usr/bin/env bash
# 段 P104(results/snv_p104/prereg.yaml・学習なし): 段 P100 の候補の、確認の組の行と写しの追加。training/ の中で実行する。
set -euo pipefail
K3G='ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch'
O=results/snv_p104
exec 9> ../results/snv_p97/gpu_row.lock
flock 9
export SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda
unset SNV_FEAT SNV_WIN_CAP_MS SNV_W_YIN SNV_TAU_UV
row() { SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$K3G" set=devA skip=64 n=158 out=$O/dev_rows_conf_s0_$1.json prereg=$O/prereg.yaml > ../$O/conf_$1.log 2>&1; }
ff() { uv run --no-sync python snv_diag/p3_ff_probe.py "$K3G" js=6,7,8 out=$O/ff216b_$1.json > ../$O/ff216b_$1.log 2>&1; }
ff product
row product
export SNV_FEAT=data/snv_p_feat_p100 SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500
ff v31
row v31
uv run --no-sync python snv_diag/p96_pair_diff.py ../$O/dev_rows_conf_s0_product.json ../$O/dev_rows_conf_s0_v31.json ../$O/pair_conf_v31.json
