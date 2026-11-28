#!/usr/bin/env bash
set -euo pipefail
P126='ens:snv_p125a_c+mask:snv_p126_gate_i80+fb:anch|snv_p125b_c+mask:snv_p126_gate_i80+fb:anch|snv_p125e_c+mask:snv_p126_gate_i80+fb:anch'
O=results/snv_p126
RCK=results/snv_p123_s/snap/ema_20k.pt
B0_JSON=../results/snv_p100/dev_rows_devA_s0_v31_tgt.json
B0_K12_JSON=../results/snv_p124/dev_rows_devA_k12_B0.json
P125_JSON=../results/snv_p125/dev_rows_devA_P125_anch.json
P125_K12_JSON=../results/snv_p125/dev_rows_devA_k12_P125_anch.json
mkdir -p ../$O ../results/snv_p126_gate_i80

uv run --no-sync python -c '
import torch
from pathlib import Path
st = torch.load(Path("../results/snv_p45_gate_t80/last.pt"), map_location="cpu", weights_only=False)
st["cfg"] = dict(st["cfg"]) | {"inst_mask": True}
st["tag"] = "snv_p126_gate_i80"
st["prereg"] = "results/snv_p126/prereg.yaml"
torch.save(st, Path("../results/snv_p126_gate_i80/last.pt"))
print("saved snv_p126_gate_i80/last.pt", st["cfg"])
'

echo "=== [0/6] CPU Ship Gate (future invariance & static latency ledger) ==="
SNV_FEAT=data/snv_p_feat_v4 SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500 SNV_F0V4=1 SNV_ESM=1 SNV_ENVC=1 SNV_RCK=$RCK \
  uv run --no-sync python snv_p3_ship.py \
    --cpc results/snv_p125a_cpc/last.pt --pca results/snv_p125a_cpc/pca_c.npy --cb results/snv_p125a_cpc/codebook_k128.npz \
    --ckpt results/snv_p125a_c/last.pt --play wst --lags 0 --shrink 0.6 \
    --ens "results/snv_p125b_cpc/last.pt;results/snv_p125b_cpc/pca_c.npy;results/snv_p125b_cpc/codebook_k128.npz;results/snv_p125b_c/last.pt,results/snv_p125e_cpc/last.pt;results/snv_p125e_cpc/pca_c.npy;results/snv_p125e_cpc/codebook_k128.npz;results/snv_p125e_c/last.pt" \
    --mask results/snv_p126_gate_i80/last.pt --dec myin --fb anch --out $O/ship

exec 9> ../results/snv_p97/gpu_row.lock
flock 9
export OMP_NUM_THREADS=4 SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda
export SNV_FEAT=data/snv_p_feat_v4 SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500
export SNV_RCK=$RCK SNV_F0V4=1

echo "=== [1/6] bgm_dev_probe P126 (mus10,lead0,claplead15) ==="
uv run --no-sync python snv_diag/i13_bgm_dev_probe.py "$P126" n=6 pool=dev mix=0 conds=mus10,lead0,claplead15 out=$O/bgm_P126.json > ../$O/bgm_P126.log 2>&1
tail -n 45 ../$O/bgm_P126.log

echo "=== [2/6] P120 & probe4 P126 ==="
uv run --no-sync python snv_diag/p126_eval.py

echo "=== [3/6] ff216 P126 (js=3,4,5 & js=6,7,8) ==="
uv run --no-sync python snv_diag/p3_ff_probe.py "$P126" js=3,4,5 out=$O/ff216a_P126.json
uv run --no-sync python snv_diag/p3_ff_probe.py "$P126" js=6,7,8 out=$O/ff216b_P126.json

echo "=== [4/6] devA (k=0,7) P126 ==="
SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$P126" set=devA n=64 out=$O/dev_rows_devA_P126.json prereg=$O/prereg.yaml > ../$O/dev_P126.log 2>&1
cat ../$O/dev_P126.log
uv run --no-sync python snv_diag/p96_pair_diff.py $B0_JSON ../$O/dev_rows_devA_P126.json ../$O/pair_devA_P126_vs_B0.json
uv run --no-sync python snv_diag/p96_pair_diff.py $P125_JSON ../$O/dev_rows_devA_P126.json ../$O/pair_devA_P126_vs_P125.json

echo "=== [5/6] devA (k=12) P126 ==="
SNV_KS=12 SNV_GUARD=1 uv run --no-sync python snv_dev_eval.py "$P126" set=devA n=64 out=$O/dev_rows_devA_k12_P126.json prereg=$O/prereg.yaml > ../$O/dev_k12_P126.log 2>&1
cat ../$O/dev_k12_P126.log
uv run --no-sync python snv_diag/p96_pair_diff.py $B0_K12_JSON ../$O/dev_rows_devA_k12_P126.json ../$O/pair_devA_k12_P126_vs_B0.json
uv run --no-sync python snv_diag/p96_pair_diff.py $P125_K12_JSON ../$O/dev_rows_devA_k12_P126.json ../$O/pair_devA_k12_P126_vs_P125.json

echo "=== [6/6] ALL STAGE P126 GATES DONE ==="
