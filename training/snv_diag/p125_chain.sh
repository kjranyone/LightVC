#!/usr/bin/env bash
set -euo pipefail
NM=$1; KIND=$2; SC=${3:-30000}; SK=${4:-40000}
PR=results/snv_p125/prereg.yaml; SH=results/snv_p125/ship/ship.json; D=results/${NM}_cpc
F=data/snv_p_feat_v4; X=data/snv_p_feat_x16_v4
export SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500 SNV_F0V4=1
CF=$F; CX=$X; DV=--dview; EX=
[ "$KIND" = a ] && DV=
[ "$KIND" = e ] && { CF=data/snv_p_feat_st_v4; CX=data/snv_p_feat_st_x16_v4; EX=--no_rend; export SNV_ENVC=1; }
SM=; [ "$SC" -lt 1000 ] && SM=--smoke
SNV_FEAT=$CF SNV_XFEAT=$CX uv run --no-sync python train_snv_p2_cpc.py --tag ${NM}_cpc --aug --xview $DV --steps $SC --prereg $PR $SM || exit 11
SNV_FEAT=$CF uv run --no-sync python snv_diag/p8b_pca.py cpc=$D/last.pt out=$D/pca_c.npy || exit 12
SNV_FEAT=$CF OMP_NUM_THREADS=1 uv run --no-sync python snv_cmn.py fit K=128 lags=0 shrink=0.6 cpc=$D/last.pt pca=$D/pca_c.npy out=$D/codebook_k128.npz || exit 13
SNV_FEAT=$F exec uv run --no-sync python train_snv_p16.py --tag ${NM}_c --cpc $D/last.pt --pca $D/pca_c.npy --cb $D/codebook_k128.npz --circ x --lags 0 --shrink 0.6 --steps $SK --seed 0 --xdir $X $EX --prereg $PR --ship $SH $SM
