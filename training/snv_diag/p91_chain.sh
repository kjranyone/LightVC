#!/usr/bin/env bash
# 段 P91・P93: 固定の窓の包絡(data/snv_p_feat_st・data/snv_p_feat_st_x16)で符号を一から 30k → PCA・符号帳 → 回路を一から 40k(符号の入力だけ固定の窓・rend なし)。training/ の中で実行する。
#   [DV=<--dview か空> PSEED=<符号の種か空>] bash snv_diag/p91_chain.sh <名(例 snv_p91e)> <prereg> [回路の種]
set -u
NM=$1; PR=$2; D=results/${NM}_cpc
SNV_FEAT=data/snv_p_feat_st SNV_XFEAT=data/snv_p_feat_st_x16 uv run --no-sync python train_snv_p2_cpc.py --tag ${NM}_cpc --aug --xview ${DV---dview} ${PSEED:+--seed $PSEED} --steps 30000 --prereg $PR || exit 11
SNV_FEAT=data/snv_p_feat_st uv run --no-sync python snv_diag/p8b_pca.py cpc=$D/last.pt out=$D/pca_c.npy || exit 12
SNV_FEAT=data/snv_p_feat_st OMP_NUM_THREADS=1 uv run --no-sync python snv_cmn.py fit K=128 lags=0 shrink=0.6 cpc=$D/last.pt pca=$D/pca_c.npy out=$D/codebook_k128.npz || exit 13
SNV_ENVC=1 exec uv run --no-sync python train_snv_p16.py --tag ${NM}_c --cpc $D/last.pt --pca $D/pca_c.npy --cb $D/codebook_k128.npz --lags 0 --shrink 0.6 --steps 40000 --seed "${3:-0}" --no_rend --prereg $PR --ship results/snv_p16/ship.json
