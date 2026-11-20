#!/usr/bin/env bash
# 段 P81・P82: 符号を snv_p30_dcpc から 10k 歩学び足す → PCA 32 と級 128 の符号帳を作り直す → 回路を一から 40k 歩。training/ の中で実行する。
#   [INIT=<ckpt か空> STEPS=<歩数>] bash snv_diag/p81_chain.sh <名(例 diag_snv_p81c)> <prereg> [符号の追加の引数 ...]   INIT を空にすると符号を一から学ぶ
set -u
NM=$1; PR=$2; shift 2
D=results/${NM}_cpc
INIT=${INIT-results/snv_p30_dcpc/last.pt}; STEPS=${STEPS-10000}
uv run --no-sync python train_snv_p2_cpc.py --tag ${NM}_cpc --aug --xview ${DV---dview} "$@" ${INIT:+--init $INIT} --steps $STEPS --prereg $PR || exit 11
uv run --no-sync python snv_diag/p8b_pca.py cpc=$D/last.pt out=$D/pca_c.npy || exit 12
OMP_NUM_THREADS=1 uv run --no-sync python snv_cmn.py fit K=128 lags=0 shrink=0.6 cpc=$D/last.pt pca=$D/pca_c.npy out=$D/codebook_k128.npz || exit 13
exec bash snv_diag/p81_circ.sh ${NM}_c $D $PR ${CSEED:-0}
