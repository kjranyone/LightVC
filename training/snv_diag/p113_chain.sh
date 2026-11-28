#!/usr/bin/env bash
# 段 P113(results/snv_p113/prereg.yaml・式 b): training/ の中で実行する。
#   bash snv_diag/p113_chain.sh b [steps_cpc] [steps_c]   = 符号も回路も新しい視点(snv_p113b_cpc → snv_p113b_c)
#   bash snv_diag/p113_chain.sh c [-] [steps_c]           = 回路だけ新しい視点(符号は製品 snv_p30_dcpc・snv_p113c_c)
set -u
ARM=$1; SC=${2:-30000}; SK=${3:-40000}
PR=results/snv_p113/prereg.yaml; SH=results/snv_p93/ship_k3g/ship.json; X=data/snv_p_feat_xbt1
unset SNV_FEAT SNV_WIN_CAP_MS SNV_W_YIN SNV_TAU_UV SNV_ENVC SNV_XFEAT
SM=; [ "$SK" -lt 1000 ] && SM=--smoke
NM=snv_p113${ARM}; [ -n "$SM" ] && NM=snv_p113sm_${ARM}
if [ "$ARM" = b ]; then
  D=results/${NM}_cpc
  SNV_XFEAT=$X uv run --no-sync python train_snv_p2_cpc.py --tag ${NM}_cpc --aug --xview --dview --steps $SC --prereg $PR $SM || exit 11
  uv run --no-sync python snv_diag/p8b_pca.py cpc=$D/last.pt out=$D/pca_c.npy || exit 12
  OMP_NUM_THREADS=1 uv run --no-sync python snv_cmn.py fit K=128 lags=0 shrink=0.6 cpc=$D/last.pt pca=$D/pca_c.npy out=$D/codebook_k128.npz || exit 13
  CP=$D/last.pt; PC=$D/pca_c.npy; CB=$D/codebook_k128.npz
else
  CP=results/snv_p30_dcpc/last.pt; PC=results/snv_p30/pca_c.npy; CB=results/snv_p30/codebook_k128.npz
fi
exec uv run --no-sync python train_snv_p16.py --tag ${NM}_c --cpc $CP --pca $PC --cb $CB --circ b --bneck 8 --lags 0 --shrink 0.6 --steps $SK --seed 0 --xdir $X --rnv 0.5 --prereg $PR --ship $SH $SM
