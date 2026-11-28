#!/usr/bin/env bash
# 段 P112(results/snv_p112/prereg.yaml): 型 b(CircB・直通 8 次元)+ 構造の参照に音素の無い発声、の回路を、K3G の式 a(snv_p24)・e(snv_p91e)で 1 本ずつ。training/ の中で実行する。
#   bash snv_diag/p112_circ.sh <a|e> [steps]
set -u
KIND=$1; SK=${2:-40000}
PR=results/snv_p112/prereg.yaml; SH=results/snv_p93/ship_k3g/ship.json
unset SNV_FEAT SNV_WIN_CAP_MS SNV_W_YIN SNV_TAU_UV SNV_ENVC
case $KIND in
  a) CP=results/snv_p24_xcpc/last.pt; PC=results/snv_p24/pca_c_xcpc.npy; CB=results/snv_p24/codebook_k128.npz; EX=;;
  e) CP=results/snv_p91e_cpc/last.pt; PC=results/snv_p91e_cpc/pca_c.npy; CB=results/snv_p91e_cpc/codebook_k128.npz; EX=--no_rend; export SNV_ENVC=1;;
  *) exit 2;;
esac
SM=; TAG=snv_p112${KIND}_c
[ "$SK" -lt 1000 ] && { SM=--smoke; TAG=snv_p112sm_${KIND}_c; }
exec uv run --no-sync python train_snv_p16.py --tag $TAG --cpc $CP --pca $PC --cb $CB --circ b --bneck 8 --lags 0 --shrink 0.6 --steps $SK --seed 0 --xdir data/snv_p_feat_x16 --rnv 0.5 $EX --prereg $PR --ship $SH $SM
