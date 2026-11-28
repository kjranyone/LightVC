#!/usr/bin/env bash
# 段 P102(results/snv_p102/prereg.yaml): 回路 1 本(training/ の中で実行)。目標の側 = T.FEAT(v3)・源の側 = v3.1(--sdir・--xdir)・符号は段 P101 のもの。
#   bash snv_diag/p102_circ.sh m0|mh|m1 a|b|e [steps]
ARM=$1; KIND=$2; SK=${3:-40000}
PR=results/snv_p102/prereg.yaml; SH=results/snv_p101/ship/ship.json
case $KIND in a) D=results/snv_p101a_cpc;; b) D=results/snv_p101r_b_cpc;; e) D=results/snv_p101r_e_cpc;; *) exit 2;; esac
case $ARM in
  m0) S=data/snv_p_feat_v31; X=data/snv_p_feat_x16_v31; DL=0;;
  mh) S=data/snv_p_feat_v31h; X=data/snv_p_feat_x16_v31h; DL=0;;
  m1) S=data/snv_p_feat_v31; X=data/snv_p_feat_x16_v31; DL=1;;
  *) exit 2;;
esac
unset SNV_FEAT SNV_WIN_CAP_MS SNV_W_YIN SNV_TAU_UV
EX=; [ "$KIND" = e ] && { EX=--no_rend; export SNV_ENVC=1; }
SM=; TAG=snv_p102${ARM}_${KIND}_c
[ "$SK" -lt 1000 ] && { SM=--smoke; TAG=snv_p102sm_${ARM}_${KIND}_c; }
exec uv run --no-sync python train_snv_p16.py --tag $TAG --cpc $D/last.pt --pca $D/pca_c.npy --cb $D/codebook_k128.npz --circ x --lags 0 --shrink 0.6 --steps $SK --seed 0 --xdir $X --sdir $S --dly $DL $EX --prereg $PR --ship $SH $SM
