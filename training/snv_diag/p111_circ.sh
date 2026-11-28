#!/usr/bin/env bash
# 段 P111(results/snv_p111/prereg.yaml): 式 b(snv_p30 の符号)の回路を 1 本・構造の参照に同じ声の音素の無い発声を足す。training/ の中で実行する。
#   bash snv_diag/p111_circ.sh <腕: x = 直通の道あり / q = 番地だけ / b = 細い直通の道 + 時間の高域通過> [steps]
set -u
ARM=$1; SK=${2:-40000}
PR=results/snv_p111/prereg.yaml; SH=results/snv_p93/ship_k3g/ship.json
unset SNV_FEAT SNV_WIN_CAP_MS SNV_W_YIN SNV_TAU_UV SNV_ENVC
SM=; TAG=snv_p111${ARM}_c
[ "$SK" -lt 1000 ] && { SM=--smoke; TAG=snv_p111sm_${ARM}_c; }
exec uv run --no-sync python train_snv_p16.py --tag $TAG --cpc results/snv_p30_dcpc/last.pt --pca results/snv_p30/pca_c.npy --cb results/snv_p30/codebook_k128.npz --circ $ARM --lags 0 --shrink 0.6 --steps $SK --seed 0 --xdir data/snv_p_feat_x16 --rnv 0.5 --prereg $PR --ship $SH $SM
