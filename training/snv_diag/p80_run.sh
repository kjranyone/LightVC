#!/usr/bin/env bash
# 段 P80(results/snv_p80/prereg.yaml): 対照 ct と往復の項 bt を、同じ VM の上で並べて学ぶ。training/ の中で実行する。
set -u
A="--cpc results/snv_p30_dcpc/last.pt --pca results/snv_p30/pca_c.npy --cb results/snv_p30/codebook_k128.npz --lags 0 --shrink 0.6 --init results/snv_p30_c/last.pt --steps 12000 --lr 1e-4 --seed 0 --prereg results/snv_p80/prereg.yaml --ship results/snv_p16/ship.json"
( set -o pipefail; uv run --no-sync python train_snv_p16.py --tag diag_snv_p80_ct $A --w_bt 0 2>&1 | sed -u 's/^/[ct] /' ) & P1=$!
( set -o pipefail; uv run --no-sync python train_snv_p16.py --tag diag_snv_p80_bt $A --w_bt 1.0 2>&1 | sed -u 's/^/[bt] /' ) & P2=$!
wait $P1; E1=$?
wait $P2; E2=$?
echo "exit ct $E1 bt $E2"
exit $((E1 + E2))
