#!/usr/bin/env bash
# 段 P83(results/snv_p83/prereg.yaml): 雑音を足した符号で回路を学び足す 2 つの腕(σ 0.3・0.6)を、同じ VM の上で並べる。training/ の中で実行する。
set -u
A="--cpc results/snv_p30_dcpc/last.pt --pca results/snv_p30/pca_c.npy --cb results/snv_p30/codebook_k128.npz --lags 0 --shrink 0.6 --init results/snv_p30_c/last.pt --steps 12000 --lr 1e-4 --seed 0 --prereg results/snv_p83/prereg.yaml --ship results/snv_p16/ship.json"
( set -o pipefail; uv run --no-sync python train_snv_p16.py --tag diag_snv_p83_n3 $A --cn 0.3 2>&1 | sed -u 's/^/[n3] /' ) & P1=$!
( set -o pipefail; uv run --no-sync python train_snv_p16.py --tag diag_snv_p83_n6 $A --cn 0.6 2>&1 | sed -u 's/^/[n6] /' ) & P2=$!
wait $P1; E1=$?
wait $P2; E2=$?
echo "exit n3 $E1 n6 $E2"
exit $((E1 + E2))
