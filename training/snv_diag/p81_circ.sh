#!/usr/bin/env bash
# 段 P81・P82: 作り直した符号(<符号の dir>/last.pt・pca_c.npy・codebook_k128.npz)の上で、回路を M29 の式で一から学ぶ(snv_p30_c と同じ引数)。training/ の中で実行する。
#   bash snv_diag/p81_circ.sh <tag> <符号の dir> <prereg> [種]
set -u
exec uv run --no-sync python train_snv_p16.py --tag "$1" --cpc "$2/last.pt" --pca "$2/pca_c.npy" --cb "$2/codebook_k128.npz" --lags 0 --shrink 0.6 --steps 40000 --seed "${4:-0}" --prereg "$3" --ship results/snv_p16/ship.json
