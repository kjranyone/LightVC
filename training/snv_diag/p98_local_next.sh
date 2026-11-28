#!/bin/bash
# 手元の GPU: 前の腕の学習が終わったら次の腕を起動する(training/ の中で)。引数 = 待つ tag・次の tag・次の引数…
cd "$(dirname "$0")/.."
W=$1; T=$2; shift 2
until grep -q "^done" ../results/$W/train.log; do sleep 30; done
MPLBACKEND=agg uv run --no-sync python train_snv_p98.py --tag $T "$@" > ../results/$T/train.log 2>&1
