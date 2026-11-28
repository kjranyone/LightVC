#!/usr/bin/env bash
# 段 P114: Colab の VM の上で大きい入力を待つ → sha256 → 展開 → 描画器の学び足し。training/ の中で実行する。引数 = <tag> <msub 0|1> [steps]
set -u
ST=/content/_lvc
n=0
while [ ! -f $ST/big.go ]; do sleep 20; n=$((n + 1)); [ $((n % 15)) = 0 ] && echo "$(date -u +%T) 届いた塊 $(ls $ST/big.[0-9]* 2>/dev/null | wc -l)"; [ $n -gt 1080 ] && exit 21; done
[ "$(cat $ST/big.go)" = "$(cat $ST/big.[0-9]* | sha256sum | cut -d' ' -f1)" ] || { echo "sha256 が合わない"; exit 22; }
cat $ST/big.[0-9]* | tar -xf - -C /content/LightVC || exit 23
rm -f $ST/big.[0-9]*
MPLBACKEND=agg exec uv run --no-sync python train_snv_p114.py --tag $1 --msub $2 --delay 240 --smooth 0.25 --steps ${3:-10000} --pdrop ${4:-0} --workers 8
