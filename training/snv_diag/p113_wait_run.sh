#!/usr/bin/env bash
# 段 P113: Colab の VM の上で大きい入力を待つ → sha256 → 展開 → 視点の置き場を組む → 鎖を 1 本。training/ の中で実行する。引数 = <腕 b|c>
set -u
ST=/content/_lvc
n=0
while [ ! -f $ST/big.go ]; do sleep 20; n=$((n + 1)); [ $((n % 15)) = 0 ] && echo "$(date -u +%T) 届いた塊 $(ls $ST/big.[0-9]* 2>/dev/null | wc -l)"; [ $n -gt 1080 ] && exit 21; done
[ "$(cat $ST/big.go)" = "$(cat $ST/big.[0-9]* | sha256sum | cut -d' ' -f1)" ] || { echo "sha256 が合わない"; exit 22; }
cat $ST/big.[0-9]* | tar -xf - -C /content/LightVC || exit 23
rm -f $ST/big.[0-9]*
python3 snv_diag/p113_xdir.py || exit 24
bash snv_diag/p113_chain.sh $1 > ../results/snv_p113$1.chain.log 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do sleep 300; echo "$(date -u +%T) $1: $(grep -a '^step' ../results/snv_p113$1.chain.log | tail -n 1 | cut -c1-100)"; done
wait $P; R=$?; echo "exit $1 $R"; exit $R
