#!/usr/bin/env bash
# 段 P112: Colab の VM の上で大きい入力を待つ → sha256 → 展開 → 回路を 1 本。training/ の中で実行する。引数 = <式 a|e>
set -u
ST=/content/_lvc
n=0
while [ ! -f $ST/big.go ]; do sleep 20; n=$((n + 1)); [ $((n % 15)) = 0 ] && echo "$(date -u +%T) 届いた塊 $(ls $ST/big.[0-9]* 2>/dev/null | wc -l)"; [ $n -gt 1080 ] && exit 21; done
[ "$(cat $ST/big.go)" = "$(cat $ST/big.[0-9]* | sha256sum | cut -d' ' -f1)" ] || { echo "sha256 が合わない"; exit 22; }
cat $ST/big.[0-9]* | tar -xf - -C /content/LightVC || exit 23
rm -f $ST/big.[0-9]*
bash snv_diag/p112_circ.sh $1 > ../results/snv_p112$1.circ.log 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do sleep 300; echo "$(date -u +%T) $1: $(grep -a '^step' ../results/snv_p112$1.circ.log | tail -n 1 | cut -c1-100)"; done
wait $P; R=$?; echo "exit $1 $R"; exit $R
