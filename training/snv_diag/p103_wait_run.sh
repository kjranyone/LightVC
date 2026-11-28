#!/usr/bin/env bash
# 段 P103: Colab の VM の上で、大きい入力(手元から塊で届く)を待つ → sha256 を確かめて展開 → 鎖を 1 本。training/ の中で実行する。引数 = <名> <式>
set -u
ST=/content/_lvc
echo "$(date -u +%T) 大きい入力を待つ"
n=0
while [ ! -f $ST/big.go ]; do
  sleep 20; n=$((n + 1))
  [ $((n % 15)) = 0 ] && echo "$(date -u +%T) 届いた塊 $(ls $ST/big.[0-9]* 2>/dev/null | wc -l)"
  [ $n -gt 1080 ] && { echo "待ちの上限"; exit 21; }
done
want=$(cat $ST/big.go)
got=$(cat $ST/big.[0-9]* | sha256sum | cut -d' ' -f1)
[ "$want" = "$got" ] || { echo "sha256 が合わない $want $got"; exit 22; }
cat $ST/big.[0-9]* | tar -xf - -C /content/LightVC || exit 23
rm -f $ST/big.[0-9]*
echo "$(date -u +%T) 展開 ok"; du -sh /content/LightVC/data/*
bash snv_diag/p103_chain.sh "$@" > ../results/$1.chain.log 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do sleep 300; echo "$(date -u +%T) $1: $(grep -a '^step' ../results/$1.chain.log | tail -n 1 | cut -c1-110)"; done
wait $P; R=$?; echo "exit $1 $R"; exit $R
