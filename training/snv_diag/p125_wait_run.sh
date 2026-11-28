#!/usr/bin/env bash
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
bash snv_diag/p125_chain.sh "$@" > ../results/$1.chain.log 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do sleep 180; echo "$(date -u +%T) $1: $(grep -aE '^step|^eval|^var64|^符号帳|TRIPWIRE' ../results/$1.chain.log | tail -n 1 | cut -c1-120)"; done
wait $P; R=$?; echo "exit $1 $R"; exit $R
