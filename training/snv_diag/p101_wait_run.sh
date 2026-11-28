#!/usr/bin/env bash
# 段 P101: Colab の VM の上で、大きい入力(手元から塊で届く)を待つ → sha256 を確かめて展開 → 2 本の鎖を並走。カーネルを塞いだまま待つ(空のままの長い転送は VM が回収される)。training/ の中で実行する。
set -u
ST=/content/_lvc
echo "$(date -u +%T) 大きい入力を待つ"
n=0
while [ ! -f $ST/big.go ]; do
  sleep 20; n=$((n + 1))
  [ $((n % 15)) = 0 ] && echo "$(date -u +%T) 届いた塊 $(ls $ST/big.[0-9]* 2>/dev/null | wc -l)"
  [ $n -gt 720 ] && { echo "待ちの上限"; exit 21; }
done
want=$(cat $ST/big.go)
got=$(cat $ST/big.[0-9]* | sha256sum | cut -d' ' -f1)
[ "$want" = "$got" ] || { echo "sha256 が合わない $want $got"; exit 22; }
cat $ST/big.[0-9]* | tar -xf - -C /content/LightVC || exit 23
rm -f $ST/big.[0-9]*
echo "$(date -u +%T) 展開 ok"; du -sh /content/LightVC/data/*
exec bash snv_diag/p101_pair.sh
