#!/usr/bin/env bash
# 段 P101: Colab の 1 台で snv_p30 の式と snv_p91e の式の鎖を並べて走らせる。training/ の中で実行する。
set -u
bash snv_diag/p101_chain.sh snv_p101r_b b > ../results/snv_p101r_b.chain.log 2>&1 &
PB=$!
bash snv_diag/p101_chain.sh snv_p101r_e e > ../results/snv_p101r_e.chain.log 2>&1 &
PE=$!
while kill -0 $PB 2>/dev/null || kill -0 $PE 2>/dev/null; do
  sleep 300
  echo "$(date -u +%T) b: $(grep -a '^step' ../results/snv_p101r_b.chain.log | tail -1 | cut -c1-120)"
  echo "$(date -u +%T) e: $(grep -a '^step' ../results/snv_p101r_e.chain.log | tail -1 | cut -c1-120)"
done
wait $PB; RB=$?
wait $PE; RE=$?
echo "exit b $RB e $RE"
[ $RB = 0 ] && [ $RE = 0 ]
