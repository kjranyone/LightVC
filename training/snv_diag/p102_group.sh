#!/usr/bin/env bash
# 段 P102: 1 台で回路を並べて走らせる(training/ の中で実行)。引数 = <腕>:<系> の並び(例 m0:a m0:b m1:b)。
set -u
PIDS=()
for j in "$@"; do
  bash snv_diag/p102_circ.sh ${j%%:*} ${j##*:} > ../results/snv_p102${j%%:*}_${j##*:}.circ.log 2>&1 &
  PIDS+=($!)
done
alive() { for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && return 0; done; return 1; }
while alive; do
  sleep 300
  for j in "$@"; do echo "$(date -u +%T) $j: $(grep -a '^step' ../results/snv_p102${j%%:*}_${j##*:}.circ.log | tail -1 | cut -c1-90)"; done
done
RC=0
J=("$@")
for i in "${!PIDS[@]}"; do wait ${PIDS[$i]}; r=$?; echo "exit ${J[$i]} $r"; [ $r = 0 ] || RC=1; done
exit $RC
