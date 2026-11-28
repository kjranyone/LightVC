#!/bin/bash
# 段 P97 の測りの列(training/ の中で・1 本ずつ): 引数 = "<tag>:<k>" の並び。重みが届くまで待つ。先に走っている devA の行が終わるまで待つ。
set -u
while pgrep -f "[s]nv_dev_eval.py" > /dev/null; do sleep 30; done
for a in "$@"; do
  t=${a%%:*}; k=${a##*:}
  until [ -f ../results/$t/snap/ema_${k}k.pt ]; do sleep 60; done
  sleep 20
  bash snv_diag/p97_eval.sh $t $k > ../results/snv_p97/eval_${t}_${k}k.log 2>&1
  echo "done $a $(date -u +%T)"
done
