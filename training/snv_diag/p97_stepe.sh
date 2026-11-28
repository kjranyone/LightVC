#!/bin/bash
# 段 P97 step e の残りの devA の行(training/ の中で・1 本ずつ)。先に走っている devA の行が終わるまで待つ。
cd "$(dirname "$0")/.."
while pgrep -f "[s]nv_dev_eval.py" > /dev/null; do sleep 30; done
for a in snv_p97_y240:e0505:0.5,0.5 snv_p97_n240:e0510:0.5,1.0 snv_p97_y240:e0510:0.5,1.0; do IFS=: read t s e <<< "$a"
  SNV_E1E=$e LVC_E1E=$e bash snv_diag/p97_eval_ek.sh $t 30 $s 0.25,0.5,0.25,0 > ../results/snv_p97/eval_${t}_30k_$s.log 2>&1; echo "done $a $(date -u +%T)"; done
