#!/bin/bash
# 段 P97 step c の診断の行の列(training/ の中で実行): 引数 1 = 待つ PID(無ければ 0)。n240 の列は直ぐ、y240 の列は PID の終了の後に走らせる。
cd "$(dirname "$0")/.."
WAITPID=${1:-0}
run() { for a in "$@"; do IFS=: read t k s e <<< "$a"; bash snv_diag/p97_eval_ek.sh $t $k $s $e > ../results/snv_p97/eval_${t}_${k}k_$s.log 2>&1; echo "done $a $(date -u +%T)"; done; }
run snv_p97_n240:30:know:0.25,0.5,0.25,0 snv_p97_n240:30:2pt:0.5,0.5,0,0 &
while [ "$WAITPID" != 0 ] && kill -0 $WAITPID 2>/dev/null; do sleep 30; done
run snv_p97_y240:30:know:0.25,0.5,0.25,0 snv_p97_y240:30:2pt:0.5,0.5,0,0 &
wait
