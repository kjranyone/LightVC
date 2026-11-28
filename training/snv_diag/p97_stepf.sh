#!/bin/bash
# 段 P97 step f の列(training/ の中で・1 本ずつ): f_y_4 の遅れと折れ → 通った形の devA の行 → step e の残りの行。
cd "$(dirname "$0")/.."
LVC_EK=1,0,0,0 LVC_C0X=4 uv run --no-sync python snv_diag/p97_probe.py f_y_4 results/snv_p97_y240/export_30k > ../results/snv_p97/probe_f_y_4.log 2>&1
while pgrep -f "[s]nv_dev_eval.py" > /dev/null; do sleep 30; done
c0x() { SNV_C0X=$3 LVC_C0X=$3 flock ../results/snv_p97/gpu_row.lock bash snv_diag/p97_eval_ek.sh $1 30 $2 1,0,0,0 > ../results/snv_p97/eval_$1_30k_$2.log 2>&1; echo "done $1 $2 $(date -u +%T)"; }
c0x snv_p97_n240 f4 4
uv run --no-sync python - <<'PY' && c0x snv_p97_y240 f4 4
import json, sys
s = json.load(open("../results/snv_p97/probe/f_y_4/probe.json"))["summary"]
ref = [13.99, 15.71, 14.57, 13.03]
sys.exit(0 if 13.77 - s["env_centroid_median"] >= 4.0 and all(a <= b + 1.0 for a, b in zip(s["zz95_conv"], ref)) and s["zz95_phrase"] <= 15.1 else 1)
PY
c0x snv_p97_n240 f10 10
for a in snv_p97_n240:e0510:0.5,1.0 snv_p97_y240:e0510:0.5,1.0; do IFS=: read t s e <<< "$a"
  SNV_E1E=$e LVC_E1E=$e flock ../results/snv_p97/gpu_row.lock bash snv_diag/p97_eval_ek.sh $t 30 $s 0.25,0.5,0.25,0 > ../results/snv_p97/eval_${t}_30k_$s.log 2>&1; echo "done $a $(date -u +%T)"; done
