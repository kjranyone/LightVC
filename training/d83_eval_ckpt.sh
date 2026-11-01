#!/usr/bin/env bash
# D83: 出力部のスナップショット 1 つを、事前登録(results/rvtab_full/prereg.yaml)の成功条件で解析的に判定する。
#   bash d83_eval_ckpt.sh <ema_XXXk.pt> <名前> <scratchpad>
set -euo pipefail
CK=$1; NM=$2; S=$3
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
OUT=../results/rvtab_full/eval_$NM; mkdir -p $OUT
EX=results/rvtab_full/export_$NM
[ -f ../$EX/weights.bin ] || CUDA_VISIBLE_DEVICES= uv run python export_rvoc.py --ckpt $CK --out ../$EX
uv run python d66_rvtab_eval.py --s $S --ft $CK --out_dir $S/d83_$NM/copy > $OUT/copy.log 2>&1
uv run python d83_pesq.py --dir $S/d83_$NM/copy --conds BASE,FT --out $OUT/pesq_copy.json
mkdir -p $S/d83_$NM/conv; ln -sfn $S/d80/_work $S/d83_$NM/conv/_work
CUDA_VISIBLE_DEVICES= uv run python d80_parallel_conv.py --out_dir $S/d83_$NM/conv --tag d83_conv_$NM --conds "TAB=--resid 0;W10=--resid 1.0;C10=--resid 1.0 --color;C05=--resid 0.5 --color" --extra "--rvoc $EX" --workers 10 > $OUT/conv.log 2>&1
CUDA_VISIBLE_DEVICES= uv run python d82_conv_natural.py --d80 $S/d83_$NM/conv --conds SELFCOPY,TAB,W10,C10,C05 --workers 10 > $OUT/natural.log 2>&1
cp ../results/conv_p0/d82.json $OUT/natural.json
for f in $S/d83_$NM/conv/m*_TAB.wav; do b=$(basename $f _TAB.wav); M=${b%%_*}; k=${b##*_t}; ln -sf /home/kojirotanaka/kjranyone/LightVC/data/male_tts_v4l_par/$M/f$k.wav $S/d83_$NM/conv/${b}_SRC.wav; done
uv run python d23_cer.py --dirs $S/d83_$NM/conv --names $NM --re '(m\d+_[0-9a-f]+_t\d+)_(.+)\.wav' --out $OUT/cer.json > $OUT/cer.log 2>&1
echo "== $NM"; cat $OUT/pesq_copy.json | head -c 300; echo; grep -E "^(cov|rms|mod) " $OUT/natural.log; grep "^$NM" $OUT/cer.log | cut -c1-400
