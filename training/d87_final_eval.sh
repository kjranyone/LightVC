#!/usr/bin/env bash
# rvtab_full の最終評価(事前登録の成功条件 + 話者らしさ + 実女声の製品経路)。学習の終了を待ってから全部走らせる。
#   bash d87_final_eval.sh <学習の PID> <scratchpad>
set -uo pipefail
PID=$1; S=$2
cd "$(dirname "$0")"
while kill -0 $PID 2>/dev/null; do sleep 30; done
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
CK=$PWD/../results/rvtab_full/snap/ema_360k.pt
OUT=../results/rvtab_full/eval_360k
sed -i 's/--workers 6 > \$OUT\/conv.log/--workers 10 > $OUT\/conv.log/; s/--workers 6 > \$OUT\/natural.log/--workers 10 > $OUT\/natural.log/' d83_eval_ckpt.sh
bash d83_eval_ckpt.sh $CK 360k $S
CUDA_VISIBLE_DEVICES= uv run python d84_identity_margin.py --dir $S/d83_360k/conv --tag d84_360k --conds SELFCOPY,TAB,W10,C10,C05 --workers 10 > $OUT/identity.log 2>&1
CUDA_VISIBLE_DEVICES= uv run python d85_product_eval.py --s $S --out_dir $S/d85_360k --extra "--rvoc results/rvtab_full/export_360k" --conds "TAB=--resid 0;W10=--resid 1.0;C10=--resid 1.0 --color;C05=--resid 0.5 --color" --workers 10 > $OUT/product.log 2>&1
uv run python d4b_lda_verifier.py --wav_dir $S/d85_360k --out $OUT/lda_real.json > $OUT/lda_real.log 2>&1
uv run python d23_cer.py --dirs $S/d85_360k --names real360k --out $OUT/cer_real.json > $OUT/cer_real.log 2>&1
CUDA_VISIBLE_DEVICES= uv run python d86_natural_real.py --s $S --dir $S/d85_360k --conds TAB,W10,C10,C05 --out $OUT/natural_real.json --workers 10 > $OUT/natural_real.log 2>&1
echo FINAL_DONE
