"""D584(推論だけ): 固定の半音での本人らしさの天井 = 目標本人の参照の録音(<組>/d37/_work/ref_TTT.wav の先頭 8 s・cen_reg とは別の録音)を WORLD で再合成し、
高さだけを操作者の SRC 8 s の f0 の中央値 × 2^(k/12) へ一律に移したもの(REFK<k>)と、移さないもの(REFW)。D571 の REALK と同じ作り(並行の文でない点だけ違う)。
出力 = <組>/d570/REFK<k>/<pNNNN_tTTT>_REFK<k>.wav・REFW/…(ピーク 0.9)。
    uv run python d584_ceiling.py --sets dev2 dev3 dev --ks 5 7 12
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import scipy.signal as ss
import soundfile as sf

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
SR = 48000


def f0med(x: np.ndarray) -> float:
    f0, _ = pw.harvest(ss.resample_poly(x.astype(np.float64), 1, 3), 16000, f0_floor=50.0, f0_ceil=1100.0, frame_period=5.0)
    return float(np.median(f0[f0 > 0]))


def one(job: tuple) -> int:
    from d6_unit_cov import load48
    S, key, ti, ks = job
    od = S / "d570"
    outs = {f"REFK{k}": 2 ** (k / 12) for k in ks} | {"REFW": None}
    if all((od / nm / f"{key}_{nm}.wav").exists() for nm in outs):
        return 0
    xs = load48(S / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * SR]
    xt = load48(S / "d37" / "_work" / f"ref_{ti:03d}.wav").astype(np.float64)[: 8 * SR]
    fp = 5.0
    f0, t = pw.harvest(ss.resample_poly(xt, 1, 3), 16000, f0_floor=50.0, f0_ceil=1100.0, frame_period=fp)
    sp = pw.cheaptrick(xt, f0, t, SR)
    ap = pw.d4c(xt, f0, t, SR)
    mt = float(np.median(f0[f0 > 0]))
    mo = f0med(xs)
    n = 0
    for nm, r in outs.items():
        o = od / nm / f"{key}_{nm}.wav"
        if o.exists():
            continue
        ratio = 1.0 if r is None else mo * r / mt
        y = pw.synthesize(f0 * ratio, sp, ap, SR, fp)
        o.parent.mkdir(parents=True, exist_ok=True)
        sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), SR, subtype="PCM_16")
        n += 1
    return n


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--sets", nargs="+", required=True)
    ap_.add_argument("--ks", type=int, nargs="+", default=[5, 7, 12])
    ap_.add_argument("--all", action="store_true", help="組の全部(関門の組の構成で d570/OURS7 が無いとき)")
    a = ap_.parse_args()
    jobs = []
    for s in a.sets:
        S = H / s
        for r in json.loads((S / "devset.json").read_text())["pairs"]:
            key = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if a.all or (S / "d570" / "OURS7" / f"{key}_OURS7.wav").exists():
                jobs.append((S, key, r["ti"], tuple(a.ks)))
    with Pool(int(os.environ.get("LVC_NPROC", "12"))) as pl:
        n = sum(pl.imap_unordered(one, jobs))
    print("written", n, "jobs", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
