"""D435: フレーム周期(200 Hz)の包絡変調の線(eval_nvoc.amline・d109 のガビ軸・本物の中央値 −0.31・[p15, p85] = [−1.37, 0.54])が ppgvoc の系統のどこで生じたか。
D434 の女声 → 女声の関門で B / D / D2 がこの統計だけで落ちた(1.63 / 2.06 / 2.31・製品 −0.11)。各走行の学習の中の評価の書き出し(ev_*/ の SR・PV・FF)を群ごとに先頭 n 本で測る。
    uv run python d435_amline_lineage.py --out ../results/conv_p0/d435.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent


def one(p: str) -> tuple:
    from d6_unit_cov import load48
    from eval_nvoc import amline
    x = load48(p)[: 8 * 48000]
    return p, amline(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--runs", nargs="*", default=["1", "4", "6", "9", "11", "14", "18", "21", "24", "26", "27", "29", "30", "34", "36", "38", "39a", "39b", "40", "41", "42"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs, keys = [], []
    for r in a.runs:
        for ev in sorted((ROOT / f"results/diag_ppgvoc{r}").glob("ev_*")):
            for g, pat in (("SR", "h*_SR.wav"), ("PV", "p*_PV.wav"), ("FF", "f*_FF.wav")):
                fs = sorted(ev.glob(pat))[: a.n]
                if fs:
                    keys.append((r, ev.name[3:], g, [str(f) for f in fs]))
                    jobs += [str(f) for f in fs]
    with Pool(a.workers) as pl:
        val = dict(pl.imap_unordered(one, jobs, chunksize=4))
    rep: dict = {}
    for r, st, g, fs in keys:
        v = np.array([val[f] for f in fs])
        rep.setdefault(r, {}).setdefault(st, {})[g] = round(float(np.median(v)), 2)
    for r, d in rep.items():
        print(r, json.dumps(d), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
