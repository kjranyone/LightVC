"""D604(推論だけ): 版の対の低い帯(198〜397 Hz・D602 と同じ水準をそろえた LTAS)と帯ごとの差。
    uv run python d604_lowpair.py --pairs OURS5Y59-OURS5Y58 ... --out <json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import NPROC, H, boot
from d602_lowband import CF, LOW, ltas


def job(p: Path) -> np.ndarray:
    return ltas(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = sorted({v for p in a.pairs for v in p.split("-")})
    files = [(nm, s, re.match(r"(p\d+_t(\d+))_", f.name).groups(), f) for s in a.sets for nm in names for f in sorted((H / s / "d570" / nm).glob("p*_t*.wav"))]
    with Pool(NPROC) as pl:
        res = pl.map(job, [f for *_, f in files])
    L: dict = {}
    for (nm, s, (key, ti), _), d in zip(files, res):
        L.setdefault(nm, {})[(s, key)] = (int(ti), d)
    rng = np.random.default_rng(604)
    rep: dict = {"bands_hz": [round(float(c), 1) for c in CF], "pairs": {}}
    for p in a.pairs:
        A_, B_ = p.split("-")
        ks = sorted(set(L[A_]) & set(L[B_]))
        tg = np.array([f"{k[0]}|{L[A_][k][0]}" for k in ks])
        D = np.array([L[A_][k][1] - L[B_][k][1] for k in ks])
        rep["pairs"][p] = {"n": len(ks), "low_mean": [round(float(D[:, LOW].mean()), 3)] + boot(D[:, LOW].mean(1), tg, rng), "per_band": [round(float(v), 2) for v in D.mean(0)]}
        print(p, rep["pairs"][p]["low_mean"], " ".join(f"{v:5.2f}" for v in rep["pairs"][p]["per_band"]), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
