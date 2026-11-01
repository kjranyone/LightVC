"""D608(推論だけ・腕の比較): 版の対の H1 − H2(D607 と同じ最小二乗・r = 入力の harvest × 2^(k/12) の遅れ合わせ)の 150〜210 Hz のファイルごとの中央値の対の差と、高さの区間ごとの中央値。
    uv run python d608_h1pair.py --pairs OURS5Y62-OURS5Y59 ... --out <json>
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
from d607_h1h2 import FB, job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--band", type=float, nargs=2, default=[150.0, 210.0], help="対の差を取る高さの帯(Hz)")
    ap.add_argument("--lowbins", action="store_true", help="100-130・130-150 の区間も")
    a = ap.parse_args()
    names = sorted({v for p in a.pairs for v in p.split("-")})
    its = []
    for s in a.sets:
        for nm in names:
            k = int(re.match(r"OURS(\d+)", nm).group(1))
            for f in sorted((H / s / "d570" / nm).glob("p*_t*.wav")):
                key, ti = re.match(r"(p\d+_t(\d+))_", f.name).groups()
                its.append((nm, s, key, int(ti), k, f))
    with Pool(NPROC) as pl:
        res = pl.map(job, its, chunksize=2)
    R: dict = {}
    for (nm, s, key, ti, k, _), x, r, d12 in res:
        n = min(len(r), len(d12))
        ok = (r[:n] > 0) & np.isfinite(d12[:n])
        R.setdefault(nm, {})[(s, key)] = (ti, r[:n][ok], d12[:n][ok])
    rep: dict = {"by_bin": {}, "pairs": {}}
    for nm, D in R.items():
        rep["by_bin"][nm] = {}
        for lo, hi in ([(100, 130), (130, 150)] if a.lowbins else []) + FB:
            d = np.concatenate([v[2][(v[1] >= lo) & (v[1] < hi)] for v in D.values()])
            rep["by_bin"][nm][f"{lo}-{hi}"] = round(float(np.median(d)), 2) if len(d) >= 50 else None
        print(nm, rep["by_bin"][nm], flush=True)
    rng = np.random.default_rng(608)

    def fmed(v: tuple) -> float:
        m = (v[1] >= a.band[0]) & (v[1] < a.band[1])
        return float(np.median(v[2][m])) if m.sum() >= 30 else np.nan
    for p in a.pairs:
        A_, B_ = p.split("-")
        ks = sorted(set(R[A_]) & set(R[B_]))
        d = np.array([fmed(R[A_][k]) - fmed(R[B_][k]) for k in ks])
        tg = np.array([f"{k[0]}|{R[A_][k][0]}" for k in ks])
        ok = np.isfinite(d)
        rep["pairs"][p] = {"n": int(ok.sum()), f"diff_{a.band[0]:g}_{a.band[1]:g}": round(float(d[ok].mean()), 3), "ci95": boot(d[ok], tg[ok], rng)}
        print(p, rep["pairs"][p], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
