"""D616(推論だけ): 有声の立ち上がりからの位置ごとの H1 − H2(D607 の最小二乗)を、候補と同じ高さ(150〜300 Hz)の本物の女声(devff2 の実女声の入力 FF_SRC・目標本人 CEN)・男声の入力 SRC と比べる。
立ち上がり = r > 0 の連続の始まり(r = 候補は入力の harvest × 2^(k/12) の遅れ合わせ・本物は自分の harvest)。位置 1〜2 / 3〜5 / 6〜10 / 21 以上。規則は results/conv_p0/d616/prereg.yaml。
    uv run python d616_onset_h1.py --out ../results/conv_p0/d616/onset_h1.json
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

from d592_refkc import NPROC, H
from d607_h1h2 import job

BINS = {"p1_2": (1, 2), "p3_5": (3, 5), "p6_10": (6, 10), "steady": (21, 10 ** 6)}


def per_file(r: np.ndarray, d: np.ndarray) -> dict:
    n = min(len(r), len(d))
    r, d = r[:n], d[:n]
    run = np.zeros(n, int)
    for i in range(n):
        run[i] = run[i - 1] + 1 if r[i] > 0 and i > 0 else int(r[i] > 0)
    ok = np.isfinite(d) & (r >= 150) & (r < 300)
    return {b: d[ok & (run >= lo) & (run <= hi)] for b, (lo, hi) in BINS.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    its = []
    for s in ("dev2", "dev3"):
        tset = set()
        for f in sorted((H / s / "d570" / "OURS5RY58G").glob("p*_t*.wav")):
            key, ti = re.match(r"(p\d+_t(\d+))_", f.name).groups()
            for nm in ("OURS5RY58H4G", "OURS5RY58G"):
                its.append((nm, s, key, int(ti), 5, H / s / "d570" / nm / f"{key}_{nm}.wav"))
            its.append(("SRC", s, key, int(ti), 0, H / s / "d56" / f"{key}_SRC.wav"))
            c = H / s / "cen_reg" / f"cen_{int(ti):03d}.wav"
            if int(ti) not in tset and c.exists():
                tset.add(int(ti))
                its.append(("CEN", s, f"t{int(ti):03d}", int(ti), 0, c))
    for f in sorted((H / "devff2" / "d56").glob("p*_t*_SRC.wav")):
        its.append(("FF_SRC", "devff2", f.stem, 1000 + len(its), 0, f))
    with Pool(NPROC) as pl:
        res = pl.map(job, its, chunksize=2)
    G: dict = {}
    for (nm, s, key, ti, k, _), x, r, d12 in res:
        g = "REAL" if nm in ("FF_SRC", "CEN") else nm
        G.setdefault(g, []).append((f"{s}|{ti}", per_file(r, d12)))
    rng = np.random.default_rng(616)
    rep: dict = {"bins": BINS, "groups": {}}

    def drop(files: list) -> float:
        a_ = np.concatenate([f[1]["p1_2"] for f in files])
        b_ = np.concatenate([f[1]["steady"] for f in files])
        return float(np.median(a_) - np.median(b_)) if len(a_) >= 20 and len(b_) >= 20 else np.nan
    for g, files in G.items():
        med = {b: (round(float(np.median(np.concatenate([f[1][b] for f in files]))), 2) if sum(len(f[1][b]) for f in files) >= 20 else None, int(sum(len(f[1][b]) for f in files))) for b in BINS}
        tg = np.array([f[0] for f in files])
        ut = np.unique(tg)
        idx = {t: np.flatnonzero(tg == t) for t in ut}
        bs = [drop([files[i] for i in np.concatenate([idx[t] for t in rng.choice(ut, len(ut))])]) for _ in range(1000)]
        rep["groups"][g] = {"median_by_pos": {b: v[0] for b, v in med.items()}, "n_by_pos": {b: v[1] for b, v in med.items()},
                            "drop_p1_2_minus_steady": round(drop(files), 2), "drop_ci95": [round(float(np.nanpercentile(bs, 2.5)), 2), round(float(np.nanpercentile(bs, 97.5)), 2)], "_bs": bs}
        print(g, {k_: v_ for k_, v_ in rep["groups"][g].items() if k_ != "_bs"}, flush=True)
    d = np.array(rep["groups"]["OURS5RY58H4G"]["_bs"]) - np.array(rep["groups"]["REAL"]["_bs"])
    rep["O2"] = {"diff_drop_cand_minus_real": round(rep["groups"]["OURS5RY58H4G"]["drop_p1_2_minus_steady"] - rep["groups"]["REAL"]["drop_p1_2_minus_steady"], 2),
                 "ci95": [round(float(np.nanpercentile(d, 2.5)), 2), round(float(np.nanpercentile(d, 97.5)), 2)]}
    print("O2", rep["O2"], flush=True)
    for g in rep["groups"]:
        rep["groups"][g].pop("_bs")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
