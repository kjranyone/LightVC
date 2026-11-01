"""D583(記録のみ): 条件の包絡の行に足した差分 dS(R の話者の分)に対する出力の包絡の追従の比。
製品 = cheap(OURS7R) − cheap(OURS7)・包絡だけの出力部 = cheap(XRE) − cheap(TAB0)。dS は <out>/rr_delta/<key>.npz(d571 の 5 ms の格子)。
帯 (c1〜4, c5〜12, c13〜24) × 尺度(3 Hz 以下 / 以上・ゼロ位相)ごとに、有声のフレーム(harvest)で傾き = Σ Δout·dS / Σ dS² と相関。
    uv run python d583_follow.py --out <dev>/d571par --res ../results/conv_p0/d583.json
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import scipy.signal as ss

BANDS = ((0, 4), (4, 12), (12, 24))
PAIRS = {"product": ("OURS7R", "OURS7"), "tabrender": ("XRE", "TAB0")}


def env(path: Path) -> tuple:
    import nvoc as N
    import pae as PA
    import train_rvoc as TR
    import torch
    from d6_unit_cov import load48
    x = load48(path).astype(np.float32)
    n = len(x) // N.HOP
    x = x[: n * N.HOP]
    f0, _ = pw.harvest(ss.resample_poly(x.astype(np.float64), 1, 3), 16000, f0_floor=50.0, f0_ceil=1100.0, frame_period=1000.0 * N.HOP / N.SR)
    f0 = np.pad(f0, (0, max(0, n - len(f0))))[:n]
    xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
    e = PA.envelope(xa, f0)[1:25]
    return e, f0 > 0


def one(job: tuple) -> dict:
    od, key, pairs = job
    dS = np.load(od / "rr_delta" / f"{key}.npz")["dS"].astype(np.float64)
    bl, al = ss.butter(2, 3.0 / 100.0)
    out = {}
    for nm, (a, b) in pairs.items():
        ea, va = env(od / "k7" / f"{key}_{a}.wav")
        eb, vb = env(od / "k7" / f"{key}_{b}.wav")
        n = min(ea.shape[1], eb.shape[1], dS.shape[1])
        d_out = ea[:, :n] - eb[:, :n]
        d_in = dS[:, :n]
        v = va[:n] & vb[:n]
        lo_o = np.stack([ss.filtfilt(bl, al, r) for r in d_out]); lo_i = np.stack([ss.filtfilt(bl, al, r) for r in d_in])
        for sc, (o_, i_) in {"slow": (lo_o, lo_i), "fast": (d_out - lo_o, d_in - lo_i)}.items():
            for lo, hi in BANDS:
                O, I = o_[lo:hi, v].ravel(), i_[lo:hi, v].ravel()
                out[f"{nm}|{sc}|c{lo + 1}-{hi}"] = (float(O @ I), float(I @ I), float(O @ O))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="名前=版A,版B(既定 product / tabrender)")
    a = ap.parse_args()
    if a.pairs:
        PAIRS.clear()
        PAIRS.update({e.split("=")[0]: tuple(e.split("=")[1].split(",")) for e in a.pairs})
    od = Path(a.out)
    keys = sorted(f.stem for f in (od / "rr_delta").glob("*.npz"))
    with Pool(8) as pl:
        rows = pl.map(one, [(od, k, dict(PAIRS)) for k in keys])
    rep = {}
    for k in rows[0]:
        oi, ii, oo = (sum(r[k][j] for r in rows) for j in range(3))
        rep[k] = {"slope": round(oi / ii, 3), "corr": round(oi / np.sqrt(ii * oo), 3)}
    Path(a.res).write_text(json.dumps(rep, indent=1))
    for k, v in rep.items():
        print(k, v, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
