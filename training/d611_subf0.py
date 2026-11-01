"""D611(推論だけ): 基音より下のうなりの物差し。ファイルごとに
b100 = 有声で基準の高さ r ≥ 160 Hz のフレーム(Hann 2048 の主ローブ ±47 Hz が 100 Hz 帯に入らない)の 89〜111 Hz の電力 ÷ 50〜8,000 Hz の電力(dB・フレームの中央値)・
subf0 = 150 ≤ r < 210 Hz のフレームの 0.3r〜0.6r の電力 ÷ 基音(r の bin ±1 の最大)(dB・中央値)。
r = 候補は入力の harvest × 2^(k/12)(遅れ合わせ)・本物の録音は自分の harvest(D607 と同じ)。規則は results/conv_p0/d611/prereg.yaml。
    uv run python d611_subf0.py --pairs OURS5Y58H5-OURS5Y58T0 ... --out <json>
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

from d592_refkc import NPROC, H, SR, boot, harvest
from d598_arm_cmp import prep_
from d605_onset import align

NF = 2048


def measures(x: np.ndarray, r: np.ndarray) -> tuple:
    w = np.hanning(NF)
    fq = np.fft.rfftfreq(NF, 1 / SR)
    b1, b2, bt = (fq >= 89) & (fq < 111), (fq >= 50) & (fq < 8000), None
    hb, sb = [], []
    for i in np.nonzero(r > 0)[0]:
        c = i * 240
        if c - NF // 2 < 0 or c + NF // 2 > len(x):
            continue
        P = np.abs(np.fft.rfft(x[c - NF // 2:c + NF // 2] * w)) ** 2 + 1e-20
        if r[i] >= 160:
            hb.append(10 * np.log10(P[b1].sum() / P[b2].sum()))
        if 150 <= r[i] < 210:
            k0 = int(round(r[i] * NF / SR))
            sub = (fq >= 0.3 * r[i]) & (fq < 0.6 * r[i])
            sb.append(10 * np.log10(P[sub].sum() / P[k0 - 1:k0 + 2].max()))
    return (float(np.median(hb)) if len(hb) >= 30 else np.nan, float(np.median(sb)) if len(sb) >= 30 else np.nan)


def job(it: tuple) -> tuple:
    from d6_unit_cov import load48
    nm, s, key, ti, k, path = it
    x = prep_(load48(path)).astype(np.float64)
    if nm.startswith("OURS") and s != "devff2":
        hs = harvest(prep_(load48(H / s / "d56" / f"{key}_SRC.wav")))
        r = align(hs * 2 ** (k / 12), harvest(x))
    else:
        r = harvest(x)
    n = min(len(r), len(x) // 240)
    return it, measures(x, r[:n])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--real", action="store_true", help="同じ高さの本物の女声(CEN・FF_SRC)の subf0 も")
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = sorted({v for p in a.pairs for v in p.split("-")})
    its = []
    for s in a.sets:
        tset = set()
        for nm in names:
            k = int(re.match(r"OURS(\d+)", nm).group(1))
            for f in sorted((H / s / "d570" / nm).glob("p*_t*.wav")):
                key, ti = re.match(r"(p\d+_t(\d+))_", f.name).groups()
                its.append((nm, s, key, int(ti), k, f))
                c = H / s / "cen_reg" / f"cen_{int(ti):03d}.wav"
                if a.real and int(ti) not in tset and c.exists():
                    tset.add(int(ti))
                    its.append(("CEN", s, f"t{int(ti):03d}", int(ti), 0, c))
    if a.real:
        for f in sorted((H / "devff2" / "d56").glob("p*_t*_SRC.wav")):
            its.append(("FF_SRC", "devff2", f.stem, 1000 + len(its), 0, f))
    with Pool(NPROC) as pl:
        res = pl.map(job, its, chunksize=2)
    M: dict = {}
    for (nm, s, key, ti, k, _), v in res:
        M.setdefault(nm, {})[(s, key, ti)] = v
    rep: dict = {"versions": {}, "pairs": {}}
    for nm, d in M.items():
        v = np.array(list(d.values()))
        rep["versions"][nm] = {"n": len(v), "b100_mean": round(float(np.nanmean(v[:, 0])), 2), "subf0_mean": round(float(np.nanmean(v[:, 1])), 2), "subf0_n": int(np.isfinite(v[:, 1]).sum())}
        print(nm, rep["versions"][nm], flush=True)
    rng = np.random.default_rng(611)
    for p in a.pairs:
        A_, B_ = p.split("-")
        ks = sorted(set(M[A_]) & set(M[B_]))
        out = {}
        for j, nmj in enumerate(("b100", "subf0")):
            d = np.array([M[A_][k][j] - M[B_][k][j] for k in ks])
            tg = np.array([f"{k[0]}|{k[2]}" for k in ks])
            ok = np.isfinite(d)
            out[nmj] = {"n": int(ok.sum()), "diff": round(float(d[ok].mean()), 3), "ci95": boot(d[ok], tg[ok], rng)}
        rep["pairs"][p] = out
        print(p, out, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
