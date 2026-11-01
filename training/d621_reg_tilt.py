"""D621(D613 と同じデータ): hf・tilt・dtilt・dhf(d109 と同じ定義)が本人の実録音で声域から下がると動くか。D613: 関門 G の ltas_d(d109 と同じ定義: 8 s・RMS 0.05・最大 − 40 dB の区間・1/3 oct 100 Hz〜10 kHz の水準をそろえた LTAS と
目標の登録の参照 <組>/d37/_work/ref_TTT.wav の先頭 30 s との RMS 差)は、目標本人の実録音でも高さが声域から下がるほど大きくなるか。規則は results/conv_p0/d613/prereg.yaml。
クリップ = female-dataset/<目標の話者>/ の参照(先頭 20 s 分)以外・3 s 以上。高さ = harvest 16 kHz の有声の中央値・Δ = 12 log2(高さ / f0_ref)。
    uv run python d621_reg_tilt.py --out ../results/conv_p0/d621/reg_tilt.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.signal as ss

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import NPROC, H

ROOT = Path(__file__).resolve().parent.parent
SR = 48000
CF = 1000.0 * 2.0 ** (np.arange(-10, 11) / 3.0)
CF = CF[(CF >= 100) & (CF <= 10000)]


def norm(y: np.ndarray) -> np.ndarray:
    return (y / (np.sqrt((y ** 2).mean()) + 1e-9) * 0.05).astype(np.float32)


def active(x: np.ndarray) -> np.ndarray:
    n = len(x) // 240
    e = 10 * np.log10((x[: n * 240].reshape(n, 240) ** 2).mean(1) + 1e-12)
    return e > e.max() - 40


def ltas(x: np.ndarray) -> np.ndarray:
    act = active(x)
    fq, _, Z = ss.stft(x, SR, nperseg=2048, noverlap=2048 - 240)
    P = np.abs(Z[:, : len(act)]) ** 2
    P = P[:, act[: P.shape[1]]].mean(1)
    d = 10 * np.log10(np.array([P[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() for c in CF]) + 1e-20)
    return d - d.mean()


def hf_tilt(x: np.ndarray) -> tuple:
    act = active(x)
    fq, _, Z = ss.stft(x, SR, nperseg=2048, noverlap=2048 - 240)
    P = np.abs(Z[:, : len(act)]) ** 2
    Pa = P[:, act[: P.shape[1]]]
    hf = float(10 * np.log10(Pa[(fq >= 4000) & (fq < 10000)].sum() / Pa[(fq >= 100) & (fq < 4000)].sum()))
    L = ltas(x)
    sel = (CF >= 300) & (CF <= 8000)
    return hf, float(np.polyfit(np.log2(CF[sel]), L[sel], 1)[0])


def job(it: tuple) -> tuple:
    import pyworld as pw
    import soundfile as sf
    from d6_unit_cov import load48
    s, ti, f0ref, ref, w = it
    if sf.info(str(w)).duration < 3.0:
        return it, None
    x = load48(w).astype(np.float64)
    f0, _ = pw.harvest(ss.resample_poly(x, 1, 3), 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
    if (f0 > 0).sum() < 50:
        return it, None
    d = 12 * np.log2(float(np.median(f0[f0 > 0])) / f0ref)
    h, t = hf_tilt(norm(x[: 8 * SR]))
    hr, tr = hf_tilt(norm(load48(ref)[: 30 * SR]))
    return it, (float(d), h, t, h - hr, t - tr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import soundfile as sf
    its = []
    for s in ("dev2", "dev3"):
        S = H / s
        for m in json.loads((S / "cen_reg" / "meta.json").read_text()):
            ws = sorted((ROOT / "female-dataset" / m["spk"]).glob("*.wav"))
            tot, k = 0.0, 0
            for w in ws:
                if tot >= 20.0:
                    break
                tot += sf.info(str(w)).duration
                k += 1
            for w in ws[k:]:
                its.append((s, m["ti"], m["f0_ref"], S / "d37/_work" / f"ref_{m['ti']:03d}.wav", w))
    with Pool(NPROC) as pl:
        res = pl.map(job, its, chunksize=4)
    rows = [(f"{it[0]}|{it[1]}",) + tuple(v) for it, v in res if v is not None]
    tg = np.array([r[0] for r in rows])
    d = np.array([r[1] for r in rows])
    rng = np.random.default_rng(621)
    ut = np.unique(tg)
    idx = {t: np.flatnonzero(tg == t) for t in ut}
    real = json.loads((ROOT / "results/conv_p0/d620_d109_dev2.json").read_text())["d570/OURS5RY58G:OURS5RY58G"]["stats"]
    rep = {"n_clips": len(rows), "n_targets": len(ut), "stats": {}}
    for j, nm in enumerate(("hf", "tilt", "dhf", "dtilt")):
        y = np.array([r[2 + j] for r in rows])

        def slope(ii: np.ndarray) -> float:
            dd, yy, tt = d[ii], y[ii], tg[ii]
            m = dd < 0
            dd, yy, tt = dd[m], yy[m], tt[m]
            xc = dd - np.array([dd[tt == t].mean() for t in tt])
            yc = yy - np.array([yy[tt == t].mean() for t in tt])
            return float((xc * yc).sum() / max((xc ** 2).sum(), 1e-9))
        bs = [slope(np.concatenate([idx[t] for t in rng.choice(ut, len(ut))])) for _ in range(2000)]
        lo, hi = real[nm]["real_p15_p85"]
        bins = {}
        for a_, b_ in ((-2, 2), (-6, -2), (-10, -6), (-14, -10), (-99, -14)):
            m = (d >= a_) & (d < b_) if a_ > -99 else (d < b_)
            bins[f"{a_}..{b_}"] = {"n": int(m.sum()), "median": round(float(np.median(y[m])), 3), "frac_below_p15": round(float((y[m] < lo).mean()), 3)}
        rep["stats"][nm] = {"slope_per_st_below": round(slope(np.arange(len(d))), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "real_p15_p85": [lo, hi], "bins": bins}
        print(nm, json.dumps(rep["stats"][nm], ensure_ascii=False), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
