"""D609(推論だけ・相関): 低い k の基音の弱さ(150〜210 Hz の H1 − H2)と、目標の声域(登録の mu)・表が示す H1 − H2(EI180)の結びつき。規則は results/conv_p0/d609/prereg.yaml。
    uv run python d609_tabreg.py --out ../results/conv_p0/d609/tabreg.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import NPROC, H
from d607_h1h2 import job


def mel_logenv() -> tuple:
    import librosa
    import zsvc as Z
    D = Z.dct_mat(128).numpy().astype(np.float64)
    fc = librosa.mel_frequencies(130, fmin=0.0, fmax=24000.0)[1:-1]
    return D, fc


def table_ei(en: Path, D: np.ndarray, fc: np.ndarray, f1: float = 180.0) -> float:
    j = json.loads((en / "enroll.json").read_text())
    T = np.fromfile(en / "enroll.bin", "<f4").reshape(j["D"], j["K"]).astype(np.float64)
    w = np.asarray(j["counts"], np.float64)
    L = D[1:25].T @ (T @ w / w.sum())
    return float((np.interp(f1, fc, L) - np.interp(2 * f1, fc, L)) * 20 / math.log(10))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    its = []
    for s in ("dev2", "dev3"):
        for f in sorted((H / s / "d570" / "OURS5RY58G").glob("p*_t*.wav")):
            key, ti = re.match(r"(p\d+_t(\d+))_", f.name).groups()
            for nm in ("OURS5RY58G", "OURS5Y59"):
                its.append((nm, s, key, int(ti), 5, H / s / "d570" / nm / f"{key}_{nm}.wav"))
            its.append(("SRC", s, key, int(ti), 0, H / s / "d56" / f"{key}_SRC.wav"))
    with Pool(NPROC) as pl:
        res = pl.map(job, its, chunksize=2)
    M: dict = {}
    for (nm, s, key, ti, k, _), x, r, d12 in res:
        n = min(len(r), len(d12))
        ok = (r[:n] > 0) & np.isfinite(d12[:n])
        rr, dd = r[:n][ok], d12[:n][ok]
        m = (rr >= 150) & (rr < 210)
        M.setdefault(nm, {})[(s, key, ti)] = (float(np.median(dd[m])) if m.sum() >= 30 else np.nan, float(np.median(dd)) if len(dd) else np.nan)
    D, fc = mel_logenv()
    tg = sorted({(s, ti) for (s, _, ti) in M["OURS5RY58G"]})
    reg, ei = {}, {}
    for s, ti in tg:
        en = H / s / "d85_work" / f"en_{ti:03d}"
        reg[(s, ti)] = math.exp(json.loads((en / "enroll.json").read_text())["mu"])
        ei[(s, ti)] = table_ei(en, D, fc)

    def per_target(nm: str, col: int = 0) -> np.ndarray:
        out = []
        for t in tg:
            v = [M[nm][k][col] for k in M[nm] if (k[0], k[2]) == t]
            v = [z for z in v if np.isfinite(z)]
            out.append(float(np.mean(v)) if v else np.nan)
        return np.array(out)
    from scipy.stats import spearmanr
    rng = np.random.default_rng(609)
    R_ = np.array([reg[t] for t in tg])
    E_ = np.array([ei[t] for t in tg])

    def rho(x: np.ndarray, y: np.ndarray) -> tuple:
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        bs = []
        for _ in range(2000):
            ii = rng.integers(0, len(x), len(x))
            bs.append(spearmanr(x[ii], y[ii]).correlation)
        return int(ok.sum()), round(float(spearmanr(x, y).correlation), 3), [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]
    rep: dict = {"n_targets": len(tg), "reg_hz_quantiles": [round(float(q), 1) for q in np.percentile(R_, [0, 25, 50, 75, 100])],
                 "ei180_quantiles": [round(float(q), 2) for q in np.percentile(E_, [0, 25, 50, 75, 100])], "rules": {}, "record": {}}
    for nm in ("OURS5RY58G", "OURS5Y59"):
        y = per_target(nm)
        sec = rep["rules"] if nm == "OURS5RY58G" else rep["record"].setdefault(nm, {})
        sec["V1"] = dict(zip(("n", "rho", "ci95"), rho(R_, y)))
        sec["V2"] = dict(zip(("n", "rho", "ci95"), rho(E_, y)))
        print(nm, "V1", sec["V1"], "V2", sec["V2"], flush=True)
    rep["rules"]["V3"] = dict(zip(("n", "rho", "ci95"), rho(R_, E_)))
    print("V3", rep["rules"]["V3"], flush=True)
    y = per_target("OURS5RY58G")
    sx = per_target("SRC", 1)
    ok = np.isfinite(y) & np.isfinite(sx)
    from scipy.stats import rankdata
    ry, rs, rr_, re_ = (rankdata(v[ok]) for v in (y, sx, R_, E_))
    res_y = ry - np.polyval(np.polyfit(rs, ry, 1), rs)
    res_r = rr_ - np.polyval(np.polyfit(rs, rr_, 1), rs)
    res_e = re_ - np.polyval(np.polyfit(rs, re_, 1), rs)
    rep["record"]["partial_given_src"] = {"reg": round(float(np.corrcoef(res_r, res_y)[0, 1]), 3), "ei180": round(float(np.corrcoef(res_e, res_y)[0, 1]), 3)}
    rep["record"]["by_reg_tercile"] = {}
    qs = np.percentile(R_, [0, 33.3, 66.7, 100])
    qs[-1] += 1
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (R_ >= lo) & (R_ < hi)
        rep["record"]["by_reg_tercile"][f"{lo:.0f}-{hi:.0f}"] = {"n": int(m.sum()), "m_file": round(float(np.nanmean(y[m])), 2), "ei180": round(float(np.mean(E_[m])), 2)}
    print(rep["record"], flush=True)
    rep["per_target"] = [{"t": f"{t[0]}|{t[1]}", "reg": round(reg[t], 1), "ei180": round(ei[t], 2), "m_file": None if not np.isfinite(v) else round(float(v), 2)} for t, v in zip(tg, y)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
