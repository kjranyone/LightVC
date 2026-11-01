"""D602(推論だけ): 候補の 200〜400 Hz の過剰(D592 の帯の平均 候補 − REFKC +4〜5.5 dB)の出所 = 操作者(男声の入力)の低い帯か・目標本人の包絡か。
d109 と同じ 1/3 オクターブ 19 帯の長時間スペクトル(8 s・RMS 0.05・有声区間 = 最大 − 40 dB・水準をそろえる)をファイルごとに。
    uv run python d602_lowband.py --out ../results/conv_p0/d602/lowband.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.signal as ss

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import NPROC, H, SR, boot

CF = 1000.0 * 2.0 ** (np.arange(-10, 11) / 3.0)
CF = CF[(CF >= 100) & (CF <= 10000)]
LOW = [i for i, c in enumerate(CF) if 190 <= c <= 400]
VERS = ["OURS5RY58G", "OURS12RY58G", "OURS5RY60G", "REFKC5", "REFKC12"]


def ltas(path: Path, sec: float = 8.0) -> np.ndarray:
    from d6_unit_cov import load48
    return ltas_x(load48(path)[: int(sec * SR)])


def ltas_x(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
    n = len(x) // 240
    e = 10 * np.log10((x[: n * 240].reshape(n, 240) ** 2).mean(1) + 1e-12)
    act = e > e.max() - 40
    fq, _, Z = ss.stft(x, SR, nperseg=2048, noverlap=2048 - 240)
    P = np.abs(Z[:, : len(act)]) ** 2
    P = P[:, act[: P.shape[1]]].mean(1)
    d = 10 * np.log10(np.array([P[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() for c in CF]) + 1e-20)
    return d - d.mean()


def job(it: tuple) -> tuple:
    return it[:4], ltas(it[4])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    its = []
    for s in ("dev2", "dev3"):
        S = H / s
        tset = set()
        for f in sorted((S / "d570" / "OURS5RY58G").glob("p*_t*.wav")):
            m = re.match(r"(p\d+_t(\d+))_", f.name)
            key, ti = m.group(1), int(m.group(2))
            if not (S / "d570" / "REFKC5" / f"{key}_REFKC5.wav").exists():
                continue
            its.append(("SRC", s, key, ti, S / "d56" / f"{key}_SRC.wav"))
            for v in VERS:
                its.append((v, s, key, ti, S / "d570" / v / f"{key}_{v}.wav"))
            if ti not in tset:
                tset.add(ti)
                its.append(("CEN", s, f"t{ti:03d}", ti, S / "cen_reg" / f"cen_{ti:03d}.wav"))
    with Pool(NPROC) as pl:
        res = pl.map(job, its)
    L: dict = {}
    for (g, s, key, ti), d in res:
        L.setdefault(g, {})[(s, key)] = (ti, d)
    rng = np.random.default_rng(602)
    rep: dict = {"bands_hz": [round(float(c), 1) for c in CF], "low_bands_hz": [round(float(CF[i]), 1) for i in LOW], "mean": {}, "pairs": {}, "corr": {}}
    for g, R in L.items():
        rep["mean"][g] = [round(float(v), 2) for v in np.mean([d for _, d in R.values()], 0)]
    keys = sorted(L["OURS5RY58G"])
    tg = np.array([f"{k[0]}|{L['OURS5RY58G'][k][0]}" for k in keys])
    for A_, B_ in (("OURS5RY58G", "REFKC5"), ("OURS12RY58G", "REFKC12"), ("OURS5RY60G", "OURS5RY58G")):
        D = np.array([L[A_][k][1] - L[B_][k][1] for k in keys])
        rep["pairs"][f"{A_}-{B_}"] = {"per_band": [[round(float(D[:, i].mean()), 2)] + boot(D[:, i], tg, rng) for i in range(len(CF))],
                                      "low_mean": [round(float(D[:, LOW].mean()), 2)] + boot(D[:, LOW].mean(1), tg, rng)}
        print(f"{A_}-{B_} low {rep['pairs'][f'{A_}-{B_}']['low_mean']}", flush=True)
    E = np.array([(L["OURS5RY58G"][k][1] - L["REFKC5"][k][1])[LOW].mean() for k in keys])
    Sx = np.array([L["SRC"][k][1][LOW].mean() for k in keys])
    Cx = np.array([L["CEN"][(k[0], f"t{L['OURS5RY58G'][k][0]:03d}")][1][LOW].mean() for k in keys])
    Rx = np.array([L["REFKC5"][k][1][LOW].mean() for k in keys])
    from scipy.stats import spearmanr
    us = sorted(set(tg))
    idx = {u: np.flatnonzero(tg == u) for u in us}
    for nm, X in (("E_vs_SRC_low", Sx), ("E_vs_CEN_low", Cx), ("E_vs_REFKC5_low", Rx)):
        bs = []
        for _ in range(2000):
            ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
            bs.append(float(spearmanr(E[ii], X[ii]).correlation))
        rep["corr"][nm] = {"n": len(E), "rho": round(float(spearmanr(E, X).correlation), 3), "ci95": [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]}
        print(nm, rep["corr"][nm], flush=True)
    within = []
    for u in us:
        ii = idx[u]
        if len(ii) >= 2:
            for p in range(len(ii)):
                for q in range(p + 1, len(ii)):
                    within.append((E[ii[p]] - E[ii[q]], Sx[ii[p]] - Sx[ii[q]]))
    if len(within) >= 10:
        w = np.array(within)
        rep["corr"]["within_target_dE_vs_dSRC"] = {"n_pairs": len(w), "rho": round(float(spearmanr(w[:, 0], w[:, 1]).correlation), 3)}
        print("within", rep["corr"]["within_target_dE_vs_dSRC"], flush=True)
    for g in ("SRC", "CEN", "REFKC5", "OURS5RY58G"):
        print(g, " ".join(f"{v:6.1f}" for v in rep["mean"][g]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
