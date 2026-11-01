"""D610(推論だけ・介入): 目標の表の『声域より下の帯』に傾き σ dB/oct を足した登録(d85_tfix<σ>)を作り、低い k の基音と倍読みが追うかを測る。規則は results/conv_p0/d610/prereg.yaml。
ΔL(f) = σ·log2(f_t / max(f, F_LO))(f < f_t)・0(f ≥ f_t)・f_t = exp(mu)。DCT は正規直交 = 全ての単位の行に δ = D[1:25] · ΔL を足すのと同じ(c0 の成分は表に無いので落ちる)。
    uv run python d610_tabfix.py --build 6 3
    uv run python d610_tabfix.py --ltas OURS5Y58T6-OURS5Y58 ... --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import NPROC, H, boot
from d602_lowband import CF, ltas
from d609_tabreg import mel_logenv

F_LO = 100.0


def delta(ft: float, sigma: float, D: np.ndarray, fc: np.ndarray) -> np.ndarray:
    dl = np.where(fc < ft, sigma * np.log2(ft / np.maximum(fc, F_LO)), 0.0) * math.log(10) / 20
    return D[1:25] @ dl


def realized_ei(T: np.ndarray, w: np.ndarray, D: np.ndarray, fc: np.ndarray, f1: float) -> float:
    L = D[1:25].T @ (T @ w / w.sum())
    return float((np.interp(f1, fc, L) - np.interp(2 * f1, fc, L)) * 20 / math.log(10))


def build(sigmas: list) -> None:
    D, fc = mel_logenv()
    for sg in sigmas:
        rep = []
        for s in ("dev2", "dev3"):
            for en in sorted((H / s / "d85_work").glob("en_*")):
                j = json.loads((en / "enroll.json").read_text())
                T = np.fromfile(en / "enroll.bin", "<f4").reshape(j["D"], j["K"]).astype(np.float64)
                d0 = delta(math.exp(j["mu"]), 0.0, D, fc)
                assert np.abs(d0).max() == 0.0
                T2 = T + delta(math.exp(j["mu"]), sg, D, fc)[:, None]
                od = H / s / f"d85_tfix{sg:g}" / en.name
                od.mkdir(parents=True, exist_ok=True)
                shutil.copy(en / "enroll.json", od / "enroll.json")
                T2.astype("<f4").tofile(od / "enroll.bin")
                w = np.asarray(j["counts"], np.float64)
                rep.append((s, en.name, round(math.exp(j["mu"])), round(realized_ei(T, w, D, fc, 180.0), 2), round(realized_ei(T2, w, D, fc, 180.0), 2)))
        d = np.array([r[4] - r[3] for r in rep])
        print(f"sigma {sg:g}: tables {len(rep)} EI180 change median {np.median(d):.2f} dB [p10 {np.percentile(d, 10):.2f}, p90 {np.percentile(d, 90):.2f}]", flush=True)
        (H / "dev2" / f"d85_tfix{sg:g}" / "build.json").write_text(json.dumps(rep))


def ltas_job(it: tuple) -> tuple:
    return it[:3], ltas(it[3])


def ltas_pairs(pairs: list, out: str) -> None:
    names = sorted({v for p in pairs for v in p.split("-")})
    its = []
    for s in ("dev2", "dev3"):
        for nm in names:
            for f in sorted((H / s / "d570" / nm).glob("p*_t*.wav")):
                key, ti = re.match(r"(p\d+_t(\d+))_", f.name).groups()
                its.append((nm, s, f"{key}|{ti}", f))
    with Pool(NPROC) as pl:
        res = pl.map(ltas_job, its, chunksize=2)
    L: dict = {}
    for (nm, s, kt), d in res:
        L.setdefault(nm, {})[(s, kt)] = d
    rng = np.random.default_rng(610)
    rep: dict = {"cf": [round(float(c), 1) for c in CF], "pairs": {}}
    for p in pairs:
        A_, B_ = p.split("-")
        ks = sorted(set(L[A_]) & set(L[B_]))
        dd = np.array([L[A_][k] - L[B_][k] for k in ks])
        tg = np.array([f"{k[0]}|{k[1].split('|')[1]}" for k in ks])
        rep["pairs"][p] = {"n": len(ks), "mean_by_band": [round(float(v), 2) for v in dd.mean(0)],
                           "band100": {"diff": round(float(dd[:, 0].mean()), 2), "ci95": boot(dd[:, 0], tg, rng)},
                           "band125": {"diff": round(float(dd[:, 1].mean()), 2), "ci95": boot(dd[:, 1], tg, rng)}}
        print(p, rep["pairs"][p]["band100"], rep["pairs"][p]["band125"], rep["pairs"][p]["mean_by_band"][:8], flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=float, nargs="*")
    ap.add_argument("--ltas", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.build:
        build(a.build)
    if a.ltas:
        ltas_pairs(a.ltas, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
