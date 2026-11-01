"""D240(学習なし・results/conv_p0/d240_prereg.yaml): 倍音どうしの相対位相(RPS: θ_k = φ_k − k·φ_1)の隣り合うフレームでの揃い方(帯域ごと)は、耳の不合格(WORLD の再合成・変換)を再現し、合格の写し合成を通すか。
f0 = harvest + stonemask(16 kHz・5 ms)・窓 = Hann 4 周期・倍音 = f0 の倍数の正確な DFT・rpc_b = cos(Δθ_k) の帯域内平均。WORLD = 本物 150 本の分析再合成。
    uv run python d240_rps_ear.py --s <dev の親> --out ../results/conv_p0/d240.json
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"
SR = 48000
SEC = 8.0
BANDS = ((0, 1000), (1000, 3000), (3000, 6000), (6000, 10000))
NAMES = ("rpc1", "rpc2", "rpc3", "rpc4")
FAIL_GROUPS = ("FAIL_cand_ab1", "FAIL_prodcand", "FAIL_convg1", "FAIL_conv_ab1", "dev:G03S10ZH", "dev:RRQSDC19LE")
DEVF = {"KNN": "d135_real", "G03S10ZH": "d114", "RRQSDC19LE": "d139"}


def load48(p: str) -> np.ndarray:
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    x, sr = sf.read(p, dtype="float64", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x[: int(SEC * SR)]


def f0of(x: np.ndarray) -> tuple:
    import pyworld
    from scipy.signal import resample_poly
    x16 = resample_poly(x, 1, 3)
    f0, t = pyworld.harvest(x16, 16000, f0_floor=70, f0_ceil=1000, frame_period=5.0)
    return pyworld.stonemask(x16, f0, t, 16000), t


def world(x: np.ndarray) -> np.ndarray:
    import pyworld
    f0, t = f0of(x)
    sp = pyworld.cheaptrick(x, f0, t, SR)
    ap = pyworld.d4c(x, f0, t, SR)
    return pyworld.synthesize(f0, sp, ap, SR, frame_period=5.0)[: len(x)]


def rps(x: np.ndarray) -> dict:
    x = x / (np.sqrt((x ** 2).mean()) + 1e-12) * 0.05
    f0, t = f0of(x)
    acc = {b: [] for b in range(len(BANDS))}
    prev = None
    for i, (f, tc) in enumerate(zip(f0, t)):
        if f <= 0:
            prev = None
            continue
        L = int(round(4 * SR / f)) | 1
        c = int(round(tc * SR))
        a, b = c - L // 2, c + L // 2 + 1
        if a < 0 or b > len(x):
            prev = None
            continue
        seg = x[a:b] * np.hanning(L)
        K = int(10000 // f)
        if K < 2:
            prev = None
            continue
        n = np.arange(L) - L // 2
        k = np.arange(1, K + 1)
        H = np.exp(-2j * np.pi * np.outer(k * f / SR, n)) @ seg
        ph = np.angle(H)
        th = np.angle(np.exp(1j * (ph - k * ph[0])))
        if prev is not None and prev[0].shape == th.shape and abs(prev[1] - f) / f < 0.1:
            d = np.cos(th - prev[0])
            fk = k * f
            for bi, (lo, hi) in enumerate(BANDS):
                m = (fk >= lo) & (fk < hi) & (k > 1)
                if m.any():
                    acc[bi].append(float(d[m].mean()))
        prev = (th, f)
    return {NAMES[bi]: (float(np.mean(v)) if len(v) > 10 else float("nan")) for bi, v in acc.items()}


def job(j: tuple) -> dict:
    grp, ident, path, form = j
    x = load48(path)
    if grp == "WORLD":
        x = world(x)
    return {"grp": grp, "id": ident, "form": form} | rps(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    S = Path(a.s)
    base = ROOT / "female-dataset"
    J = []
    for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]:
        if r["grp"] == "REAL_POOL":
            ws = sorted((base / r["id"]).glob("*.wav"))
            J.append(("REAL_POOL", r["id"], str(ws[len(ws) // 2]), ""))
            J.append(("WORLD", r["id"], str(ws[len(ws) // 2]), "world"))
        elif not r["grp"].startswith("dev:"):
            d, t, L = r["id"].split("/")
            J.append((r["grp"], r["id"], str(EB / d / t / f"{L}.wav"), r["form"]))
    for nm, dd in DEVF.items():
        for f in sorted((S / "dev" / dd).glob(f"*_{nm}.wav")):
            J.append((f"dev:{nm}", f"dev/{f.name}", str(f), nm))
    with Pool(a.workers) as pl:
        rows = pl.map(job, J, chunksize=2)
    rep: dict = {}
    vals = lambda g, m: np.array([r[m] for r in rows if r["grp"] == g], float)
    rep["real_p15_p85"] = {m: [round(float(np.nanpercentile(vals("REAL_POOL", m), q)), 4) for q in (15, 85)] for m in NAMES}
    grps = sorted({r["grp"] for r in rows})
    rep["medians"] = {g: {m: round(float(np.nanmedian(vals(g, m))), 4) for m in NAMES} | {"n": int(len(vals(g, NAMES[0])))} for g in grps}
    out = lambda m, v: not (rep["real_p15_p85"][m][0] <= v <= rep["real_p15_p85"][m][1])
    rep["axis"] = {}
    for m in NAMES:
        pass_ok = all(not out(m, rep["medians"][g][m]) for g in ("PASS_rvoc", "PASS_bigvgan"))
        w = out(m, rep["medians"]["WORLD"][m])
        det = {g: out(m, rep["medians"][g][m]) for g in FAIL_GROUPS}
        rep["axis"][m] = {"pass_ok": pass_ok, "world_out": w, "fail_detected": det, "world_ok": pass_ok and w, "axis_ok": pass_ok and w and all(det.values())}
    rep["trial1"] = {r["form"]: {m: round(r[m], 4) for m in NAMES} for r in rows if r["id"].startswith("cand_ab1/cand_01/")}
    rep["verdict"] = "PHASE_EXPLAINS" if any(v["axis_ok"] for v in rep["axis"].values()) else "WORLD_ONLY" if any(v["world_ok"] for v in rep["axis"].values()) else "NONE"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep | {"rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
