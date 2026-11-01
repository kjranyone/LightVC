"""D619(推論だけ): 生成器の出力に、励起の f0 を追う因果な適応ピーキング EQ(第 1 倍音だけ持ち上げる)を掛ける。
フレーム i(240 標本)ごとに、出力の過去 2 周期(フレームの始まりまで = 先読み 0)の H1 − H2 を f0 の最小二乗(Hann)で測り、
g_raw = clamp(目標 − H1H2, 0, GMAX)・g = 0.7 g_prev + 0.3 g_raw(無声・f0 ≥ 目標の声域では g_raw = 0)。双二次の RBJ ピーキング(Q 3・中心 = f0)をフレームの係数で状態を引き継いで掛ける。
f0 = 製品の励起の高さ(e2e/eval_<組>_pre_nof0 の条件の列 25 × 2^(k/12)・前置き 12 s の後ろ)。出力 = 元の描画に掛けてピーク 0.9 に戻す。規則は results/conv_p0/d619/prereg.yaml。
    uv run python d619_h1eq.py --base OURS5RY58G --k 5 --target 6 --sfx Q6 --sets dev2 dev3
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
SR, HOP, PRE = 48000, 240, 2400
Q, GMAX = 3.0, 15.0


def peq(f0: float, g: float) -> tuple:
    A = 10 ** (g / 40)
    w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / (2 * Q)
    c = np.cos(w0)
    a0 = 1 + al / A
    return np.array([1 + al * A, -2 * c, 1 - al * A]) / a0, np.array([1.0, -2 * c / a0, (1 - al / A) / a0])


def h12(y: np.ndarray, end: int, f0: float) -> float:
    p = int(round(SR / f0))
    st = end - 2 * p
    if st < 0:
        return np.nan
    n = np.arange(2 * p)
    w = np.hanning(2 * p)
    ph = 2 * np.pi * f0 * n / SR
    B = np.stack([np.ones_like(ph), np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph)], 1)
    co, *_ = np.linalg.lstsq(B * w[:, None], y[st:end] * w, rcond=None)
    a1, a2 = np.hypot(co[1], co[2]), np.hypot(co[3], co[4])
    return float(20 * np.log10((a1 + 1e-12) / (a2 + 1e-12)))


def h1eq(y: np.ndarray, f0: np.ndarray, ft: float, target: float, fmax: float = np.inf) -> tuple:
    y = y.astype(np.float64)
    out = np.zeros_like(y)
    zi = np.zeros(2)
    g = 0.0
    gs = []
    for i in range(min(len(f0), len(y) // HOP)):
        f = f0[i]
        if f > 0 and f < min(ft, fmax):
            d = h12(y, i * HOP, f)
            graw = 0.0 if not np.isfinite(d) else float(np.clip(target - d, 0.0, GMAX))
        else:
            graw = 0.0
        g = 0.7 * g + 0.3 * graw
        fc = f if f > 0 else (f0[i - 1] if i > 0 and f0[i - 1] > 0 else 200.0)
        b, a = peq(fc, g)
        out[i * HOP:(i + 1) * HOP], zi = ss.lfilter(b, a, y[i * HOP:(i + 1) * HOP], zi=zi)
        gs.append(g)
    n = min(len(f0), len(y) // HOP) * HOP
    out[n:] = y[n:]
    return out, np.array(gs)


def job(it: tuple) -> tuple:
    import train_ppgvoc as P
    s, key, ti, k, src, dst, target, fmax = it
    y, _ = sf.read(src)
    c, _ = P.load_dump(H / "e2e" / f"eval_{s}_pre_nof0", key)
    c = c[PRE:]
    f0 = np.where(c[:, 26] > 0.5, 200.0 * np.exp(c[:, 25]) * 2 ** (k / 12), 0.0)
    ft = float(np.exp(json.loads((H / s / "d85_work" / f"en_{ti:03d}" / "enroll.json").read_text())["mu"]))
    o, gs = h1eq(y, f0, ft, target, fmax)
    o = o / (np.abs(o).max() + 1e-9) * 0.9
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, o.astype(np.float32), SR)
    v = gs[f0[: len(gs)] > 0]
    return key, float(np.mean(v)) if len(v) else 0.0, float(np.percentile(v, 90)) if len(v) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--sfx", required=True)
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fmax", type=float, default=float("inf"), help="D624: EQ を働かせる f0 の上限(Hz)")
    a = ap.parse_args()
    if a.selftest:
        t = np.arange(SR * 2) / SR
        f0 = np.full(400, 120.0)
        y = 0.1 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 240 * t) + 0.2 * np.sin(2 * np.pi * 360 * t)
        o, gs = h1eq(y, f0, 433.0, 6.0)
        print("selftest: H1-H2 in", round(h12(y, len(y), 120.0), 2), "out", round(h12(o, len(o), 120.0), 2), "final gain", round(float(gs[-1]), 2))
        o0, _ = h1eq(y, np.zeros(400), 433.0, 6.0)
        print("selftest unvoiced identity max|diff|", float(np.abs(o0 - y).max()))
        return 0
    its = []
    for s in a.sets:
        for f in sorted((H / s / "d570" / a.base).glob("p*_t*.wav")):
            key = f.name[:10]
            nm = f"{a.base}{a.sfx}"
            its.append((s, key, int(key[7:10]), a.k, f, H / s / "d570" / nm / f"{key}_{nm}.wav", a.target, a.fmax))
    with Pool(12) as pl:
        res = pl.map(job, its, chunksize=2)
    g = np.array([r[1] for r in res])
    print(a.base + a.sfx, "files", len(res), "mean voiced gain dB", round(float(g.mean()), 2), "p90 of file p90", round(float(np.percentile([r[2] for r in res], 90)), 2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
