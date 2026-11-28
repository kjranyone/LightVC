import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parent.parent.parent
IN = ROOT / "results/snv_p96/probe_in"
PR = ROOT / "results/snv_p97/probe/product"
SPS = ["ca91ecce", "cb87e9f4", "cc5576e7", "cd094b9e"]
SR, H = 48000, 240
STEP = 48


def follower(x: np.ndarray, rel_ms: float) -> np.ndarray:
    a = np.abs(x)
    g = np.exp(-1.0 / (rel_ms * 1e-3 * SR))
    o = np.empty_like(a)
    s = 0.0
    for i, v in enumerate(a):
        s = v if v > s else s * g
        o[i] = s
    return o


def onset_strength(l: np.ndarray) -> np.ndarray:
    return np.maximum(np.diff(l, prepend=l[0]), 0.0)


def lag_ms(ref: np.ndarray, y: np.ndarray, span: int = 40) -> tuple[float, float]:
    a, b = onset_strength(ref), onset_strength(y)
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    c = np.array([float((a[max(0, -k):n - max(0, k)] * b[max(0, k):n - max(0, -k)]).sum()) for k in range(-span, span + 1)])
    k = np.arange(-span, span + 1)
    w = np.maximum(c - 0.5 * c.max(), 0.0)
    return float((w * k).sum() / w.sum()), float(k[c.argmax()])


def hold(fr: np.ndarray, n: int) -> np.ndarray:
    t = np.arange(n) * STEP
    return fr[np.clip((t - H + 1) // H + 0, 0, len(fr) - 1)] * (t >= H) + fr[0] * (t < H)


def main() -> None:
    rows = []
    for sp in SPS:
        x = np.fromfile(IN / sp / "in.f32", np.float32).astype(np.float64)
        ys = np.fromfile(PR / f"{sp}_conv.csv.ysrc", np.float32).reshape(-1, 29)
        n = len(x) // STEP
        p = np.convolve(x * x, np.ones(240) / 240, mode="same")
        ref = 0.5 * np.log(p[::STEP][:n] + 1e-9)
        c0 = ys[:, 0].astype(np.float64) * 10 / 2
        sm = np.copy(c0)
        sm[2:] = 0.25 * c0[:-2] + 0.5 * c0[1:-1] + 0.25 * c0[2:]
        r = {"sp": sp, "c0": lag_ms(ref, hold(c0, n)), "c0_3pt": lag_ms(ref, hold(sm, n))}
        for rel in [5.0, 10.0, 20.0]:
            f = np.log(follower(x, rel) + 1e-4)
            fr = f[H - 1::H]
            r[f"fol{int(rel)}_frame"] = lag_ms(ref, hold(fr, n))
            r[f"fol{int(rel)}_raw"] = lag_ms(ref, f[::STEP][:n])
            d = np.diff(fr)
            st = np.abs(d)
            r[f"fol{int(rel)}_jit"] = float(np.median(st))
        r["c0_jit"] = float(np.median(np.abs(np.diff(c0))))
        rows.append(r)
        print(r, flush=True)
    for k in rows[0]:
        if k != "sp":
            v = [q[k][0] if isinstance(q[k], tuple) else q[k] for q in rows]
            print(k, round(float(np.median(v)), 2))


if __name__ == "__main__":
    main()
