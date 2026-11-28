from __future__ import annotations

import os

import numpy as np

import nvoc as N
import pae as PA

SR = N.SR
NFFT = PA.NFFT
Q1 = -0.15
F0_FLOOR = 3.0 * SR / (NFFT - 3.0)
DEFAULT_F0 = 500.0
LOWCAP = os.environ.get("SNV_CT_LOWCAP", "") == "1"
_Q = np.arange(1, NFFT // 2 + 1) / SR


def half_k(mode: str) -> float:
    return {"hann3": 1.5, "sine2": 1.0}[mode] if mode in ("hann3", "sine2") else float(mode[-3:]) / 200.0


def frame(x: np.ndarray, pos: int, f0: float, mode: str, cap: float = 1e9) -> np.ndarray:
    k = half_k(mode)
    fw = max(f0, k / cap)
    half = int(np.floor(k * SR / fw + 0.5))
    base = np.arange(-half, half + 1)
    idx = np.clip(pos + base, 0, len(x) - 1)
    if mode == "hann3":
        w = 0.5 * np.cos(np.pi * base / 1.5 / SR * fw) + 0.5
    elif mode == "sine2":
        w = np.cos(np.pi * base * fw / (2.0 * SR)).clip(min=0.0)
    elif mode.startswith("tuk"):
        e = 2 * k - 1
        w = np.cos(0.5 * np.pi * np.clip((np.abs(base) * fw / SR - (1 - e) / 2) / e, 0.0, 1.0))
    else:
        e = 2 * k - 1
        w = np.sqrt(np.clip((k - np.abs(base) * fw / SR) / e, 0.0, 1.0))
    w = w / np.sqrt((w * w).sum())
    s = x[idx] * w
    s = s - w * (s.sum() / w.sum())
    p = np.abs(np.fft.rfft(s, NFFT)) ** 2
    df = SR / NFFT
    fa = np.arange(NFFT // 2 + 1) * df
    up = 2 + int(f0 * NFFT / SR)
    p[:up] = p[:up] + np.interp(fa[:up], (f0 - fa[:up])[::-1], p[:up][::-1])
    width = f0 * 2.0 / 3.0
    b = int(width * NFFT / SR) + 1
    mir = np.concatenate([p[1 : b + 1][::-1], p, p[-b - 1 : -1][::-1]])
    seg = np.cumsum(mir) * df
    xs = -(b - 0.5) * df + np.arange(len(seg)) * df
    p = (np.interp(fa + width / 2, xs, seg) - np.interp(fa - width / 2, xs, seg)) / width
    p = p + 2.2e-16
    c = np.fft.irfft(np.log(p), NFFT)
    lift = np.ones(NFFT // 2 + 1)
    lift[1:] = np.sin(np.pi * f0 * _Q) / (np.pi * f0 * _Q) * ((1 - 2 * Q1) + 2 * Q1 * np.cos(2 * np.pi * _Q * f0))
    c[: NFFT // 2 + 1] *= lift
    c[NFFT // 2 + 1 :] = c[1 : NFFT // 2][::-1]
    return np.exp(np.fft.rfft(c, NFFT).real)


def amp(x: np.ndarray, f0: np.ndarray, mode: str = "hann3", f_sw: float = 1e9, cap_ms: float = 1e12) -> np.ndarray:
    import snv_p0_front as P

    fb, _ = PA._np_consts()
    T = len(f0)
    xa = np.concatenate([np.zeros(P.PAD_L), x.astype(np.float64), np.zeros(PA.A)])
    out = np.empty((T, NFFT // 2 + 1))
    for t in range(T):
        f = float(f0[t])
        v = f > F0_FLOOR
        m = mode if v and f < f_sw else "hann3"
        fe = f if v else DEFAULT_F0
        k = half_k(m)
        T0 = 1.0 / (f if f > 0 else 200.0)
        cap = cap_ms * 1e-3 if m != "hann3" or (LOWCAP and 0 < f <= F0_FLOOR) else 1e9
        tau = min(t * P.H, (t + 1) * P.H - 2 - int(np.ceil(min((k if f > 0 else 1.5) * T0, cap) * SR)))
        out[t] = frame(xa, int(np.floor(P.PAD_L + tau + 0.001 * SR / SR + 0.5)), fe, m, cap)
    return np.sqrt(out) @ fb.T
