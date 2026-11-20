"""段 D5(paper_snv rev41 §5 M1 rev41): WORLD の分析合成で声の高さと声道の長さを動かす(生成器の参照の加工と、物差しの較正に使う)。
shift(x, sr, f0_ratio, alpha): 高さを f0_ratio 倍・周波数の軸を 1/alpha に縮める(フォルマントが 1/alpha = 声道が alpha 倍)。
shift_rough(..., ap_add, tilt, bw)(段 P52・§5 M39): 同じ分析合成の前に、4 kHz より下の非周期性を ap + ap_add·(1 − ap) に・包絡に 1 kHz 基準の傾き tilt dB/oct・bw で包絡のケプストラムを 40 次で切る(帯域幅を広げる)。slow(x, r): 再標本化だけで速度を r 倍(高さとフォルマントが同じ比 r で動く)。
    cd training && uv run --no-sync python snv_d5_world.py selftest
"""
from __future__ import annotations

import sys

import numpy as np


def analyze(x: np.ndarray, sr: int) -> tuple:
    import pyworld
    x = x.astype(np.float64)
    f0, t = pyworld.harvest(x, sr, f0_floor=50.0, f0_ceil=700.0, frame_period=5.0)
    return f0, pyworld.cheaptrick(x, f0, t, sr), pyworld.d4c(x, f0, t, sr)


def warp(sp: np.ndarray, alpha: float) -> np.ndarray:
    n = sp.shape[1]
    src = np.minimum(np.arange(n) * alpha, n - 1.0)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = src - i0
    return np.ascontiguousarray(sp[:, i0] * (1 - w) + sp[:, i1] * w)


def shift(x: np.ndarray, sr: int, f0_ratio: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    import pyworld
    f0, sp, ap = analyze(x, sr)
    if alpha != 1.0:
        sp, ap = warp(sp, alpha), np.clip(warp(ap, alpha), 0.0, 1.0)
    y = pyworld.synthesize(f0 * f0_ratio, sp, ap, sr, 5.0)
    return y[: len(x)].astype(np.float32)


def shift_rough(x: np.ndarray, sr: int, f0_ratio: float = 1.0, alpha: float = 1.0, ap_add: float = 0.0, tilt: float = 0.0, bw: bool = False) -> np.ndarray:
    import pyworld
    f0, sp, ap = analyze(x, sr)
    if alpha != 1.0:
        sp, ap = warp(sp, alpha), np.clip(warp(ap, alpha), 0.0, 1.0)
    hz = np.arange(sp.shape[1]) * sr / (2 * (sp.shape[1] - 1))
    if tilt != 0.0:
        sp = np.ascontiguousarray(sp * 10 ** (tilt * np.log2(np.maximum(hz, 50.0) / 1000.0) / 10)[None])
    if bw:
        c = np.fft.irfft(np.log(np.maximum(sp, 1e-20)), axis=1)
        c[:, 41:-40] = 0.0
        sp = np.ascontiguousarray(np.exp(np.fft.rfft(c, axis=1).real))
    if ap_add > 0.0:
        ap = np.clip(ap + ap_add * (1 - ap) * np.clip((4000.0 - hz) / 1000.0, 0.0, 1.0)[None], 0.0, 1.0)
    y = pyworld.synthesize(np.ascontiguousarray(f0 * f0_ratio), sp, ap, sr, 5.0)
    return y[: len(x)].astype(np.float32)


def slow(x: np.ndarray, r: float) -> np.ndarray:
    from scipy.signal import resample_poly
    q = int(round(r * 100))
    return resample_poly(x.astype(np.float64), 100, q).astype(np.float32)


def f0_median(x: np.ndarray, sr: int) -> float:
    import pyworld
    f0, _ = pyworld.harvest(x.astype(np.float64), sr, f0_floor=50.0, f0_ceil=700.0, frame_period=5.0)
    return float(np.median(f0[f0 > 0])) if (f0 > 0).any() else float("nan")


def selftest() -> None:
    sr = 48000
    t = np.arange(sr) / sr
    x = sum(np.sin(2 * np.pi * 150 * h * t) / h for h in range(1, 30)).astype(np.float32) * 0.1
    m0 = f0_median(x, sr)
    assert abs(m0 - 150) < 3, m0
    assert abs(f0_median(shift(x, sr, 0.6), sr) - 90) < 3
    assert abs(f0_median(slow(x, 0.8), sr) - 120) < 3
    sp = np.zeros((1, 1025))
    sp[0, 400] = 1.0
    assert abs(int(np.argmax(warp(sp, 1.25))) - 320) <= 1
    print("selftest ok")


if __name__ == "__main__":
    if "selftest" in sys.argv[1:]:
        selftest()
