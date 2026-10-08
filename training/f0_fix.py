"""因果な f0 の後処理(先読み 0): オクターブ誤りを直近の有声フレームの中央値へ折り返し、直近 3 有声フレームの中央値でならす。

  r = 直近 N 有声フレームの log f0 の中央値。d = log2(f/r) が ±1 ± tol なら f を 2 倍/半分に折り返す。
  出力 = 折り返し後の直近 3 有声フレームの中央値(無声は 0 のまま)。フレーム t の値はフレーム ≤ t だけで決まる。
"""
from __future__ import annotations

import numpy as np


def fix_f0(f0: np.ndarray, n_ref: int = 15, tol: float = 0.3, n_med: int = 3) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros_like(f0, dtype=np.float32)
    folded = np.zeros_like(f0, dtype=np.float32)
    hist: list[float] = []
    recent: list[float] = []
    for t, f in enumerate(f0):
        if f <= 0:
            recent = []
            continue
        g = float(f)
        if len(hist) >= 5:
            r = float(np.median(hist[-n_ref:]))
            d = np.log2(g / r)
            if abs(d - 1) < tol:
                g /= 2
            elif abs(d + 1) < tol:
                g *= 2
        hist.append(g)
        folded[t] = g
        recent.append(g)
        out[t] = float(np.median(recent[-n_med:]))
    return out, folded


def logf0_stats(f0: np.ndarray) -> tuple[float, float]:
    v = f0[f0 > 0]
    if len(v) < 20:
        return 0.0, 0.0
    lv = np.log(v)
    return float(np.median(lv)), float(max(np.std(lv), 0.05))


def _selftest() -> None:
    f = np.full(200, 120.0, np.float32)
    f[50:53] = 240.0
    f[100] = 60.0
    f[150:160] = 0
    g, _ = fix_f0(f)
    assert np.allclose(g[g > 0], 120.0), g[45:60]
    t = np.arange(200)
    glide = (120 * 2 ** (t / 200)).astype(np.float32)
    g2, _ = fix_f0(glide)
    assert np.max(np.abs(np.log2(g2[5:] / glide[5:]))) < 0.02
    print("f0_fix selftest OK")


if __name__ == "__main__":
    _selftest()
