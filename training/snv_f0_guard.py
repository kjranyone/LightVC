"""高さの番(paper_snv 段 P95 step c3・crates/lightvc-core/src/f0_guard.rs と同じ計算)。網が見る値は変えず、描画器に渡す高さと有声だけを因果に整える。
前段の短い高い読み(母音の倍音・子音の所)を、母音の途中(直前の高さから 0.7 オクターブ超の上への飛びで、登録の中央値の 1.6 倍より高い)で、直前のフレームが有声なら直前の高さに保ち、無声の後(GAP フレーム以内は同じ判定・それより後は中央値の 2.2 倍より高い)なら無声にし、HOLD フレームを超えて続いたら受け入れる。
med = 操作者の登録の有声のフレームの f0 の中央値。clean(f0, med) = 整えた f0 の列。
"""
from __future__ import annotations

import numpy as np

GAP = 10
HOLD = 12
J_CTX = np.float32(0.7)
J_ABS = np.float32(2.2)
J_HI = np.float32(1.6)
STABLE = np.float32(0.1)


def median_f0(f0s: list[np.ndarray]) -> float:
    v = np.concatenate([f[f > 0] for f in f0s]) if f0s else np.zeros(0)
    return float(np.sort(v.astype(np.float32))[(len(v) - 1) // 2]) if len(v) else 0.0


def clean(f0: np.ndarray, med: float) -> np.ndarray:
    f0 = f0.astype(np.float32)
    med = np.float32(med)
    out = np.zeros_like(f0)
    fg, gap, hi, prev = np.float32(0), GAP + 1, 0, np.float32(0)
    for t, f in enumerate(f0):
        if f <= 0:
            gap, hi = min(gap + 1, GAP + 1), 0
        else:
            ctx = gap <= GAP and fg > 0
            off = (np.log2(f / fg) > J_CTX and f > J_HI * med) if ctx else (med > 0 and f > J_ABS * med)
            if off:
                hi += 1
                if hi > HOLD and prev > 0 and abs(np.log2(f / prev)) < STABLE:
                    out[t], fg, gap = f, f, 0
                elif gap == 0:
                    out[t] = fg
                else:
                    gap = min(gap + 1, GAP + 1)
            else:
                out[t], fg, gap, hi = f, f, 0, 0
        prev = f
    return out


def unhold(f0: np.ndarray, ap: np.ndarray, n: int = 2) -> np.ndarray:
    out, run = f0.astype(np.float32).copy(), 0
    for t in range(len(out)):
        run = run + 1 if (out[t] > 0 and ap[t] >= 0.999) else 0
        if run > n:
            out[t] = 0.0
    return out
