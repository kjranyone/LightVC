"""出力の番(paper_snv 段 P95 step c・crates/lightvc-core/src/out_guard.rs と同じ計算)。網が見る値は変えず、出力の波形だけに掛ける。
大きさの番 = 入力の動きが裏付けない大きさの出力を下げる。尖頭値の番 = ブロックの尖頭値が CEIL を超えたら利得を下げ、残りを tanh で丸める。
stream(x, y, p_in, p_out) = Rust の流しと同じ並び(y は x より LAT サンプル遅れている)。apply(x, w, p_in, p_out) = 遅れを除いた出力 w(入力と同じ時刻)に掛ける。
"""
from __future__ import annotations

import numpy as np

H = 240
LAT = 480
REF_N = 9
ATTACK = np.float32(0.3)
REL_ACTIVE = np.float32(2.0 * 0.005)
REL_IDLE = np.float32(0.2 * 0.005)
ACTIVE_DB = np.float32(30.0)
MARGIN = np.float32(10.0)
G_MIN = np.float32(-40.0)
UP_DB = np.float32(3.0)
RAMP = 48
CEIL = np.float32(0.9)
LIM_REL = np.float32(0.3)
P_NOMINAL = -26.0


def level_db(x: np.ndarray) -> np.float32:
    ms = float((x.astype(np.float64) ** 2).sum() / max(len(x), 1))
    return np.float32(10.0 * np.log10(max(ms, 1e-14)))


def speech_level(utts: list[np.ndarray]) -> float:
    l = sorted(float(level_db(u[i * H:(i + 1) * H])) for u in utts for i in range(len(u) // H))
    return l[int(round((len(l) - 1) * 0.9))] if l else P_NOMINAL


def soft_clip(y: np.ndarray) -> np.ndarray:
    a = np.abs(y)
    return np.where(a <= CEIL, y, np.sign(y) * (CEIL + (1 - CEIL) * np.tanh((a - CEIL) / (1 - CEIL)))).astype(np.float32)


def stream(x: np.ndarray, y: np.ndarray, p_in: float, p_out: float) -> tuple[np.ndarray, np.ndarray]:
    f = np.float32
    t = min(len(x), len(y)) // H
    out = y[: t * H].astype(np.float32).copy()
    hist = np.full(REF_N, -140.0, np.float32)
    p_in, p_out, g0, gl0, gt0 = f(p_in), f(p_out), f(0), f(0), f(0)
    tr = np.zeros((t, 5), np.float32)
    ramp = np.arange(1, H + 1, dtype=np.float32)
    for i in range(t):
        yb = out[i * H:(i + 1) * H]
        hist[i % REF_N] = level_db(x[i * H:(i + 1) * H])
        r = hist.max()
        lo = level_db(yb)
        rel = REL_ACTIVE if r > p_in - ACTIVE_DB else REL_IDLE
        p_in = p_in + ATTACK * (r - p_in) if r > p_in else p_in - rel
        allow = p_out + (r - p_in) + MARGIN
        want = f(min(max(allow - lo, G_MIN), 0.0))
        g = want if want < g0 else f(min(want, g0 + UP_DB))
        lg = lo + g
        p_out = p_out + ATTACK * (lg - p_out) if lg > p_out else p_out - rel
        pk = f(np.abs(yb).max()) * f(10.0) ** (g / f(20.0))
        need = f(20.0) * np.log10(CEIL / pk) if pk > CEIL else f(0)
        gl = f(min(need, min(gl0 + LIM_REL, 0.0)))
        gt = g + gl
        a0, a1 = f(10.0) ** (gt0 / f(20.0)), f(10.0) ** (gt / f(20.0))
        n = RAMP if gt < gt0 else H
        if a0 != 1.0 or a1 != 1.0:
            a = np.where(ramp <= n, a0 + (a1 - a0) * ramp / f(n), a1).astype(np.float32)
            yb *= a
        out[i * H:(i + 1) * H] = soft_clip(yb)
        tr[i] = (r - p_in, lo, allow, g, gl)
        g0, gl0, gt0 = g, gl, gt
    return out, tr


def apply(x: np.ndarray, w: np.ndarray, p_in: float, p_out: float) -> np.ndarray:
    n = len(w)
    y = np.concatenate([np.zeros(LAT, np.float32), w.astype(np.float32)])
    xx = np.concatenate([x.astype(np.float32), np.zeros(max(len(y) - len(x), 0) + H, np.float32)])
    y = np.concatenate([y, np.zeros((-len(y)) % H, np.float32)])
    out, _ = stream(xx, y, p_in, p_out)
    return out[LAT:LAT + n]
