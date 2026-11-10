"""P5a(synth_nam_vc §8・学習なし): 前段の因果の包絡(128 帯 mel の log 振幅の DCT c0..c24)から共鳴の位置を取る = 写せない演奏の型の候補。
包絡 → mel の log 振幅(DCT の逆)→ 0〜F_BAND の線形の格子へ補間 → パワーの余弦変換で自己相関 → 帯域を選んだ線形予測(次数 P_LPC・Levinson)
→ 根から共鳴(周波数 > F_LO・帯域幅 < B_HI の低い順に 4 個・欠けは直前のフレームの値を保つ)と LSF(P_LPC 個)。
フレームごとに閉じる(先読み 0・発話の統計なし)。話者の座標への正規化(登録の平均と標準偏差)は呼ぶ側で登録の定数として。
    cd training && uv run --no-sync python snv_p5_formant.py test
"""
from __future__ import annotations

import sys

import numpy as np

import nvoc as N
import pae as PA

F_BAND = 5500.0
N_LIN = 257
P_LPC = 10
F_LO = 120.0
B_HI = 700.0
N_FMT = 4
F_DEF = np.array([500.0, 1500.0, 2500.0, 3500.0])
B_DEF = np.array([100.0, 120.0, 150.0, 200.0])
_C: dict = {}


def _consts() -> tuple[np.ndarray, np.ndarray]:
    if not _C:
        import librosa
        _, dct = PA._np_consts()
        fc = librosa.mel_frequencies(n_mels=N.N_MEL + 2, fmin=0.0, fmax=N.FMAX)[1:-1]
        fl = np.linspace(0.0, F_BAND, N_LIN)
        W = np.zeros((N_LIN, N.N_MEL))
        j = np.clip(np.searchsorted(fc, fl) - 1, 0, N.N_MEL - 2)
        t = np.clip((fl - fc[j]) / (fc[j + 1] - fc[j]), 0.0, 1.0)
        W[np.arange(N_LIN), j] = 1 - t
        W[np.arange(N_LIN), j + 1] = t
        _C["lin"] = W @ dct[:25].T
        m = np.arange(P_LPC + 1)[:, None]
        k = np.arange(N_LIN)[None]
        w = np.full(N_LIN, 2.0)
        w[0] = w[-1] = 1.0
        _C["cos"] = w * np.cos(np.pi * m * k / (N_LIN - 1))
    return _C["lin"], _C["cos"]


def lpc(env: np.ndarray) -> np.ndarray:
    lin, cos = _consts()
    P = np.exp(2.0 * (lin @ env.astype(np.float64)))
    r = (cos @ P).T
    r[:, 0] *= 1.0 + 1e-6
    T = r.shape[0]
    a = np.zeros((T, P_LPC + 1))
    a[:, 0] = 1.0
    e = r[:, 0].copy()
    for i in range(1, P_LPC + 1):
        k = -(r[:, i] + (a[:, 1:i] * r[:, i - 1:0:-1]).sum(1)) / e
        a[:, 1:i] = a[:, 1:i] + k[:, None] * a[:, i - 1:0:-1]
        a[:, i] = k
        e = e * (1 - k * k)
    return a


def _roots(c: np.ndarray) -> np.ndarray:
    T, n = c.shape[0], c.shape[1] - 1
    M = np.zeros((T, n, n))
    M[:, 0, :] = -c[:, 1:] / c[:, :1]
    M[:, np.arange(1, n), np.arange(n - 1)] = 1.0
    return np.linalg.eigvals(M)


def formants(env: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = _roots(lpc(env))
    f = np.angle(z) / np.pi * F_BAND
    b = -np.log(np.maximum(np.abs(z), 1e-9)) * 2.0 * F_BAND / np.pi
    ok = (z.imag > 0) & (f > F_LO) & (f < F_BAND - 100.0) & (b < B_HI)
    f = np.where(ok, f, np.inf)
    o = np.argsort(f, 1)[:, :N_FMT]
    F = np.take_along_axis(f, o, 1)
    B = np.take_along_axis(b, o, 1)
    T = len(F)
    Fo, Bo = np.empty((N_FMT, T)), np.empty((N_FMT, T))
    pf, pb = F_DEF.copy(), B_DEF.copy()
    for t in range(T):
        g = np.isfinite(F[t])
        pf = np.where(g, F[t], pf)
        pb = np.where(g, B[t], pb)
        Fo[:, t], Bo[:, t] = pf, pb
    return Fo.astype(np.float32), Bo.astype(np.float32)


def lsf(env: np.ndarray) -> np.ndarray:
    a = lpc(env)
    ar = a[:, ::-1]
    pz = np.concatenate([a, np.zeros((len(a), 1))], 1) + np.concatenate([np.zeros((len(a), 1)), ar], 1)
    qz = np.concatenate([a, np.zeros((len(a), 1))], 1) - np.concatenate([np.zeros((len(a), 1)), ar], 1)
    w = np.concatenate([np.angle(_roots(pz)), np.angle(_roots(qz))], 1)
    w = np.sort(np.where((w > 1e-4) & (w < np.pi - 1e-4), w, np.inf), 1)[:, :P_LPC]
    return (w / np.pi * F_BAND).T.astype(np.float32)


def _synth_env(fs: list[float], bs: list[float], tilt: float = 0.0) -> np.ndarray:
    fb, dct = PA._np_consts()
    fr = np.linspace(0, N.SR / 2, PA.NFFT // 2 + 1)
    zz = np.exp(-1j * 2 * np.pi * fr / N.SR)
    A = np.ones_like(zz)
    for f, b in zip(fs, bs):
        r = np.exp(-np.pi * b / N.SR)
        th = 2 * np.pi * f / N.SR
        A = A * (1 - 2 * r * np.cos(th) * zz + r * r * zz * zz)
    amp = np.abs(1 / A) * (1 + fr / 1000.0) ** (-tilt)
    return dct[:25] @ np.log(np.maximum(amp @ fb.T, 1e-5))


def _selftest() -> None:
    cases = [([500, 1500, 2500, 3500], [80, 100, 120, 150], 0.0), ([300, 2200, 2900, 3800], [60, 90, 120, 160], 1.0),
             ([800, 1200, 2600, 3600], [90, 100, 130, 180], 0.5), ([400, 1000, 2800, 4300], [70, 90, 140, 200], 1.5)]
    for fs, bs, tl in cases:
        env = np.stack([_synth_env(fs, bs, tl)] * 3, 1)
        F, B = formants(env)
        L = lsf(env)
        rel = np.abs(F[:, 0] / np.array(fs) - 1)
        print("true", fs, "est", np.round(F[:, 0]).tolist(), "B", np.round(B[:, 0]).tolist(), "maxrel", round(float(rel.max()), 3), "lsf", np.round(L[:, 0]).tolist())
        assert rel[:3].max() < 0.08, rel
        assert np.all(np.diff(L[:, 0]) > 0) and np.isfinite(L).all()
        assert np.allclose(F[:, 0], F[:, 2])
    env = np.stack([_synth_env([500, 1500, 2500, 3500], [80, 100, 120, 150]), np.zeros(25)], 1)
    env[:, 1] = -50.0
    F, _ = formants(env)
    assert np.isfinite(F).all()
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["test"]:
        _selftest()
