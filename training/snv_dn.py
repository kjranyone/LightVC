"""paper_snv §5 M45(段 P61): 符号の入力だけ、続く雑音の床を引く(学習なし・因果・登録の定数と過去だけ)。
包絡 c0..c24 を 128 帯 mel の対数振幅に戻し、帯ごとのパワー P(t, b) から雑音の床 N(t, b) を a 倍して引く: P' = max(P − a N, f² P)。対数に戻して DCT c0..c24。
N = 過去 win フレーム(1.5 s)の、時間で均した P の最小(発話の前は操作者の登録の床で埋める)。登録の床 = 登録の全フレームの、均した P の帯ごとの下位 q 分位。
描画の経路(回路の出力・高さ・有声・周期性)には使わない。演奏の入力(源と操作者の登録)の包絡だけに掛ける。
    cd training && uv run --no-sync python snv_dn.py test
"""
from __future__ import annotations

import sys

import numpy as np

import pae as PA
import train_snv_p1 as T

WIN, SMOOTH, Q = 300, 10, 0.05


def _power(env: np.ndarray) -> np.ndarray:
    _, dct = PA._np_consts()
    return np.exp(2.0 * (dct[:25].T @ env.astype(np.float64)))


def _smooth(P: np.ndarray, n: int = SMOOTH) -> np.ndarray:
    a = np.exp(-1.0 / n)
    out = np.empty_like(P)
    acc = P[:, 0].copy()
    for t in range(P.shape[1]):
        acc = a * acc + (1 - a) * P[:, t]
        out[:, t] = acc
    return out


def floor_of(envs: list[np.ndarray], q: float = Q) -> np.ndarray:
    return np.quantile(np.concatenate([_smooth(_power(e)) for e in envs], 1), q, axis=1)


def denoise(env: np.ndarray, n0: np.ndarray, a: float = 2.0, f: float = 0.1, win: int = WIN) -> np.ndarray:
    _, dct = PA._np_consts()
    P = _power(env)
    S = np.concatenate([np.repeat(n0[:, None], win - 1, 1), _smooth(P)], 1)
    Nf = np.stack([S[:, t:t + win].min(1) for t in range(P.shape[1])], 1)
    P2 = np.maximum(P - a * Nf, f * f * P)
    return (dct[:25] @ (0.5 * np.log(np.maximum(P2, 1e-10)))).astype(np.float32)


def y_env(y: np.ndarray) -> np.ndarray:
    return np.concatenate([y[:1] * 10 + PA.C0_SIL, y[1:25] * 10], 0)


def dn_utts(us: list[dict], enr: list[dict], a: float = 2.0, f: float = 0.1) -> tuple[list[dict], list[dict]]:
    n0 = floor_of([y_env(u["y"]) for u in enr])

    def one(u: dict) -> dict:
        e = denoise(y_env(u["y"]), n0, a, f)
        y = u["y"].copy()
        y[0], y[1:25] = (e[0] - PA.C0_SIL) / 10, e[1:25] / 10
        s, n = T.voiced_mean(e, u["f0"])
        return u | {"y": y, "vs": s.astype(np.float32), "vn": n}
    return [one(u) for u in us], [one(u) for u in enr]


def test() -> None:
    g = np.random.default_rng(0)
    _, dct = PA._np_consts()
    Tn = 600
    on = (np.arange(Tn) % 200 >= 100)[None]
    Ls = np.log(1e-4 + on * np.linspace(1, 0.5, 128)[:, None])
    Ln = np.log(0.05) + 0.1 * g.standard_normal(128)[:, None] + np.zeros((1, Tn))
    clean = dct[:25] @ Ls
    noisy = dct[:25] @ (0.5 * np.log(np.exp(2 * Ls) + np.exp(2 * Ln)))
    n0 = floor_of([noisy])
    d = denoise(noisy, n0, 2.0, 0.1)
    lo = lambda e: dct[:25].T @ e
    off = ~on[0]
    off[:100] = True
    drop = float((lo(noisy) - lo(d))[:, off][:, 30:].mean())
    assert abs(drop - np.log(10.0)) < 0.3, drop
    keep = float(np.abs(lo(d) - lo(noisy))[:, on[0]].mean())
    assert keep < 0.05, keep
    d2 = denoise(noisy[:, :400], n0, 2.0, 0.1)
    assert np.abs(d2 - d[:, :400]).max() < 1e-6, "因果でない"
    same = denoise(clean, floor_of([clean]), 0.0, 0.1)
    assert np.abs(same - clean).max() < 1e-3, np.abs(same - clean).max()
    print("snv_dn test ok(雑音だけのフレームは床まで下がる", round(drop, 2), "・声の強いフレームは動かない", round(keep, 3), "・因果・a = 0 で恒等)")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        test()
