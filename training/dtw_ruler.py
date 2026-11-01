"""DTW で時間を合わせる解析の物差し(同じ文の別の発話どうしを比べる = 変換の採点に使う)。
dist(y, x): 両方を各自の教師 f0 で CheapTrick → DCT24(c1..24)と 128 mel の対数(発話平均を引く)→ DCT24 の系列で DTW(ユークリッド)→ 経路に沿った env の RMSE・logmel の L1(両方有声の対 / 全対)。
"""
from __future__ import annotations

import numpy as np


def analyse(x: np.ndarray) -> dict:
    import librosa
    import f0hi as H
    import nvoc as N
    import pae as PA
    n = len(x) // N.HOP
    x = x[: n * N.HOP].astype(np.float32)
    f0, _ = H.teacher_f0(x, n)
    f0 = f0[:n].astype(np.float64)
    xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
    env = PA.envelope(xa, f0)[:, :n].astype(np.float64)
    m = np.log(librosa.feature.melspectrogram(y=x, sr=48000, n_fft=2048, hop_length=240, n_mels=128, fmax=16000, power=2.0) + 1e-8)[:, :n]
    m = np.pad(m, ((0, 0), (0, n - m.shape[1])), mode="edge")
    return {"env": env, "lm": m - m.mean(), "v": f0 > 0, "n": n}


def dist(ay: dict, ax: dict) -> dict:
    import librosa
    D, wp = librosa.sequence.dtw(X=ay["env"][1:25], Y=ax["env"][1:25], metric="euclidean")
    i, j = wp[::-1, 0], wp[::-1, 1]
    vv = ay["v"][i] & ax["v"][j]
    de = ay["env"][1:25][:, i] - ax["env"][1:25][:, j]
    dl = ay["lm"][:, i] - ax["lm"][:, j]
    return {"env": float(np.sqrt((de[:, vv] ** 2).mean())) if vv.any() else float("nan"), "lm": float(np.abs(dl).mean()),
            "lm_v": float(np.abs(dl[:, vv]).mean()) if vv.any() else float("nan"), "path_len_ratio": float(len(i) / max(ay["n"], ax["n"]))}
