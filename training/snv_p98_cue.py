"""段 P98(paper_snv): 速い道の読み = 入力の波形から直に読む音量(山を追う読み・1.25 ms 刻み・先読み 0・発話全体の統計なし)。
帯域 4 つ(全帯域・700 Hz 未満・700〜3000 Hz・3000 Hz 超 = 2 次の Butterworth の因果の IIR)ごとに、60 標本の小区間の |x| の最大 m_k を取り、
s_k = max(m_k, s_{k−1}·g)(立ち上がりは即・戻りは 10 ms)。保存するのは log(s_k + 1e-7)(float16・[4, K])。
使うとき: a_k = log(G·s_k + 1e-4)(G = マイクのゲイン)・u_k = (a_k − ema_k)·0.25(ema = 時定数 40 ms の 1 次の走行平均・初期値 a_0)= ゲインに依らない「直近の音量の動き」。
    cd training && uv run --no-sync python snv_p98_cue.py pack
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt

ROOT = Path(__file__).resolve().parent.parent
SR = 48000
SUB = 60
KSUB = 4
REL_MS = 10.0
TAU_MS = 40.0
EPS_STORE = 1e-7
EPS = 1e-4
SCALE = 0.25
OUT = ROOT / "data/snv_p98_cue"
SOS = [butter(2, 700.0, "lowpass", fs=SR, output="sos"), butter(2, [700.0, 3000.0], "bandpass", fs=SR, output="sos"), butter(2, 3000.0, "highpass", fs=SR, output="sos")]


def followers(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    K = len(x) // SUB
    c = SUB / SR * 1000.0 / REL_MS
    k = np.arange(K) * c
    out = []
    for b in [x] + [sosfilt(s, x) for s in SOS]:
        m = np.abs(b[: K * SUB]).reshape(K, SUB).max(1)
        ls = np.maximum.accumulate(np.log(np.maximum(m, 1e-12)) + k) - k
        out.append(np.log(np.exp(ls) + EPS_STORE))
    return np.stack(out).astype(np.float32)


def cue(l: np.ndarray, gain: float = 1.0) -> np.ndarray:
    a = np.log(gain * np.maximum(np.exp(l.astype(np.float64)) - EPS_STORE, 0.0) + EPS)
    b = 1.0 - np.exp(-(SUB / SR * 1000.0) / TAU_MS)
    ema, _ = lfilter([b], [1.0, -(1.0 - b)], a, axis=-1, zi=(a[:, :1] * (1.0 - b)))
    return ((a - ema) * SCALE).astype(np.float32)


def _one(job: tuple[str, str]) -> int:
    src, dst = job
    if Path(dst).exists():
        return 0
    x, sr = sf.read(src, dtype="float32")
    assert sr == SR and x.ndim == 1, src
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, followers(x).astype(np.float16))
    return 1


def pack() -> None:
    jobs = []
    for side, base, ext in [("t", ROOT / "data/snv_p97_wav", "flac"), ("x", ROOT / "data/snv_p_wav_x16", "wav")]:
        for kind in ["e1", "e1nv"]:
            for w in sorted((base / kind).glob(f"*/*.{ext}")):
                jobs.append((str(w), str(OUT / side / kind / w.parent.name / f"{w.stem}.npy")))
    with ProcessPoolExecutor(12) as ex:
        n = sum(ex.map(_one, jobs, chunksize=64))
    print("files", len(jobs), "written", n, flush=True)


def _selftest() -> None:
    rng = np.random.default_rng(0)
    x = np.zeros(SR, np.float32)
    x[12000:36000] = 0.3 * np.sin(2 * np.pi * 120 * np.arange(24000) / SR) * (1 + 0.3 * rng.standard_normal(24000))
    l = followers(x)
    u, u2 = cue(l), cue(l, 4.0)
    k0 = 12000 // SUB
    print("立ち上がりの小区間", k0, "全帯域の u", np.round(u[0, k0 - 2 : k0 + 4], 3), "ゲイン 4 倍との差の最大(声の区間)", float(np.abs(u - u2)[:, k0 + 2 : 36000 // SUB].max()))
    s = np.exp(l[0].astype(np.float64)) - EPS_STORE
    ref, st = [], 0.0
    g = np.exp(-(SUB / SR * 1000.0) / REL_MS)
    for m in np.abs(x[: len(x) // SUB * SUB]).reshape(-1, SUB).max(1):
        st = max(m, st * g)
        ref.append(st)
    assert np.allclose(s, ref, rtol=1e-3, atol=1e-6), float(np.abs(s - np.array(ref)).max())
    x2 = x.copy()
    x2[30000:] = rng.standard_normal(SR - 30000).astype(np.float32)
    d = np.abs(cue(followers(x2)) - u)
    first = int(np.nonzero(d.max(0) > 0)[0][0])
    assert first >= 30000 // SUB, first
    print("未来不変性: 標本 30000 以降の書き換え → 最初に変わる小区間", first, "(= 標本", first * SUB, ")")
    print("snv_p98_cue selftest OK")


if __name__ == "__main__":
    pack() if sys.argv[1:] == ["pack"] else _selftest()
