"""段 P118(学習なし): 描画器が乗せる雑音の物差し。帯域ごとに、隣り合う周期どうしの波形の相関(周期性)を測る。声の区間・高さは元の音の pYIN。相関 r を 10 log10(r / (1 − r))(dB・高いほど澄む)で出す。
    cd training && uv run --no-sync python snv_diag/p118_bp.py <元の音> <比べる音> [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

BANDS = [(80, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
SR = 48000


def f0_of(x: np.ndarray) -> np.ndarray:
    f, vf, _ = librosa.pyin(librosa.resample(x, orig_sr=SR, target_sr=16000), fmin=70, fmax=700, sr=16000, frame_length=1024, hop_length=160, center=True)
    return np.where(vf, np.nan_to_num(f), 0.0)


def bp(x: np.ndarray, f0: np.ndarray) -> dict:
    out = {}
    for lo, hi in BANDS:
        xb = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x)
        rs = []
        for t in np.where(f0 > 0)[0]:
            T0 = SR / f0[t]
            c = t * 480
            L = int(2 * T0)
            a = c - L
            if a < 0 or c + L + int(T0) + 4 > len(xb):
                continue
            s0 = xb[a : c + L]
            best = -1.0
            for o in (-2, -1, 0, 1, 2):
                d = int(round(T0)) + o
                s1 = xb[a + d : c + L + d]
                best = max(best, float((s0 * s1).sum() / (np.sqrt((s0 * s0).sum() * (s1 * s1).sum()) + 1e-20)))
            rs.append(min(max(best, 1e-3), 0.999))
        r = np.array(rs)
        out[(lo, hi)] = float(np.median(10 * np.log10(r / (1 - r)))) if len(r) else float("nan")
    return out


def main() -> None:
    x = librosa.load(sys.argv[1], sr=SR, mono=True)[0]
    f0 = f0_of(x)
    a = bp(x, f0)
    fmt = lambda d: " ".join(f"{v:+5.1f}" for v in d.values())
    print("帯域(Hz)                  ", " ".join(f"{lo//1000 if lo >= 1000 else 0.08:>4}k" for lo, _ in BANDS))
    print(Path(sys.argv[1]).name.ljust(26), fmt(a))
    for p in sys.argv[2:]:
        y = librosa.load(p, sr=SR, mono=True)[0]
        n = min(len(x), len(y))
        b = bp(y[:n], f0[: n // 480])
        print(Path(p).name.ljust(26), fmt(b), "| 差", " ".join(f"{b[k] - a[k]:+5.1f}" for k in a))


if __name__ == "__main__":
    main()
