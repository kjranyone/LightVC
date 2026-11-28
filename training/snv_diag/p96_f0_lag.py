import json
import sys
from pathlib import Path

import librosa
import numpy as np

SR = 48000
STEP_MS = 2.0


def f0(w: np.ndarray, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    w16 = librosa.resample(w.astype(np.float32), orig_sr=SR, target_sr=16000)
    f, v, _ = librosa.pyin(w16, fmin=60, fmax=700, sr=16000, frame_length=1024, hop_length=32, center=True)
    lf = np.where(v & np.isfinite(f), np.log(np.nan_to_num(f, nan=1.0)), np.nan)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, lf)
    return lf


def detrend(lf: np.ndarray, n: int) -> np.ndarray:
    m = np.isfinite(lf).astype(float)
    z = np.nan_to_num(lf, nan=0.0)
    k = np.ones(n)
    s = np.convolve(z, k, mode="same")
    c = np.convolve(m, k, mode="same")
    return np.where((m > 0) & (c >= n * 0.5), lf - s / np.maximum(c, 1), np.nan)


def lag(fx: np.ndarray, fy: np.ndarray, shift_ratio: bool) -> list[float]:
    n = min(len(fx), len(fy))
    a0, b0 = detrend(fx[:n], 75), detrend(fy[:n], 75)
    lags = list(range(-10, 41))
    rs = []
    for l in lags:
        a = a0[10 : n - 41]
        b = b0[10 + l : n - 41 + l]
        m = np.isfinite(a) & np.isfinite(b)
        if shift_ratio:
            m &= np.abs(a - b) < 0.2
        av, bv = a[m] - a[m].mean(), b[m] - b[m].mean()
        rs.append(float((av * bv).sum() / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12)))
    j = int(np.argmax(rs))
    d = 0.0
    if 0 < j < len(rs) - 1:
        den = rs[j - 1] - 2 * rs[j] + rs[j + 1]
        d = 0.5 * (rs[j - 1] - rs[j + 1]) / den if den != 0 else 0.0
    return [round((lags[j] + d) * STEP_MS, 2), round(rs[j], 3)]


def main() -> None:
    work, dst = Path(sys.argv[1]), Path(sys.argv[2])
    res: dict = {}
    for sp in sys.argv[3:]:
        d = work / sp
        x = np.fromfile(d / "in.f32", dtype=np.float32)
        fx = f0(x, dst.parent / "pyin" / f"{sp}_in.npy")
        delayed = np.concatenate([np.zeros(480, np.float32), x])[: len(x)]
        r = {"calib_10ms": lag(fx, f0(delayed, dst.parent / "pyin" / f"{sp}_delayed.npy"), False)}
        for tag in ["copy_noesm", "copy", "conv"]:
            y = np.fromfile(d / f"p96_{tag}.f32", dtype=np.float32)
            fy = f0(y, dst.parent / "pyin" / f"{sp}_{tag}.npy")
            r[tag] = lag(fx, fy, False)
        res[sp] = r
        print(sp, json.dumps(r), flush=True)
    res["median_ms"] = {t: float(np.median([res[s][t][0] for s in sys.argv[3:]])) for t in ["calib_10ms", "copy_noesm", "copy", "conv"]}
    json.dump(res, open(dst, "w"), indent=1)
    print(json.dumps(res["median_ms"]))


if __name__ == "__main__":
    main()
