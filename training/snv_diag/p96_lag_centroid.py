import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from p96_lag_probe import band_env, highpass

LO, HI = -20, 80


def curve(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> np.ndarray:
    n = min(len(a), len(b))
    a, b, m = a[:n], b[:n], m[:n]
    idx = np.nonzero(m)[0]
    idx = idx[(idx + HI < n) & (idx + LO >= 0)]
    av = a[idx] - a[idx].mean()
    r = []
    for l in range(LO, HI + 1):
        bv = b[idx + l] - b[idx + l].mean()
        r.append(float((av * bv).sum() / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12)))
    return np.array(r)


def centroid(r: np.ndarray) -> list[float]:
    j = int(r.argmax())
    w = np.maximum(r - 0.5 * r[j], 0.0)
    lo = j
    while lo > 0 and w[lo - 1] > 0:
        lo -= 1
    hi = j
    while hi < len(w) - 1 and w[hi + 1] > 0:
        hi += 1
    l = np.arange(LO, HI + 1)[lo : hi + 1]
    return [round(float((w[lo : hi + 1] * l).sum() / w[lo : hi + 1].sum()), 2), int(LO + j), round(float(r[j]), 3)]


def lag(x: np.ndarray, y: np.ndarray, band: str = "all") -> list[float]:
    n = min(len(x), len(y))
    ex, ey = band_env(x[:n]), band_env(y[:n])
    act = ex["all"] > ex["all"].max() - 35
    return centroid(curve(highpass(ex[band], 200), highpass(ey[band], 200), act))


def main() -> None:
    work, dst = Path(sys.argv[1]), Path(sys.argv[2])
    sps = sys.argv[3:]
    res: dict = {}
    names = ["calib", "p96_copy_noesm", "p96_copy", "p96_conv"] + [f"p96k_{k}_{m}" for k in ["K_now", "K_half", "K_zero", "K_q", "K_none"] for m in ["conv", "copy"]]
    for nm in names:
        v = {}
        for sp in sps:
            x = np.fromfile(work / sp / "in.f32", dtype=np.float32)
            y = np.concatenate([np.zeros(480, np.float32), x])[: len(x)] if nm == "calib" else np.fromfile(work / sp / f"{nm}.f32", dtype=np.float32)
            v[sp] = {b: lag(x, y, b) for b in ["all", "lo", "mid", "hi"]}
        res[nm] = v
        res[nm + "/median_centroid_ms"] = {b: float(np.median([v[s][b][0] for s in sps])) for b in ["all", "lo", "mid", "hi"]}
        print(nm, "centroid", res[nm + "/median_centroid_ms"], "per-speaker all [centroid, argmax, r]:", [v[s]["all"] for s in sps], flush=True)
    json.dump(res, open(dst, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
