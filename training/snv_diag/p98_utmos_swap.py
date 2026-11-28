import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent.parent
PR = ROOT / "results/snv_p98/probe"
SPS = ["ca91ecce", "cb87e9f4", "cc5576e7", "cd094b9e"]
SR = 48000
CH = 8 * SR
HOP = 60


def env(x: np.ndarray, win: int) -> np.ndarray:
    w = np.hanning(win + 2)[1:-1]
    p = np.convolve(x.astype(np.float64) ** 2, w / w.sum(), mode="same")
    return np.sqrt(p[::HOP] + 1e-10)


def apply_gain(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    t = np.arange(len(x)) / HOP
    return (x * np.interp(t, np.arange(len(g)), g)).astype(np.float32)


def shift(e: np.ndarray, ms: float) -> np.ndarray:
    k = int(round(ms * SR / 1000 / HOP))
    return e if k == 0 else np.concatenate([e[k:], np.repeat(e[-1:], k)]) if k > 0 else np.concatenate([np.repeat(e[:1], -k), e[:k]])


def smooth(e: np.ndarray, ms: float) -> np.ndarray:
    k = int(round(ms * SR / 1000 / HOP))
    l = np.log(e)
    return np.exp(0.25 * np.roll(l, k) + 0.5 * l + 0.25 * np.roll(l, -k))


def main() -> None:
    arms, win_ms = sys.argv[1].split(","), float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    win = int(win_ms * SR / 1000)
    dev = "cuda"
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(dev).eval()

    def mos(x: np.ndarray) -> list[float]:
        out = []
        for i in range(0, len(x) - CH // 2, CH):
            c = resample_poly(x[i : i + CH].astype(np.float64), 1, 3).astype(np.float32)
            out.append(float(utmos(torch.from_numpy(c)[None].to(dev), 16000).item()))
        return out

    rd = lambda n, s: sf.read(PR / n / f"{s}.wav", dtype="float32")[0]
    summ = json.load(open(PR / "product_py/probe.json"))["summary"]["env_centroid_median"]
    res: dict = {"win_ms": win_ms, "rows": {}}
    acc: dict = {}
    for s in SPS:
        p = rd("product_py", s)
        ep = env(p, win)
        acc.setdefault("product", []).extend(mos(p))
        acc.setdefault("product+self_smooth5", []).extend(mos(apply_gain(p, smooth(ep, 5.0) / ep)))
        for a in arms:
            x = rd(a, s)[: len(p)]
            x = np.concatenate([x, np.zeros(len(p) - len(x), np.float32)])
            ex = env(x, win)
            d = summ - json.load(open(PR / a / "probe.json"))["summary"]["env_centroid_median"]
            cl = lambda g: np.clip(g, 0.1, 10.0)
            acc.setdefault(a, []).extend(mos(x))
            acc.setdefault(a + "+prod_env", []).extend(mos(apply_gain(x, cl(shift(ep, d) / ex))))
            acc.setdefault("product+env_of_" + a, []).extend(mos(apply_gain(p, cl(shift(ex, -d) / ep))))
            acc.setdefault(a + "+self_smooth5", []).extend(mos(apply_gain(x, smooth(ex, 5.0) / ex)))
            acc.setdefault(a + "+self_smooth10", []).extend(mos(apply_gain(x, smooth(ex, 10.0) / ex)))
    base = np.array(acc["product"])
    for k, v in acc.items():
        d = np.array(v) - base
        res["rows"][k] = {"utmos": round(float(np.mean(v)), 3), "diff": round(float(d.mean()), 3), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 3), "n": len(v)}
        print(k, res["rows"][k], flush=True)
    json.dump(res, open(ROOT / f"results/snv_p98/utmos_swap_w{int(win_ms)}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
