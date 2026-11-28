import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import snv_dev_eval as DE
import snv_p0_front as P

ROOT = Path(__file__).resolve().parent.parent.parent
WS = [int(w) for w in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1200, 960, 800, 600, 480]
LAGS = np.arange(-4, 9)


def peak(a: np.ndarray, r: np.ndarray) -> float:
    c = []
    for l in LAGS:
        t = np.arange(max(0, -l), min(len(r), len(a) - l))
        k = (r[t] > 0) & (a[t + l] > 0)
        u, v = np.log(a[t + l][k]), np.log(r[t][k])
        u, v = u - u.mean(), v - v.mean()
        c.append(float((u * v).sum() / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
    c = np.array(c)
    j = int(np.clip(c.argmax(), 1, len(c) - 2))
    den = c[j - 1] - 2 * c[j] + c[j + 1]
    return float((LAGS[j] + (np.clip(0.5 * (c[j - 1] - c[j + 1]) / den, -0.5, 0.5) if den < 0 else 0.0)) * 5.0)


def onsets(av: np.ndarray, rv: np.ndarray) -> list[float]:
    out = []
    for t in range(6, len(rv) - 12):
        if rv[t] and not rv[t - 6 : t].any() and rv[t : t + 10].all():
            w = np.nonzero(av[t - 5 : t + 16])[0]
            if len(w):
                out.append(float(w[0] - 5) * 5.0)
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    items = DE.cv_items("devA", n)
    acc = {w: {q: [] for q in ("hit", "false", "gross", "lag_v", "lag_f")} for w in WS}
    for it in items:
        x = librosa.load(it["u"]["wav_path"], sr=P.SR, mono=True)[0].astype(np.float32)
        x = x[: P.n_frames(len(x)) * P.H]
        T = P.n_frames(len(x))
        x16 = librosa.resample(x, orig_sr=P.SR, target_sr=16000)
        rf, rv, _ = librosa.pyin(x16, fmin=50, fmax=500, sr=16000, frame_length=1024, hop_length=80, center=True)
        rf = np.where(rv, np.nan_to_num(rf), 0.0)
        for w in WS:
            P.W_YIN = w
            f0 = P.f0_track(torch.from_numpy(x))[0].numpy()
            P.W_YIN = 1200
            m = min(T, len(rf), len(f0))
            a, r = f0[:m], rf[:m]
            av, rvv = a > 0, r > 0
            q = acc[w]
            q["hit"].append(float((av & rvv).sum() / max(rvv.sum(), 1)))
            q["false"].append(float((av & ~rvv).sum() / max((~rvv).sum(), 1)))
            b = av & rvv
            q["gross"].append(float((np.abs(1200 * np.log2(a[b] / r[b])) > 300).mean()) if b.sum() else 0.0)
            q["lag_v"] += onsets(av, rvv)
            q["lag_f"].append(peak(a, r))
    res = {str(w): {k: round(float(np.mean(v)), 4) for k, v in acc[w].items()} | {"lag_v_median": float(np.median(acc[w]["lag_v"])), "n_onsets": len(acc[w]["lag_v"])} for w in WS}
    for w in WS:
        print(w, json.dumps(res[str(w)]), flush=True)
    json.dump(res, open(ROOT / ("results/snv_p100/f0" + ("_" + sys.argv[3] if len(sys.argv) > 3 else "") + ".json"), "w"), indent=1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
