import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import snv_dev_eval as DE
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent
FORMS = {
    "product": ("", {}),
    "n240_10k": ("results/snv_p97_n240/snap/ema_10k.pt", {}),
    "n240_10k_e70": ("results/snv_p97_n240/snap/ema_10k.pt", {"SNV_EK": "0.7,0.3,0,0"}),
    "n240_10k_3pt": ("results/snv_p97_n240/snap/ema_10k.pt", {"SNV_EK": "0.25,0.5,0.25,0"}),
    "r240_10k": ("results/snv_p99_r240/snap/ema_10k.pt", {}),
    "r240_10k_e70": ("results/snv_p99_r240/snap/ema_10k.pt", {"SNV_EK": "0.7,0.3,0,0"}),
}


def lm(x: np.ndarray) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=x, sr=48000, n_fft=1024, hop_length=240, n_mels=64, fmin=60, fmax=7600, power=1.0)
    return np.log(m + 1e-4)


def best(a: np.ndarray, y: np.ndarray) -> tuple[float, int, np.ndarray]:
    r = []
    for s in range(0, 1441, 120):
        b = lm(y[s:])
        n = min(a.shape[1], b.shape[1])
        r.append((float(np.abs(a[:, :n] - b[:, :n]).mean()), s, b[:, :n]))
    return min(r, key=lambda q: q[0])


def rough(m: np.ndarray, act: np.ndarray) -> float:
    d = np.abs(np.diff(m, axis=1))
    k = act[1:] & act[:-1]
    return float(d[:, k[: d.shape[1]]].mean())


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    items = DE.cv_items("devA", n)
    src = [librosa.load(it["u"]["wav_path"], sr=EV.SR, mono=True)[0].astype(np.float32) for it in items]
    A = [lm(x) for x in src]
    res: dict = {"n": len(items)}
    for name, (ck, sw) in FORMS.items():
        for k in ("SNV_ESM", "SNV_EK", "SNV_EK0", "SNV_RCK"):
            os.environ.pop(k, None)
        os.environ.update(sw)
        if ck:
            os.environ["SNV_RCK"] = ck
        gen = EV.load_gen()
        dist, shift, rr = [], [], []
        for it, a in zip(items, A):
            u = it["u"]
            y, f0 = u["y"], u["f0"]
            v = (f0 > 0).astype(np.float32)
            lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
            cond = torch.from_numpy(np.concatenate([y[:25], lf[None], v[None], np.clip(y[25:29], 0, 1.5) * v[None]], 0))[None].to(EV.DEV)
            cond = torch.cat([EV.smooth_env(gen, cond[:, :25]), cond[:, 25:]], 1)
            w = EV.draw(gen, cond, torch.from_numpy(f0.astype(np.float32))[None].to(EV.DEV), None)
            d, s, b = best(a, np.asarray(w, np.float32))
            e = a.mean(0)
            act = e > np.percentile(e, 40)
            dist.append(d)
            shift.append(s)
            rr.append(rough(b, act[: b.shape[1]]) / rough(a[:, : b.shape[1]], act[: b.shape[1]]))
        res[name] = {"logmel_l1": round(float(np.mean(dist)), 4), "shift_median": float(np.median(shift)), "rough_ratio": round(float(np.mean(rr)), 4), "rough_ratio_se": round(float(np.std(rr) / np.sqrt(len(rr))), 4), "dist": [round(q, 4) for q in dist]}
        print(name, {k: q for k, q in res[name].items() if k != "dist"}, flush=True)
    p = np.array(res["product"]["dist"])
    for name in FORMS:
        d = np.array(res[name]["dist"]) - p
        print(name, "logmel − product", round(float(d.mean()), 4), "±", round(float(1.96 * d.std() / np.sqrt(len(d))), 4))
        res[name]["vs_product"] = [round(float(d.mean()), 4), round(float(1.96 * d.std() / np.sqrt(len(d))), 4)]
    json.dump(res, open(ROOT / "results/snv_p99/faith.json", "w"), indent=1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
