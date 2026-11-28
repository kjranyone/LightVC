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
EDGES = [0, 500, 1000, 2000, 4000, 8000, 12000, 24000]
FORMS = {
    "product": "",
    "n240_10k": "results/snv_p97_n240/snap/ema_10k.pt",
    "r240_10k": "results/snv_p99_r240/snap/ema_10k.pt",
    "rc240_5k": "results/snv_p99_rc240/snap/ema_5k.pt",
}


def bands(x: np.ndarray) -> np.ndarray:
    n = len(x) // 2048 * 2048
    s = (np.abs(np.fft.rfft(x[:n].reshape(-1, 2048) * np.hanning(2048), axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(2048, 1 / 48000)
    return np.array([10 * np.log10(s[(f >= a) & (f < b)].sum() + 1e-12) for a, b in zip(EDGES[:-1], EDGES[1:])])


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    items = DE.cv_items("devA", n)
    src = np.stack([bands(librosa.load(it["u"]["wav_path"], sr=EV.SR, mono=True)[0].astype(np.float32)) for it in items])
    res: dict = {"edges": EDGES, "n": len(items), "src_sd": np.round(src.std(0), 2).tolist()}
    for name, ck in FORMS.items():
        for k in ("SNV_ESM", "SNV_EK", "SNV_EK0", "SNV_RCK"):
            os.environ.pop(k, None)
        if ck:
            os.environ["SNV_RCK"] = ck
        gen = EV.load_gen()
        out = []
        for it in items:
            u = it["u"]
            y, f0 = u["y"], u["f0"]
            v = (f0 > 0).astype(np.float32)
            lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
            cond = torch.from_numpy(np.concatenate([y[:25], lf[None], v[None], np.clip(y[25:29], 0, 1.5) * v[None]], 0))[None].to(EV.DEV)
            cond = torch.cat([EV.smooth_env(gen, cond[:, :25]), cond[:, 25:]], 1)
            out.append(bands(EV.draw(gen, cond, torch.from_numpy(f0.astype(np.float32))[None].to(EV.DEV), None)))
        d = np.stack(out) - src
        d = d - d[:, :5].mean(1, keepdims=True)
        sc = src - src[:, :5].mean(1, keepdims=True)
        slope = [float(np.polyfit(sc[:, b] - sc[:, b].mean(), d[:, b] - d[:, b].mean(), 1)[0]) for b in range(len(EDGES) - 1)]
        res[name] = {"mean": np.round(d.mean(0), 2).tolist(), "sd": np.round(d.std(0), 2).tolist(), "slope": np.round(slope, 3).tolist()}
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(ROOT / "results/snv_p99/ltas.json", "w"), indent=1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
