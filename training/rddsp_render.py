import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from kansei_train import CACHE, DATA

OUT = Path("../results/rddsp_v16")


def gain_match(y: np.ndarray, gt: np.ndarray) -> np.ndarray:
    g = np.sqrt((gt ** 2).mean() / ((y ** 2).mean() + 1e-12))
    y = y * g
    p = np.abs(y).max()
    return (y * (0.95 / p) if p > 0.95 else y).astype(np.float32)


def main() -> None:
    import librosa
    OUT.mkdir(parents=True, exist_ok=True)
    for uid in [p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]]:
        x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
        gt = torch.tensor(x[: R.SR * 8])
        y, h, nz, p = R.resynthesize(gt)
        n = min(len(y), len(gt))
        g = gt[:n].numpy()
        sf.write(OUT / f"{uid}_gt.wav", g, R.SR)
        sf.write(OUT / f"{uid}_rddsp.wav", gain_match(y[:n].numpy(), g), R.SR)
        R.NOISE_HOPDIV = 4
        yo, _, _, _ = R.resynthesize(gt)
        sf.write(OUT / f"{uid}_v15old.wav", gain_match(yo[:n].numpy(), g), R.SR)
        R.NOISE_HOPDIV = 6
        R.NOISE_GL, R.GLOBAL_GAIN = 8, True
        y8, _, _, _ = R.resynthesize(gt)
        sf.write(OUT / f"{uid}_offline.wav", gain_match(y8[:n].numpy(), g), R.SR)
        R.NOISE_GL, R.GLOBAL_GAIN, R.NOISE_PREFILTER = 0, False, True
        R.NOISE_HOPDIV = 6
        print(f"    GCI {len(p['gci'])}  interval med {float(__import__('torch').diff(p['gci']).float().median()):.0f}")
        sf.write(OUT / f"{uid}_harmonic.wav", gain_match(h[:n].numpy(), g), R.SR)
        sf.write(OUT / f"{uid}_breath.wav", gain_match(nz[:n].numpy(), g), R.SR)
        f0 = p["f0"]
        print(f"  {uid}  f0 {float(f0[f0>50].median()):5.0f}Hz  "
              f"MVF {float(p['mvf'][f0>50].median()):6.0f}Hz  "
              f"voiced {100*float((f0>50).float().mean()):4.1f}%  "
              f"ap[0] {float(p['ap'][0].median()):.3f}")
    print(f"wrote -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
