"""Where does the project's OWN realtime vocoder sit in this harness?

rddsp_calibrate.py placed BigVGAN v2 at 4.228 and the DSP core at 3.159 on the
same twelve held-out utterances, so PESQ 4.0 is reachable here -- by a 122M
non-causal GAN vocoder at hop 512. The question that actually matters for this
product is where a LOW-LATENCY neural vocoder lands, and the project already
ships one (freeC), so it is measurable rather than arguable.

Three systems, one scorer, one set of utterances:
  identity        the metric's own ceiling
  BigVGAN v2      large, non-causal, the field's reference point
  freeC           this project's realtime vocoder
  rddsp           this session's parametric DSP core
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from kansei_train import CACHE, DATA
from rddsp_loop import score_one
from train_z1 import load_freebig
from causal_mel import causal_mel, centered_mel

DEV = "cpu"
CK = "checkpoints/freeC/foundation_lowlatency_5p8ms.pt"


def main() -> None:
    vf = load_freebig(CK)[0]
    vf = vf.to(DEV).eval()
    n_par = sum(p.numel() for p in vf.parameters()) / 1e6
    print(f"freeC: {n_par:.1f}M params", flush=True)

    uids = [p.stem for p in sorted(CACHE.glob("*.npz"))][40:52]
    free_c, free_cz, dsp = [], [], []
    with torch.no_grad():
        for uid in uids:
            x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
            gt = torch.tensor(x[: R.SR * 6])
            g = gt[None].to(DEV)
            yc = vf(centered_mel(g, n_fft=2048, hop=128)).squeeze().cpu()
            yz = vf(causal_mel(g, n_fft=2048, hop=128)).squeeze().cpu()
            n = min(len(yc), len(gt))
            free_c.append(score_one(gt[:n], yc[:n]))
            free_cz.append(score_one(gt[:n], yz[:n]))
            yd, _, _, _ = R.resynthesize(gt)
            dsp.append(score_one(gt, yd))
            print(f"  {uid}  freeC(centered) {free_c[-1]:.3f}  "
                  f"freeC(causal mel) {free_cz[-1]:.3f}  rddsp {dsp[-1]:.3f}", flush=True)

    print(f"\n{'system':42s} {'PESQ':>7s}")
    print(f"{'identity (STFT round trip)':42s} {4.644:7.3f}")
    print(f"{'BigVGAN v2 44kHz (122M, non-causal)':42s} {4.228:7.3f}")
    print(f"{'freeC (this project, realtime, centered mel)':42s} {np.mean(free_c):7.3f}")
    print(f"{'freeC (causal mel)':42s} {np.mean(free_cz):7.3f}")
    print(f"{'rddsp DSP core (parametric, causal)':42s} {np.mean(dsp):7.3f}")


if __name__ == "__main__":
    main()
