"""Control for the freeC reading on the H-F test set.

rddsp_calibrate_hf.py scores freeC at ~1.7-2.0 on unseen speakers, against the
2.888 recorded in 9.4. Three explanations are possible and they have completely
different consequences:

  1. the code path here differs from the one that produced 2.888 (a bug in this
     file, and the number means nothing)
  2. the speakers differ: 9.4's set is twelve utterances of af1ad5575a3fa383,
     which is the speaker the project has the most data for
  3. freeC genuinely collapses on unseen speakers

Running the IDENTICAL code path on af1ad separates 1 from 2 and 3: if af1ad
comes back near 2.888 here, the path is sound and the drop is about speakers,
not about this file. Nothing in this script may differ from the calibration
except which audio goes in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import score_one
from rddsp_gpu import build
from kansei_train import DATA, CACHE
from train_z1 import load_freebig
from causal_mel import centered_mel

SECONDS = 5


def main() -> None:
    vf = load_freebig("checkpoints/freeC/foundation_lowlatency_5p8ms.pt")[0].to("cpu").eval()

    uids = [p.stem for p in sorted(CACHE.glob("*.npz"))][40:52]
    a = []
    with torch.no_grad():
        for uid in uids:
            x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
            gt = torch.tensor(x[: R.SR * SECONDS])
            y = vf(centered_mel(gt[None], n_fft=2048, hop=128)).squeeze()
            a.append(score_one(gt, y[: gt.shape[-1]]))
    print(f"  af1ad5575a3fa383 (9.4's calibration speaker), {len(a)} utterances: "
          f"{np.mean(a):.4f}", flush=True)

    _, te = build(80, 12)
    b = []
    with torch.no_grad():
        for it in te:
            gt = it["gt"]
            y = vf(centered_mel(gt[None], n_fft=2048, hop=128)).squeeze()
            b.append(score_one(gt, y[: gt.shape[-1]]))
    print(f"  H-F test set, {len(b)} unseen speakers:                      "
          f"{np.mean(b):.4f}", flush=True)
    print(f"\n  same weights, same mel interface, same scorer, same file.")
    print(f"  difference: {np.mean(a) - np.mean(b):+.4f}")


if __name__ == "__main__":
    main()
