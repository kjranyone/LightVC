"""PESQ-driven improvement loop.

Objective is PESQ-vs-gt, the only metric this project has validated against the
ear. LSD/crest/band-energy are NOT used: they moved monotonically the right way
across v2->v4 while PESQ went the other way.

Analysis is cached per utterance so a synthesis-side candidate costs one render,
not one full analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from pesq import pesq
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from kansei_train import CACHE, DATA

N_SEC = 8


def align(ref, deg):
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    lag = int(np.argmax(np.abs(correlate(deg, ref, mode="full")))) - (len(ref) - 1)
    if lag > 0:
        deg = np.concatenate([deg[lag:], np.zeros(lag, np.float32)])
    elif lag < 0:
        deg = np.concatenate([np.zeros(-lag, np.float32), deg[:lag]])
    return ref, deg


def score_one(gt: torch.Tensor, y: torch.Tensor) -> float:
    n = min(len(y), len(gt))
    a = gt[:n].numpy().astype(np.float64)
    b = y[:n].numpy().astype(np.float64)
    b = b * np.sqrt((a ** 2).mean() / ((b ** 2).mean() + 1e-12))
    ar = librosa.resample(a, orig_sr=R.SR, target_sr=16000)
    br = librosa.resample(b, orig_sr=R.SR, target_sr=16000)
    ar, br = align(ar, br)
    return pesq(16000, ar, br, "wb")


class Cache:
    def __init__(self):
        self.items = []
        for uid in [p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]]:
            x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
            gt = torch.tensor(x[: R.SR * N_SEC])
            self.items.append((uid, gt))

    @property
    def an_gt(self):
        return [(gt,) for _, gt in self.items]

    def build(self):
        """(Re)run analysis with the current module-level settings."""
        self.an = []
        for uid, gt in self.items:
            p = R.analyze(gt)
            f0 = p["f0"]
            fm = float(f0[f0 > 50].median())
            gci, soe = R.zff_gci(gt, fm)
            f0_at = f0[(gci // R.HOP).clamp(0, len(f0) - 1)]
            vm = f0_at > 50
            gci, soe, f0_at = gci[vm], soe[vm], f0_at[vm]
            if R.GCI_REFINE:
                gci, soe = R.refine_gci(gci, soe, f0_at)
                f0_at = f0[(gci // R.HOP).clamp(0, len(f0) - 1)]
            Xg = R.harmonic_analysis_at(gt, gci, f0_at)
            self.an.append((gt, p, gci, Xg, f0_at, fm))
        return self

    def score(self, **kw) -> float:
        """Go through resynthesize(), the same path the renderer uses.

        Calling synthesize_v2() directly skipped every argument resynthesize()
        builds itself -- harm_ap among them, which is why HARM_AP measured as
        exactly 0.0000 and looked like "no effect" when it was simply never
        applied."""
        out = []
        for gt, *_ in self.an_gt:
            y, _, _, _ = R.resynthesize(gt, **kw)
            out.append(score_one(gt, y))
        return float(np.mean(out))

if __name__ == "__main__":
    round5()
