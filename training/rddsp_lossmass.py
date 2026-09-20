"""Where does the magnitude loss actually spend itself?

12.21'' left one thing unexplained and non-tautological: `mrstft` contains a
LOG-MAGNITUDE L1 term at three resolutions -- a loss whose whole job is to fix
magnitude -- and yet 6000 GPU steps of a 0.501M net moved the octave-band level
error by 0.045 dB, one seventh of what 257 fixed numbers do.

A log-magnitude L1 has a known failure mode. With

    L = | log(|Y| + eps) - log(|T| + eps) |,   eps = 1e-5

bins where the target is NEAR SILENT are not down-weighted -- they are
UP-weighted, because the log expands the bottom of the range. A bin at -80 dB
that is wrong by a factor of 3 contributes as much as a bin at -10 dB that is
wrong by a factor of 3, and there are vastly more of the former. If most of the
loss mass sits below the level where anything is audible, the optimiser is
spending its capacity on silence, and the bands the listener named get whatever
is left.

Measure it directly: bucket every (frequency, time) bin by its TARGET level
relative to that utterance's peak, and report what fraction of the loss, and of
the loss GRADIENT magnitude, each bucket carries. Buckets are in dB below peak,
so "-80 dB and below" is inaudible by any standard.

No model is involved -- this is a property of the loss and the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_gpu import build

RES = ((256, 64), (512, 128), (1024, 256))
EDGES = [0, -20, -40, -60, -80, -100, -300]


def main() -> None:
    _, te = build(80, 12)
    mass = np.zeros(len(EDGES) - 1)
    grad = np.zeros(len(EDGES) - 1)
    count = np.zeros(len(EDGES) - 1)
    for it in te:
        n = min(it["gt"].shape[-1], it["y"].shape[-1])
        gt, y = it["gt"][:n], it["y"][:n]
        for nf, hp in RES:
            w = torch.hann_window(nf)
            T = torch.stft(gt, nf, hp, nf, w, center=True, return_complex=True).abs()
            Y = torch.stft(y, nf, hp, nf, w, center=True, return_complex=True).abs()
            m = min(T.shape[-1], Y.shape[-1])
            T, Y = T[:, :m], Y[:, :m]
            lvl = 20 * torch.log10((T / T.max().clamp(min=1e-12)).clamp(min=1e-15))
            l1 = (torch.log(Y + 1e-5) - torch.log(T + 1e-5)).abs()
            # |d/dY| of the log-L1 term. The 1/(Y+eps) factor is what blows the
            # bottom of the range up: a bin at -80 dB has a derivative ~1e4
            # times larger than one at -0 dB.
            g = 1.0 / (Y + 1e-5)
            for b in range(len(EDGES) - 1):
                sel = (lvl <= EDGES[b]) & (lvl > EDGES[b + 1])
                if sel.any():
                    mass[b] += float(l1[sel].sum())
                    grad[b] += float((l1[sel] * g[sel]).sum())
                    count[b] += int(sel.sum())
    tot_m, tot_g, tot_c = mass.sum(), grad.sum(), count.sum()
    print(f"\n  {len(te)} held-out speakers, 3 resolutions, prior = DSP core.")
    print("  Bins bucketed by TARGET level below that utterance's peak.\n")
    print(f"  {'band of the target':>22} {'% of bins':>10} {'% of loss':>11} "
          f"{'% of |grad|':>12}")
    for b in range(len(EDGES) - 1):
        lo = f"{EDGES[b]}" if b else "  0"
        hi = "-inf" if EDGES[b + 1] <= -300 else f"{EDGES[b+1]}"
        print(f"  {lo:>8} .. {hi:>8} dB {100*count[b]/tot_c:9.2f}% "
              f"{100*mass[b]/tot_m:10.2f}% {100*grad[b]/tot_g:11.2f}%")
    print("\n  If most of the loss and its gradient sit below -60 dB, the objective")
    print("  is spending itself on bins no listener can hear, and the presence")
    print("  band the ear named competes with them on equal footing.")


if __name__ == "__main__":
    main()
