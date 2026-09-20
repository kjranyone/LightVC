"""Which half of the remaining gap is MAGNITUDE and which is PHASE?

The capacity sweep just refuted the attribution in 12.9''': at the identical
schedule and data, 4x the parameters made it WORSE on both sets

    ch 48  0.501M  loss 1.911  TRAIN +0.119  TEST +0.236
    ch 96  1.997M  loss 1.842  TRAIN +0.115  TEST +0.224

and the run-to-run spread of ch 48 is 0.0014 (two independent inits: +0.2372,
+0.2358), so -0.012 is ten sigma. More capacity lowers the LOSS and lowers the
PESQ. Two systems cannot both be capacity-limited when the bigger one fits its
own training data no better (+0.115 vs +0.119). So the limiter is the objective
or the parameterisation, not the parameter count.

This probe decides which, WITHOUT training anything. Take the DSP prior y and
the target x, transform both with the harness's own STFT (512/128, center), and
cross them:

    swap M   |X| with angle(Y)   -- perfect magnitude, our phase
    swap P   |Y| with angle(X)   -- our magnitude, perfect phase

Then iSTFT. Both crossed spectrograms are INCONSISTENT, so what comes back is
the overlap-add projection of them -- which is exactly the situation of a
TF-domain net, whose additive complex residual is projected the same way. The
numbers are therefore reachable-in-principle ceilings for this interface, not
abstractions.

Reading:
  swap M high, swap P low  -> the deficit is magnitude; a magnitude loss can
                              still steer this, and mrstft is the right tool.
  swap M low, swap P high  -> the deficit is phase; mrstft is BLIND to it and
                              no amount of capacity trained under it can close
                              the gap, which is precisely the ch48/ch96 result.

id is the same STFT/iSTFT round trip on x alone: the harness's own ceiling, so
that a shortfall in the swaps is not confused with a shortfall in the transform.
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
from rddsp_dspecies_multi import pick

NFFT, HOP = 512, 128
SECONDS = 5


def stft(x: torch.Tensor) -> torch.Tensor:
    return torch.stft(x, NFFT, HOP, NFFT, torch.hann_window(NFFT),
                      center=True, return_complex=True)


def istft(S: torch.Tensor, n: int) -> torch.Tensor:
    return torch.istft(S, NFFT, HOP, NFFT, torch.hann_window(NFFT),
                       center=True, length=n)


def main() -> None:
    _, te_p = pick(80, 12)
    rows: dict[str, list[float]] = {k: [] for k in
                                    ("dsp", "swapM", "swapP", "id", "magonly_gain")}
    for p in te_p:
        x, _ = librosa.load(str(p), sr=R.SR, mono=True)
        if len(x) < R.SR * 2:
            continue
        gt = torch.tensor(x[: R.SR * SECONDS])
        try:
            y, _, _, _ = R.resynthesize(gt)
        except Exception:
            continue
        n = min(gt.shape[-1], y.shape[-1])
        gt, y = gt[:n], y[:n]
        X, Y = stft(gt), stft(y)
        T = min(X.shape[-1], Y.shape[-1])
        X, Y = X[:, :T], Y[:, :T]

        eps = 1e-8
        M = X.abs() * (Y / (Y.abs() + eps))       # gt magnitude, our phase
        P = Y.abs() * (X / (X.abs() + eps))       # our magnitude, gt phase

        rows["dsp"].append(score_one(gt, y))
        rows["swapM"].append(score_one(gt, istft(M, n)))
        rows["swapP"].append(score_one(gt, istft(P, n)))
        rows["id"].append(score_one(gt, istft(X, n)))
        # NOT an independent measurement -- res = M - Y, so Y + res IS M, and
        # this row is identically swapM. It is here only as a numerical check
        # that routing the same spectrum through the net's additive-residual
        # interface changes nothing. If it ever differs from swapM, the residual
        # path has a bug. Do not quote it as evidence for anything else.
        res = M - Y
        rows["magonly_gain"].append(score_one(gt, istft(Y + res, n)))
        print(f"  {p.parent.name[:14]:16s} dsp {rows['dsp'][-1]:.3f}  "
              f"swapM {rows['swapM'][-1]:.3f}  swapP {rows['swapP'][-1]:.3f}",
              flush=True)

    print("\n  ---- mean over %d held-out speakers ----" % len(rows["dsp"]))
    for k in ("dsp", "swapM", "swapP", "id", "magonly_gain"):
        print(f"  {k:14s} {np.mean(rows[k]):.4f}")
    dsp, sm, sp, idn = (np.mean(rows[k]) for k in ("dsp", "swapM", "swapP", "id"))
    print(f"\n  magnitude head-room  {sm - dsp:+.3f}   (perfect |X|, our phase)")
    print(f"  phase     head-room  {sp - dsp:+.3f}   (our |Y|, perfect phase)")
    print(f"  transform ceiling    {idn:.3f}")
    print("\n  A magnitude-only objective can reach at most swapM. If swapM is")
    print("  below the target, capacity trained under mrstft cannot get there,")
    print("  and the ch48/ch96 inversion is explained without invoking capacity.")


if __name__ == "__main__":
    main()
