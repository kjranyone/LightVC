"""Is the "neural refinement" just one static EQ curve?

rddsp_mismatch.py fed the trained net the mel of a DIFFERENT speaker and 88.8%
of its +0.234 survived. So the mel channel carries almost none of the gain, and
the null hypothesis 6.4 exists to test is live: the net is a fixed post-filter on
the prior, not a conditioned mapping.

The cheapest possible refutation of the net is to build that post-filter by hand
and see whether it does the same job. Fit ONE frequency response on the training
speakers,

    g(f) = mean_train mean_t |X(f,t)|  /  mean_train mean_t |Y(f,t)|

apply it to the held-out priors as S' = g(f) * Y, and score. No time dependence,
no conditioning, no parameters beyond 257 numbers -- and it is fitted on the same
80 training utterances the net saw, so it is not cheating on the test set.

If g(f) alone reproduces the gain, the 0.501M net and its 40 minutes of GPU are
not buying anything a 257-tap curve does not, and the honest description of
12.9's "+0.237 neural gain" is "the DSP core has a systematic spectral tilt".

Three variants, in increasing strength, to locate where the net's contribution
actually lives:
  eq_global   one curve for the whole corpus                  (no conditioning)
  eq_voiced   one curve, fitted on the energy-weighted mean   (same, better fit)
  eq_oracle   one curve PER TEST UTTERANCE, fitted on itself  (upper bound of
              any time-invariant filter, unreachable in deployment)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from rddsp_loop import score_one
from rddsp_gpu import build, stft, istft, prior_wave


def prior_of(it, pn: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    return prior_wave(dict(y=it["y"]), g, "cpu", pn)


def curve(items, weighted: bool, pn: float = 0.0) -> torch.Tensor:
    num = torch.zeros(stft(items[0]["gt"]).shape[0], dtype=torch.float64)
    den = torch.zeros_like(num)
    for i, it in enumerate(items):
        n = min(it["gt"].shape[-1], it["y"].shape[-1])
        X = stft(it["gt"][:n]).abs().double()
        Y = stft(prior_of(it, pn, 1000 + i)[:n]).abs().double()
        T = min(X.shape[-1], Y.shape[-1])
        X, Y = X[:, :T], Y[:, :T]
        if weighted:
            # Energy weighting stops silent frames, where both spectra are
            # near zero, from dominating a ratio of means.
            w = (Y ** 2).sum(0, keepdim=True)
            num += (X * w).sum(-1)
            den += (Y * w).sum(-1)
        else:
            num += X.sum(-1)
            den += Y.sum(-1)
    return (num / den.clamp(min=1e-12)).float()


def apply_curve(it, g: torch.Tensor, pn: float = 0.0, seed: int = 999):
    n = min(it["gt"].shape[-1], it["y"].shape[-1])
    Y = stft(prior_of(it, pn, seed)[:n])
    return istft(Y * g[:, None], n)


def main() -> None:
    tr, te = build(80, 12)
    print(f"  train {len(tr)}  test {len(te)}\n", flush=True)

    PN = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    print(f"  prior noise {PN}  (0 = clean DSP core, 0.05 = what arm C was fed)\n",
          flush=True)
    g_glob = curve(tr, weighted=False, pn=PN)
    g_wgt = curve(tr, weighted=True, pn=PN)
    print(f"  fitted curve range: global {g_glob.min():.3f}..{g_glob.max():.3f}  "
          f"weighted {g_wgt.min():.3f}..{g_wgt.max():.3f}", flush=True)

    rows = {k: [] for k in ("prior", "eq_global", "eq_voiced", "eq_oracle")}
    for it in te:
        n = min(it["gt"].shape[-1], it["y"].shape[-1])
        gt = it["gt"][:n]
        rows["prior"].append(score_one(gt, prior_of(it, PN, 999)[:n]))
        rows["eq_global"].append(score_one(gt, apply_curve(it, g_glob, PN)))
        rows["eq_voiced"].append(score_one(gt, apply_curve(it, g_wgt, PN)))
        rows["eq_oracle"].append(score_one(gt, apply_curve(it, curve([it], True, PN), PN)))
        print(f"  prior {rows['prior'][-1]:.3f}  glob {rows['eq_global'][-1]:.3f}  "
              f"vced {rows['eq_voiced'][-1]:.3f}  oracle {rows['eq_oracle'][-1]:.3f}",
              flush=True)

    pr = np.array(rows["prior"])
    print(f"\n  ---- {len(pr)} unseen speakers, prior noise {PN} ----")
    print(f"  prior (no filter)  {pr.mean():.4f}")
    for k in ("eq_global", "eq_voiced", "eq_oracle"):
        v = np.array(rows[k])
        d = v - pr
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {k:11s}        {v.mean():.4f}   gain {d.mean():+.4f} +-{ci:.4f}")
    print("\n  Compare with the 0.501M net's gain over ITS prior. If a 257-number")
    print("  time-invariant curve, fitted without any conditioning, lands in the")
    print("  same place, the net's contribution is that curve.")


if __name__ == "__main__":
    main()
