"""Task B from docs/paper_benchmark_degeneracy.md, posed AT RATE.

The paper lists "give the least-squares fit an intra-window linear term" as the
one untested way to raise the harmonic branch. Posed naively it doubles the
parameters (4 reals per harmonic per period instead of 2), which is exactly the
2x-overcomplete regime the paper rejects ENV_RATE=2 for. So pose it at rate:

    HALF the nodes (one per TWO periods), each carrying {value, slope}
    = 4 reals per harmonic per 2 periods
    = 2 reals per harmonic per period
    = the shipping rate, exactly.

The question is then a real one about the basis rather than the budget: for a
fixed number of reals per second, is a sparse grid of values-and-slopes a better
description of a harmonic's complex envelope than a dense grid of values?

Both arms are rendered by the same synthesiser and scored on the same
utterances; only the parameterisation differs.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from kansei_train import CACHE, DATA
from rddsp_loop import score_one


def ls_slope(x: torch.Tensor, f0: torch.Tensor, pos: torch.Tensor, fmax: float,
             dense: torch.Tensor, kmax: int = 400) -> torch.Tensor:
    """Least squares with basis {cos, sin, n*cos, n*sin} at `pos`, evaluated on
    `dense`. Returns harmonic_analysis_at()'s convention on the dense grid, so
    the renderer is untouched -- what changed is where the information came
    from, not how many samples the oscillator gets."""
    n = x.shape[-1]
    Phi, _ = R.phase_track(f0, n)
    fm = R.median_f0(f0)
    L = int(R.LS_PERIODS * R.SR / fm) | 1
    half = L // 2
    idx = pos.clamp(0, n - 1).long()
    f0_at = f0[(idx // R.HOP).clamp(0, len(f0) - 1)]
    f0_at = torch.where(f0_at > 50, f0_at, torch.full_like(f0_at, fm))
    w = torch.hann_window(L, periodic=False, dtype=torch.float64)
    off = torch.arange(-half, half + 1)
    tt = off.double() / half                       # normalised window position
    xd = x.double()
    out = torch.zeros(kmax, len(dense), dtype=torch.complex64)

    didx = dense.clamp(0, n - 1).long()
    j = torch.searchsorted(idx.double(), didx.double()).clamp(1, len(idx) - 1)
    span = (idx[j] - idx[j - 1]).double().clamp(min=1.0)
    # CUBIC HERMITE between the two bracketing nodes, using both values and both
    # slopes. Evaluating the local linear model around the NEAREST node instead
    # is extrapolation, and it scored below plain interpolation of the same
    # nodes -- a property of that shortcut, not of the basis.
    tfrac = ((didx - idx[j - 1]).double() / span).clamp(0, 1)
    t2, t3 = tfrac * tfrac, tfrac * tfrac * tfrac
    H00 = 2 * t3 - 3 * t2 + 1
    H10 = t3 - 2 * t2 + tfrac
    H01 = -2 * t3 + 3 * t2
    H11 = t3 - t2
    # slopes come out of the fit in units of the normalised window (half samples)
    sc = span / half

    for s in range(0, len(idx), R.LS_CHUNK):
        sl = slice(s, s + R.LS_CHUNK)
        fl = f0_at[sl].double().clamp(min=50.0)[:, None]
        K = max(1, min(kmax, int(min(fmax, R.SR / 2 - R.SR / R.NFFT) / float(fl.min()))))
        ks = torch.arange(1, K + 1, dtype=torch.float64)
        ii = (idx[sl][:, None] + off[None, :]).clamp(0, n - 1)
        ph = Phi[ii][:, :, None] * ks[None, None, :]
        c, sn = torch.cos(ph), -torch.sin(ph)
        t = tt[None, :, None]
        A = torch.cat([c, sn, c * t, sn * t], dim=-1)
        live = (ks[None, :] * fl < min(fmax, R.SR / 2 - R.SR / R.NFFT)).double()
        A = A * torch.cat([live] * 4, dim=-1)[:, None, :]
        Aw = A * w[None, :, None]
        G = Aw.transpose(1, 2) @ A
        r = Aw.transpose(1, 2) @ xd[ii][:, :, None]
        d = torch.diagonal(G, dim1=1, dim2=2)
        d = (d.sum(-1) / live.sum(-1).clamp(min=1.0) / 4.0)[:, None, None]
        G = G + R.LS_RIDGE * d * torch.eye(4 * K, dtype=torch.float64)[None]
        th = torch.linalg.solve(G, r)[:, :, 0]
        c0 = torch.complex(th[:, :K], th[:, K:2 * K])          # value
        c1 = torch.complex(th[:, 2 * K:3 * K], th[:, 3 * K:])  # slope
        # a dense point is handled here when BOTH bracketing nodes are in chunk
        lo_i, hi_i = j - 1, j
        m = (lo_i >= s) & (hi_i < s + len(fl))
        if not bool(m.any()):
            continue
        a_, b_ = lo_i[m] - s, hi_i[m] - s
        env = (H00[m][:, None] * c0[a_] + H01[m][:, None] * c0[b_]
               + (H10[m] * sc[m])[:, None] * c1[a_]
               + (H11[m] * sc[m])[:, None] * c1[b_])
        car = torch.exp(1j * (Phi[didx[m]][:, None] * ks[None, :]))
        out[:K, m] = (env * car).T.to(torch.complex64)
    return out


def main() -> None:
    uids = [p.stem for p in sorted(CACHE.glob("*.npz"))]
    ids = uids[40:52]                     # held out from every tuning run
    rows: dict[str, list[float]] = {}
    for uid in ids:
        x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
        gt = torch.tensor(x[: R.SR * 6])
        n = gt.shape[-1]
        y, h, nz, p = R.resynthesize(gt)
        rows.setdefault("shipping (value @ 1/period)", []).append(score_one(gt, y))

        f0 = p["f0"]
        gci = p["gci"]
        fmax = 2.5 * float(p["mvf"].max())
        half = gci[::2]
        if len(half) < 4:
            continue
        Xg = ls_slope(gt, f0, half, fmax, gci)
        kw = dict(pos=gci)
        y2, _, _ = R.synthesize_v2(f0, p["mvf"], p["apbins"], p["noisemag"], n,
                                   gci, Xg, f0[(gci // R.HOP).clamp(0, len(f0) - 1)],
                                   noisefine=p.get("noisefine"), **kw)
        rows.setdefault("value+slope @ 1/2periods", []).append(score_one(gt, y2))
        # control: same halved node grid, value only -- isolates "half the nodes"
        Xc = R.harmonic_ls(gt, f0, half, fmax)
        y3, _, _ = R.synthesize_v2(f0, p["mvf"], p["apbins"], p["noisemag"], n,
                                   half, Xc, f0[(half // R.HOP).clamp(0, len(f0) - 1)],
                                   noisefine=p.get("noisefine"), pos=half)
        rows.setdefault("value only @ 1/2periods", []).append(score_one(gt, y3))

    print(f"\n{'parameterisation':30s} {'PESQ':>7s}  {'reals/harmonic/period':>22s}")
    ref = {"shipping (value @ 1/period)": "2", "value+slope @ 1/2periods": "2 (same)",
           "value only @ 1/2periods": "1 (half)"}
    for k, v in rows.items():
        print(f"{k:30s} {np.mean(v):7.4f}  {ref.get(k,''):>22s}")


if __name__ == "__main__":
    main()
