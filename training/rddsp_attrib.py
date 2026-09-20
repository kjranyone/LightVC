"""Where does the PESQ actually go? Band-wise oracle substitution.

Blending our output with gt in the STFT domain under complementary masks is
exact (the grid is COLA, so istft(stft(x)) == x), so "gt below c" measures the
PESQ recovered by making everything under c perfect, and "gt above c" the same
for the top. That attributes the 3.406 -> 4.644 gap to a band instead of to a
branch, which is the question the residual experiment could not answer.

Also reports how much of the signal the harmonic bank actually cancels, per
band, restricted to VOICED frames -- h-only PESQ is dominated by the 38% of
frames where h is silent by construction and says nothing about its accuracy.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import Cache, score_one

EDGES = [0, 500, 1000, 2000, 4000, 8000, 22050]


def mask_below(c: float, T: int) -> torch.Tensor:
    fb = torch.arange(R.NB, dtype=torch.float32) * R.SR / R.NFFT
    return (torch.sigmoid((c - fb) / (0.05 * max(c, 1.0)))[:, None]).expand(R.NB, T)


def blend(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    X, Y = R.stft(x), R.stft(y)
    T = min(X.shape[-1], Y.shape[-1], m.shape[-1])
    return R.istft(m[:, :T] * X[:, :T] + (1 - m[:, :T]) * Y[:, :T], n)


def main() -> None:
    c = Cache()
    rows: dict[str, list[float]] = {}
    cancel: list[list[float]] = []

    for uid, gt in c.items:
        n = gt.shape[-1]
        y, h, nz, p = R.resynthesize(gt)
        rows.setdefault("baseline", []).append(score_one(gt, y))
        T = R.stft(gt).shape[-1]

        for cut in (500.0, 1000.0, 2000.0, 4000.0, 8000.0):
            m = mask_below(cut, T)
            rows.setdefault(f"gt below {cut:.0f}", []).append(score_one(gt, blend(gt, y, m)))
            rows.setdefault(f"gt above {cut:.0f}", []).append(score_one(gt, blend(gt, y, 1 - m)))

        # how much does the harmonic bank cancel, in VOICED frames only?
        f0 = p["f0"]
        Xg_ = R.stft(gt)
        Rg = R.stft(gt - h)
        Tn = min(Xg_.shape[-1], Rg.shape[-1], f0.shape[-1])
        v = (f0[:Tn] > 50)
        binhz = R.SR / R.NFFT
        row = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            i, j = int(lo / binhz), int(hi / binhz)
            ea = (Xg_[i:j, :Tn].abs() ** 2)[:, v].sum()
            eb = (Rg[i:j, :Tn].abs() ** 2)[:, v].sum()
            row.append(10 * math.log10(float(eb + 1e-12) / float(ea + 1e-12)))
        cancel.append(row)
        print(f"[{uid}] synth MVF med {float(p['mvf'].median()):.0f}Hz  "
              f"voiced {100*float(v.float().mean()):.0f}%  cyc {p['cyc']}")

    print(f"\n{'variant':20s} {'PESQ':>7s} {'delta':>7s}")
    b = sum(rows["baseline"]) / len(rows["baseline"])
    for k, v in rows.items():
        m = sum(v) / len(v)
        print(f"{k:20s} {m:7.4f} {m-b:+7.4f}")

    print("\nharmonic-bank cancellation in voiced frames, dB (lower = h explains more)")
    print("  " + "  ".join(f"{lo/1000:g}-{hi/1000:g}k" for lo, hi in zip(EDGES[:-1], EDGES[1:])))
    avg = [sum(r[i] for r in cancel) / len(cancel) for i in range(len(EDGES) - 1)]
    print("  " + "  ".join(f"{d:+6.1f}" for d in avg))


if __name__ == "__main__":
    main()
