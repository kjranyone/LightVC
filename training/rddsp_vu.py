"""Voiced vs unvoiced split of the sub-1 kHz magnitude error.

In voiced frames the harmonic peaks below 1 kHz are accurate to -0.1..-0.3 dB
and ap is 0.003..0.027, so almost nothing down there is stochastic. Yet swapping
in gt's magnitude below 1 kHz is worth +0.667 PESQ. The remaining suspect is the
35-42% of frames that are UNVOICED, where UNVOICED_FULL hands the entire
spectrum to the phase-blind breath branch and the magnitude that survives
overlap-add is only correct in expectation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import Cache, score_one


def band_mask(c: float, T: int) -> torch.Tensor:
    fb = torch.arange(R.NB, dtype=torch.float32) * R.SR / R.NFFT
    return torch.sigmoid((c - fb) / (0.05 * max(c, 1.0)))[:, None].expand(R.NB, T)


def main() -> None:
    c = Cache()
    rows: dict[str, list[float]] = {}

    for uid, gt in c.items:
        n = gt.shape[-1]
        y, h, nz, p = R.resynthesize(gt)
        rows.setdefault("baseline", []).append(score_one(gt, y))
        X, Y = R.stft(gt), R.stft(y)
        T = min(X.shape[-1], Y.shape[-1], p["f0"].shape[-1])
        X, Y = X[:, :T], Y[:, :T]
        v = (p["f0"][:T] > 50).float()[None]
        xm, xp = X.abs(), X / (X.abs() + R.EPS)
        ym, yp = Y.abs(), Y / (Y.abs() + R.EPS)

        for cut in (1000.0, 22050.0):
            bm = band_mask(cut, T)
            for tag, fm in (("V", v), ("U", 1 - v), ("all", torch.ones_like(v))):
                m = bm * fm
                Z = m * (ym * xp) + (1 - m) * Y
                rows.setdefault(f"<{cut:.0f} {tag} PHASE", []).append(score_one(gt, R.istft(Z, n)))
                Z = m * (xm * yp) + (1 - m) * Y
                rows.setdefault(f"<{cut:.0f} {tag} MAG", []).append(score_one(gt, R.istft(Z, n)))
                Z = m * X + (1 - m) * Y
                rows.setdefault(f"<{cut:.0f} {tag} BOTH", []).append(score_one(gt, R.istft(Z, n)))

    print(f"\n{'variant':22s} {'PESQ':>7s} {'delta':>7s}")
    b = sum(rows["baseline"]) / len(rows["baseline"])
    for k, val in rows.items():
        m = sum(val) / len(val)
        print(f"{k:22s} {m:7.4f} {m-b:+7.4f}")


if __name__ == "__main__":
    main()
