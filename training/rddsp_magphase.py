"""Split the sub-2 kHz loss into MAGNITUDE error and PHASE error.

rddsp_attrib.py put the whole remaining gap under 2 kHz, and lowering the
inter-harmonic noise floor there only made it worse, so the floor is carrying
real content. This asks the next question directly: inside a band, keep our
magnitude and take gt's phase (and vice versa). Whichever substitution recovers
the gap is the one that is broken.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import Cache, score_one


def lag_of(a: torch.Tensor, b: torch.Tensor) -> int:
    x = a.numpy().astype(np.float64)
    y = b.numpy().astype(np.float64)
    n = min(len(x), len(y))
    return int(np.argmax(np.abs(correlate(y[:n], x[:n], mode="full")))) - (n - 1)


def band_mask(c: float, T: int) -> torch.Tensor:
    fb = torch.arange(R.NB, dtype=torch.float32) * R.SR / R.NFFT
    return torch.sigmoid((c - fb) / (0.05 * max(c, 1.0)))[:, None].expand(R.NB, T)


def main() -> None:
    c = Cache()
    rows: dict[str, list[float]] = {}

    for uid, gt in c.items:
        n = gt.shape[-1]
        y, h, nz, p = R.resynthesize(gt)
        lg = lag_of(gt, y)
        print(f"[{uid}] lag {lg} samples ({1000*lg/R.SR:+.2f} ms)")
        rows.setdefault("baseline", []).append(score_one(gt, y))

        X, Y = R.stft(gt), R.stft(y)
        T = min(X.shape[-1], Y.shape[-1])
        X, Y = X[:, :T], Y[:, :T]
        xm, xp = X.abs(), X / (X.abs() + R.EPS)
        ym, yp = Y.abs(), Y / (Y.abs() + R.EPS)

        for cut in (500.0, 1000.0, 2000.0, 22050.0):
            m = band_mask(cut, T)
            # inside the band take gt's phase, keep our magnitude
            Z = m * (ym * xp) + (1 - m) * Y
            rows.setdefault(f"<{cut:.0f} gt PHASE", []).append(score_one(gt, R.istft(Z, n)))
            # inside the band take gt's magnitude, keep our phase
            Z = m * (xm * yp) + (1 - m) * Y
            rows.setdefault(f"<{cut:.0f} gt MAG", []).append(score_one(gt, R.istft(Z, n)))

    print(f"\n{'variant':22s} {'PESQ':>7s} {'delta':>7s}")
    b = sum(rows["baseline"]) / len(rows["baseline"])
    for k, v in rows.items():
        m = sum(v) / len(v)
        print(f"{k:22s} {m:7.4f} {m-b:+7.4f}")


if __name__ == "__main__":
    main()
