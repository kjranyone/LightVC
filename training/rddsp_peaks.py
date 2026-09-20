"""Is the sub-1 kHz magnitude error at the harmonic PEAKS or in the VALLEYS?

The harmonic bank is driven by measured amplitudes, so a level error at the
peaks would have to be systematic. The prime suspect is the aperiodicity weight:
synthesize_v2 scales every harmonic by (1 - ap), but below MVF the breath branch
only renders the min-pooled inter-harmonic FLOOR -- it never puts the removed
ap fraction back. If so the peaks are low by exactly 20*log10(1-ap) and the
energy is simply lost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import Cache


def db(a: torch.Tensor, b: torch.Tensor) -> float:
    return 10 * torch.log10((a + 1e-20) / (b + 1e-20)).item()


def main() -> None:
    c = Cache()
    for uid, gt in c.items:
        n = gt.shape[-1]
        y, h, nz, p = R.resynthesize(gt)
        X, Y, H = R.stft(gt), R.stft(y), R.stft(h)
        T = min(X.shape[-1], Y.shape[-1], p["f0"].shape[-1])
        X, Y, H = X[:, :T], Y[:, :T], H[:, :T]
        f0 = p["f0"][:T]
        v = f0 > 50
        binhz = R.SR / R.NFFT
        ap = p["apbins"][:, :T]

        print(f"\n[{uid}] voiced {100*float(v.float().mean()):.0f}%  "
              f"f0 {float(f0[v].median()):.0f}Hz  ap(0-3k) med {float(ap[0][v].median()):.3f}  "
              f"mvf {float(p['mvf'].median()):.0f}Hz")
        print(f"{'band':>10s} {'peak dB':>9s} {'valley dB':>10s} {'all dB':>8s} "
              f"{'h/gt peak':>10s} {'1-ap':>7s}")

        for lo, hi in ((0, 500), (500, 1000), (1000, 2000)):
            pk = torch.zeros(R.NB, T, dtype=torch.bool)
            for t in range(T):
                if not v[t]:
                    continue
                f = float(f0[t])
                for k in range(1, int((hi) / f) + 1):
                    fk = k * f
                    if fk < lo or fk >= hi:
                        continue
                    b = int(round(fk / binhz))
                    pk[max(0, b - 1): b + 2, t] = True
            i, j = int(lo / binhz), int(hi / binhz)
            bandv = torch.zeros(R.NB, T, dtype=torch.bool)
            bandv[i:j, :] = v[None, :]
            mpk, mvl = pk & bandv, (~pk) & bandv
            ex, ey = (X.abs() ** 2), (Y.abs() ** 2)
            eh = H.abs() ** 2
            apk = float(ap[0][v].median())
            print(f"{lo:5d}-{hi:4d} {db(ey[mpk].sum(), ex[mpk].sum()):9.2f} "
                  f"{db(ey[mvl].sum(), ex[mvl].sum()):10.2f} "
                  f"{db(ey[bandv].sum(), ex[bandv].sum()):8.2f} "
                  f"{db(eh[mpk].sum(), ex[mpk].sum()):10.2f} "
                  f"{20*torch.log10(torch.tensor(1-apk)).item():7.2f}")


if __name__ == "__main__":
    main()
