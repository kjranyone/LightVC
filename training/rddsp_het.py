"""Heterodyne each low harmonic out of gt and out of our render and compare.

The STFT view cannot answer this: a 2048-sample frame is 46 ms, twelve glottal
cycles, so a bin magnitude is already an average. Demodulating harmonic k by
exp(-j k Phi) -- the SAME phase track synthesis uses -- and low-passing at f0/2
gives its complex envelope per sample, which is exactly the quantity the GCI
grid samples and linearly interpolates. If the sampling is what loses the
magnitude, it shows up here as envelope error and nowhere else.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import Cache


def demod(x: torch.Tensor, Phi: torch.Tensor, k: int, L: int) -> torch.Tensor:
    z = x.double() * torch.exp(-1j * k * Phi)
    w = torch.hann_window(L, dtype=torch.float64)
    w = w / w.sum()
    zr = torch.nn.functional.conv1d(z.real[None, None], w[None, None].flip(-1), padding=L // 2)[0, 0]
    zi = torch.nn.functional.conv1d(z.imag[None, None], w[None, None].flip(-1), padding=L // 2)[0, 0]
    return (zr + 1j * zi)[: x.shape[-1]]


def main() -> None:
    c = Cache()
    for uid, gt in c.items:
        n = gt.shape[-1]
        y, h, nz, p = R.resynthesize(gt)
        f0 = p["f0"]
        f0u = R.frame_upsample(f0.double(), n).clamp(min=0.0)
        Phi = 2 * math.pi * torch.cumsum(f0u, dim=-1) / R.SR
        fm = float(f0[f0 > 50].median())
        L = int(R.SR / fm) | 1
        v = f0u > 50

        print(f"\n[{uid}] f0 {fm:.0f}Hz  window {L} samples ({1000*L/R.SR:.1f} ms)")
        print(f"{'k':>3s} {'f_k':>6s} {'|A| corr':>9s} {'mean dB':>8s} {'std dB':>7s} "
              f"{'ph rms deg':>11s} {'h only dB':>10s} {'h std':>7s}")
        for k in range(1, int(2000 / fm) + 2):
            if k * fm > 2200:
                break
            ax = demod(gt, Phi, k, L).abs()
            ay = demod(y, Phi, k, L).abs()
            ah = demod(h, Phi, k, L).abs()
            px = torch.angle(demod(gt, Phi, k, L))
            py = torch.angle(demod(y, Phi, k, L))
            m = v & (ax > 0.02 * float(ax[v].max()))
            if m.sum() < 100:
                continue
            dx, dy, dh = ax[m], ay[m], ah[m]
            edb = 20 * torch.log10((dy + 1e-12) / (dx + 1e-12))
            ehb = 20 * torch.log10((dh + 1e-12) / (dx + 1e-12))
            cc = float(torch.corrcoef(torch.stack([dx, dy]))[0, 1])
            dph = torch.remainder(py[m] - px[m] + math.pi, 2 * math.pi) - math.pi
            print(f"{k:3d} {k*fm:6.0f} {cc:9.4f} {float(edb.mean()):8.2f} "
                  f"{float(edb.std()):7.2f} {math.degrees(float((dph**2).mean().sqrt())):11.1f} "
                  f"{float(ehb.mean()):10.2f} {float(ehb.std()):7.2f}")


if __name__ == "__main__":
    main()
