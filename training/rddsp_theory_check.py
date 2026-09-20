"""Numerical verification of Theorem 1 (docs/paper_benchmark_degeneracy.md 2.2).

The theorem says the demodulate / low-pass / remodulate / sum composition is LTI
with transfer function G(nu) = Sum_k W(nu - k*nu0), and that G is the CONSTANT
P*w[0] whenever the window is shorter than two pitch periods.

Two things are checked, both against the closed form rather than against
intuition:

  1. the composition's measured transfer function -- obtained by driving it with
     white noise, which has no harmonic structure at all, so any output is by
     definition not explained by a harmonic model;
  2. the predicted constant P*w[0], and the Hann corollary G = 2/EP.

If the measured gain matches P*w[0] and the ripple is at numerical zero for
EP <= 2, Theorem 1 is verified on the implementation and not merely asserted.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R

SR = R.SR
F0 = 220.0
P = SR / F0


def compose(x: torch.Tensor, ep: float, kmax: int) -> torch.Tensor:
    """Per-sample envelopes: the r -> infinity limit the theorem is about."""
    n = x.shape[-1]
    L = int(ep * P) | 1
    w = torch.hann_window(L, periodic=False, dtype=torch.float64)
    w = w / w.sum()
    phi = 2 * math.pi * F0 * torch.arange(n, dtype=torch.float64) / SR
    y = torch.zeros(n, dtype=torch.float64)
    for k in range(1, kmax + 1):
        car = torch.exp(1j * k * phi)
        a = 2.0 * R._lp_centred(x.double() * car.conj(), w)
        y = y + (a * car).real
    return y


def main() -> None:
    n = SR * 2
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, generator=g) * 0.2          # NO harmonic structure
    kmax = int((SR / 2 - F0) / F0)
    print(f"f0 {F0:.0f} Hz  P {P:.1f} samples  harmonics {kmax}  "
          f"(full band, so the composition should be the identity when L < 2P)\n")
    print(f"{'EP':>5} {'L':>6} {'pred P*w[0]':>12} {'measured':>10} "
          f"{'ripple dB':>10} {'SNR vs x dB':>12}")
    for ep in (1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0):
        L = int(ep * P) | 1
        w = torch.hann_window(L, periodic=False, dtype=torch.float64)
        w = w / w.sum()
        pred = float(P * w[L // 2])
        y = compose(x, ep, kmax)
        m = slice(4 * L, n - 4 * L)                 # drop the filter edges
        # Welch transfer-function estimate. The raw per-bin ratio Y/X is not an
        # estimator: with white noise |X| is Rayleigh, so the ratio blows up
        # wherever the excitation happens to be small, and its max/min says more
        # about that than about G.
        nf, hop = 4096, 1024
        wn = torch.hann_window(nf, dtype=torch.float64)
        Xs = torch.stft(x.double()[m], nf, hop, nf, wn, center=False, return_complex=True)
        Ys = torch.stft(y[m], nf, hop, nf, wn, center=False, return_complex=True)
        Sxy = (Ys * Xs.conj()).mean(-1)
        Sxx = (Xs.abs() ** 2).mean(-1)
        Hs = Sxy / (Sxx + 1e-20)
        fb = torch.fft.rfftfreq(nf, 1 / SR)
        keep = (fb > 2 * F0) & (fb < SR / 2 - 2 * F0)
        mag = Hs[keep].abs()
        meas = float(mag.mean())
        ripple = 20 * math.log10(float(mag.max() / mag.min().clamp(min=1e-12)))
        r = x.double()[m]
        s = y[m] / max(meas, 1e-9)
        snr = 10 * math.log10(float((r ** 2).sum() / ((s - r) ** 2).sum().clamp(min=1e-20)))
        print(f"{ep:5.2f} {L:6d} {pred:12.4f} {meas:10.4f} {ripple:10.3f} {snr:12.2f}")
    print("\nTheorem 1: G = P*w[0] exactly when the window support is inside +-P,")
    print("i.e. EP <= 2. Above that the m != 0 Poisson terms appear and G ripples.")
    print("A composition that returns WHITE NOISE at high SNR is not a harmonic model.")


if __name__ == "__main__":
    main()
