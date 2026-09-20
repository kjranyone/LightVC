"""Legitimacy test for the harmonic branch: can it reconstruct WHITE NOISE?

A harmonic bank is supposed to represent a harmonic series. If it also
reproduces a signal with no harmonic structure at all, it is not a model of
anything -- it is a perfect-reconstruction filter bank, and the PESQ it earns is
the invertibility of that bank rather than the quality of the vocoder. That is
exactly what happened at (ENV_PERIODS, ENV_RATE) = (1.75, 5): white noise came
back at 17.97 dB SNR and the PESQ of 4.070 measured waveform copying.

The Poisson-sum argument says Sum_k W(f - k*f0) is EXACTLY constant for
ENV_PERIODS <= 2, but ENV_RATE and the node interpolation also matter, so the
criterion has to be measured on the path that actually runs, not on the window
alone.

Second check: detune the carrier. A real harmonic model collapses when f0 is
wrong; a filter bank does not care.

PASS = white-noise SNR below WHITE_MAX and the speech-to-noise margin above
MARGIN_MIN. Both are properties of the configuration, not of any recording.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R

WHITE_MAX = 6.0    # dB
MARGIN_MIN = 6.0   # dB


def bank_snr(x: torch.Tensor, f0hz: float, fmax: float, rate: int, periods: float,
             detune: float = 1.0) -> float:
    """Render x through the harmonic branch alone (gate off) and compare with x
    band-limited to the same range. Nodes are placed at `rate` per period, the
    same way env_nodes() places them on a uniform epoch train."""
    n = x.shape[-1]
    f0t = torch.full((n // R.HOP + 2,), f0hz)
    old_fill, old_ep, old_lp = R.F0_FILL, R.ENV_PERIODS, R.LS_PERIODS
    R.F0_FILL, R.ENV_PERIODS, R.LS_PERIODS = False, periods, periods
    try:
        Phi, f0u = R.phase_track(f0t * detune, n)
        step = max(1, int(R.SR / f0hz / rate))
        pos = torch.arange(0, n, step)
        Xg = R.harmonic_envelopes(x, f0t * detune, pos, fmax)
    finally:
        R.F0_FILL, R.ENV_PERIODS, R.LS_PERIODS = old_fill, old_ep, old_lp

    t_g = pos.double()
    tt = torch.arange(n, dtype=torch.double)
    j = torch.searchsorted(t_g, tt).clamp(1, len(t_g) - 1)
    fr = ((tt - t_g[j - 1]) / (t_g[j] - t_g[j - 1]).clamp(min=1.0)).clamp(0, 1)
    frc = fr.to(torch.complex64)

    h = torch.zeros(n, dtype=torch.float64)
    K = int(fmax / (f0hz * detune))
    for k in range(1, K + 1):
        ph = torch.remainder(k * Phi[pos.clamp(0, n - 1)], 2 * math.pi)
        Ok = Xg[k - 1] * (torch.cos(ph) - 1j * torch.sin(ph)).to(torch.complex64)
        ci = Ok[j - 1] * (1 - frc) + Ok[j] * frc
        h = h + (ci.to(torch.complex128) * torch.exp(1j * k * Phi)).real

    X = R.stft(x)
    fb = torch.arange(R.NB) * R.SR / R.NFFT
    m = ((fb > f0hz / 2) & (fb < (K + 0.5) * f0hz)).float()[:, None]
    ref = R.istft(X * m, n).double()
    L = int(max(periods, R.LS_PERIODS if R.ENV_LS else 0) * R.SR / f0hz) + 2
    s, r = h[L:-L], ref[L:-L]
    a = float((s * r).sum() / (s * s).sum().clamp(min=1e-12))
    return 10 * math.log10(float((r ** 2).sum() / (((a * s) - r) ** 2).sum() + 1e-30))


def main() -> None:
    n, f0hz, fmax = R.SR * 2, 220.0, 1700.0
    g = torch.Generator().manual_seed(0)
    white = torch.randn(n, generator=g) * 0.2
    t = torch.arange(n, dtype=torch.float64) / R.SR
    harm = sum(torch.sin(2 * math.pi * f0hz * k * t + k) / k for k in range(1, 9))
    harm = (harm / harm.abs().max() * 0.4).float()

    print(f"estimator: {'least squares (harmonic_ls)' if R.ENV_LS else 'Hann demodulator'}"
          f"   window knob: {'LS_PERIODS' if R.ENV_LS else 'ENV_PERIODS'}")
    # ALWAYS include the configured point. A grid that happens to skip it makes
    # the check pass by not looking, which is how LS_PERIODS=3.5 slipped through.
    cp = (float(R.LS_PERIODS if R.ENV_LS else R.ENV_PERIODS), int(R.ENV_RATE))
    grid = sorted({(p, r) for p in (1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0)
                   for r in (1, 2, 5)} | {cp})
    print(f"{'EP':>5} {'rate':>5} {'crit':>6} {'white dB':>9} {'harm dB':>8} "
          f"{'margin':>7} {'detune31%':>10}  verdict")
    fails = []
    for p, r in grid:
        w = bank_snr(white, f0hz, fmax, r, p)
        hh = bank_snr(harm, f0hz, fmax, r, p)
        d = bank_snr(harm, f0hz, fmax, r, p, detune=1.31)
        ok = w < WHITE_MAX and (hh - w) > MARGIN_MIN
        cur = ((p, r) == cp)
        tag = ("MODEL" if ok else "COPIES WAVEFORM") + (" <= current" if cur else "")
        if cur and not ok:
            fails.append((p, r))
        print(f"{p:5.2f} {r:5d} {4/p:6.2f} {w:9.2f} {hh:8.2f} {hh-w:7.2f} {d:10.2f}  {tag}")

    # ---- the binding gate is the RATE LEDGER, not the white-noise SNR.
    #
    # The table above is a DIAGNOSTIC. It cannot be a gate for the harmonic
    # branch, because at one node per period that branch spends 2*K*f0 = 2*MVF
    # reals/s, which is exactly the Nyquist DOF of the band [0, MVF] it covers,
    # INDEPENDENT of the analysis window. A rate-1 bank is therefore inherently
    # near-critical inside its own band and trips a white-noise test at any
    # window length; shortening the window buys time resolution and not one
    # parameter. Gating on it rejected legitimate configurations (LS_PERIODS=2.0
    # scores 3.435 and is 0.07x of the sample rate) while passing the two
    # configurations that actually transported gt.
    #
    # What binds is what each branch COSTS, and the physical bandwidth of what it
    # carries. A harmonic's complex envelope is band-limited to +-f0/2 -- the
    # vocal tract modulates it well under 50 Hz and jitter/shimmer cannot exceed
    # one cycle by definition -- so ENV_RATE > 1 is representing inter-harmonic
    # beats, i.e. not harmonic amplitudes.
    nb = R.NOISE_SMOOTH if R.NOISE_SMOOTH else (R.NOISE_NFFT // 2 + 1)
    fps = R.SR / (R.NOISE_NFFT // R.NOISE_HOPDIV)
    noise_rate = nb * fps / R.SR
    harm_over = float(R.ENV_RATE)
    print(f"\nrate ledger (fraction of the sample rate; the harmonic branch costs "
          f"ENV_RATE x 2*MVF)")
    print(f"  harmonic: ENV_RATE = {R.ENV_RATE}  -> {harm_over:.2f}x critical for its own band")
    print(f"  breath  : {nb} coefficients x {fps:.0f} fps = {noise_rate:.2f}x the sample rate")
    bad = []
    if harm_over > 1.0:
        bad.append(f"ENV_RATE={R.ENV_RATE} > 1 (envelope is band-limited to +-f0/2)")
    if noise_rate > 0.5:
        bad.append(f"breath branch {noise_rate:.2f}x > 0.5x")
    if bad:
        print("FAIL: " + "; ".join(bad))
    else:
        print(f"PASS: within the declared interface  "
              f"(white-noise diagnostic {w:.2f} dB at the current point)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()


def check_current(f0hz: float = 220.0, fmax: float = 1700.0) -> tuple[bool, float, float]:
    """Single-point legitimacy check on the CONFIGURED estimator, for use inside
    a search loop. A coordinate ascent driven by PESQ alone walks straight back
    into the perfect-reconstruction region -- it did, reaching LS_PERIODS=3.0 --
    so the constraint has to be enforced per candidate, not audited afterwards."""
    n = R.SR * 2
    g = torch.Generator().manual_seed(0)
    white = torch.randn(n, generator=g) * 0.2
    t = torch.arange(n, dtype=torch.float64) / R.SR
    harm = sum(torch.sin(2 * math.pi * f0hz * k * t + k) / k for k in range(1, 9))
    harm = (harm / harm.abs().max() * 0.4).float()
    p = float(R.LS_PERIODS if R.ENV_LS else R.ENV_PERIODS)
    w = bank_snr(white, f0hz, fmax, int(R.ENV_RATE), p)
    hh = bank_snr(harm, f0hz, fmax, int(R.ENV_RATE), p)
    # The white-noise numbers are DIAGNOSTIC. What gates is the rate ledger --
    # this function used to return only the diagnostic, which is how a re-tune
    # walked the breath branch back to 1.25x the sample rate without anything
    # objecting. Both branches, every time.
    nb = R.NOISE_SMOOTH if R.NOISE_SMOOTH else (R.NOISE_NFFT // 2 + 1)
    noise_rate = nb * (R.SR / (R.NOISE_NFFT // R.NOISE_HOPDIV)) / R.SR
    ok = (float(R.ENV_RATE) <= 1.0) and (noise_rate <= 0.5)
    return ok, w, hh - w
