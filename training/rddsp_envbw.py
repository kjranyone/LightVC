"""Is the demodulated harmonic envelope actually band-limited to +-f0/2?

This closes the open review item recorded in 13:

    ENV_RATE=1 gives rho = rho*, which does NOT satisfy the strict inequality of
    Corollary 3.2. What rescues the equality is the physical assumption that the
    complex envelope of a harmonic is band-limited to +-f0/2 -- and the only
    tool that could test it, the white-noise reconstruction test, was declared
    inadmissible for the harmonic branch by 2.4. Measuring the demodulated
    envelope's spectrum settles it in one page. NOT DONE.

Measuring it naively is circular: the implementation's own estimator low-passes
at ~f0/3 (LS_PERIODS = 3), so of course its output is band-limited. The
measurement has to demodulate through a filter WIDER than f0/2 and ask how much
energy lands outside. That filter also admits the neighbouring harmonics, which
sit exactly at +-f0 -- so raw excess energy above f0/2 is ambiguous between
"fast envelope" and "leakage from harmonic k+-1".

The control resolves it, and is the same device 6.4 requires of any learned
claim. Build a synthetic signal with the SAME f0 track and the SAME harmonic
count whose envelopes are band-limited BY CONSTRUCTION to +-f0/8, and push it
through the identical pipeline. Whatever appears above f0/2 there is leakage and
estimator artefact, because nothing else could have produced it. The excess of
the real signal over that control is the part that is genuinely fast.

Reading:
  real ~= control          -> the envelope IS band-limited; rho = rho* is safe
                              in practice and ENV_RATE=1 is not hiding a channel
  real >> control          -> the envelope is NOT band-limited; ENV_RATE=1
                              undersamples it, and the ledger's equality case is
                              not rescued by physics. Either the interface is
                              lossy (fine, but say so) or the estimator is
                              carrying the difference somewhere undeclared
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
from rddsp_dspecies_multi import pick

# A demodulator of W periods has a main lobe of half-width ~f0*2/W, so it can
# only SEE excess energy out to that edge. W=2 covers the first octave above the
# interface's Nyquist, (f0/2, f0) -- the octave that matters most, since energy
# just above Nyquist aliases to just below it. W=1 doubles the reach to 2*f0 at
# the price of putting the neighbouring harmonics squarely inside the passband;
# the control absorbs that, because it has the same neighbours.
LP_WIDTHS = (2.0, 1.0)
KS = (1, 2, 3, 4, 6, 8)
SECONDS = 5


def demod(x: torch.Tensor, Phi: torch.Tensor, k: int, L: int) -> torch.Tensor:
    w = torch.hann_window(L, periodic=False, dtype=torch.float64)
    w = w / w.sum()
    return 2.0 * R._lp_centred(x.double() * torch.exp(1j * k * Phi).conj(), w)


def mod_spectrum(a: torch.Tensor, nf: int = 8192):
    """Two-sided modulation spectrum of a complex envelope, in Hz."""
    a = a - a.mean()
    hop = nf // 4
    w = torch.hann_window(nf, dtype=torch.float64)
    S = torch.stft(a, nf, hop, nf, w, center=False, return_complex=True)
    P = (S.abs() ** 2).mean(-1)
    f = torch.fft.fftfreq(nf, 1 / R.SR).double()
    o = torch.argsort(f)
    return f[o], P[o]


def frac_above(f, P, cut):
    tot = float(P.sum())
    return float(P[f.abs() > cut].sum()) / max(tot, 1e-30)


def synth_control(Phi: torch.Tensor, f0m: float, n: int, kmax: int,
                  gen: torch.Generator, bw_div: float = 8.0) -> torch.Tensor:
    """Same carriers, envelopes band-limited to +-f0/bw_div by construction.

    Built by low-passing white noise with a Hann of bw_div periods, whose main
    lobe half-width is f0/bw_div * ... -- the exact cutoff does not matter, only
    that it is far inside f0/2, which a window of 8 periods guarantees."""
    L = int(bw_div * R.SR / f0m) | 1
    w = torch.hann_window(L, periodic=False, dtype=torch.float64)
    w = w / w.sum()
    y = torch.zeros(n, dtype=torch.float64)
    for k in range(1, kmax + 1):
        z = torch.randn(n, generator=gen, dtype=torch.float64) + \
            1j * torch.randn(n, generator=gen, dtype=torch.float64)
        b = R._lp_centred(z, w) * (1.0 / k)
        y = y + (b * torch.exp(1j * k * Phi)).real
    return y


def pick_from(root: Path, n: int, seed: int = 0):
    import random
    spk = sorted(p.name for p in root.iterdir() if p.is_dir())
    rng = random.Random(seed)
    rng.shuffle(spk)
    out = []
    for s in spk[:n]:
        w = sorted((root / s).glob("*.wav"))
        if w:
            out.append(w[rng.randrange(len(w))])
    return out


def main() -> None:
    # The female set has median f0 215-472 Hz, so +-f0/2 is a wide band and
    # band-limitation is easy to satisfy. A male voice at 100 Hz has to fit the
    # same envelope inside +-50 Hz. If the assumption holds only for high pitch,
    # ENV_RATE=1 is a female-only interface and the ledger has to say so.
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if root is not None:
        te_p = pick_from(root, 6)
        print(f"  corpus {root.name}", flush=True)
    else:
        _, te_p = pick(80, 12)
    rows = {(w, k): {"real": [], "ctrl": []} for w in LP_WIDTHS for k in KS}
    f0s = []
    nutt = 0
    for p in te_p[:6]:
        x, _ = librosa.load(str(p), sr=R.SR, mono=True)
        if len(x) < R.SR * 2:
            continue
        gt = torch.tensor(x[: R.SR * SECONDS])
        try:
            _, _, _, prm = R.resynthesize(gt)
        except Exception:
            continue
        f0 = prm["f0"]
        if not bool((f0 > 50).any()):
            continue
        n = gt.shape[-1]
        Phi, _ = R.phase_track(f0, n)
        f0m = float(R.median_f0(f0))
        P = R.SR / f0m
        f0s.append(f0m)
        kmax = max(KS)
        g = torch.Generator().manual_seed(nutt)
        ctrl = synth_control(Phi, f0m, n, kmax, g)
        # Voiced samples only: an envelope measured across an unvoiced gap is
        # measuring the gap, and every harmonic model is silent there anyway.
        v = R.frame_upsample((f0 > 50).double(), n) > 0.5
        for wper in LP_WIDTHS:
            L = int(wper * P) | 1
            m = slice(4 * L, n - 4 * L)
            for k in KS:
                for tag, sig in (("real", gt), ("ctrl", ctrl)):
                    a = demod(sig, Phi, k, L)[m] * v[m]
                    fr, Pw = mod_spectrum(a)
                    rows[(wper, k)][tag].append(frac_above(fr, Pw, f0m / 2))
        nutt += 1
        print(f"  {p.parent.name[:14]:16s} f0 {f0m:6.1f} Hz  P {P:6.1f}  "
              f"k=1 real {rows[(2.0,1)]['real'][-1]*100:5.2f}%  "
              f"ctrl {rows[(2.0,1)]['ctrl'][-1]*100:5.2f}%", flush=True)

    print(f"\n  ---- energy above f0/2 in the demodulated envelope, "
          f"{nutt} speakers, median f0 {np.median(f0s):.0f} Hz ----\n")
    for wper in LP_WIDTHS:
        print(f"  demodulator Hann of {wper} periods -> passband +-{2/wper:.0f}*f0, "
              f"so excess is detectable in (f0/2, {2/wper:.0f}*f0)")
        print(f"  {'k':>3} {'real %':>9} {'control %':>11} {'excess %':>10}")
        for k in KS:
            r = 100 * float(np.mean(rows[(wper, k)]["real"]))
            c = 100 * float(np.mean(rows[(wper, k)]["ctrl"]))
            print(f"  {k:3d} {r:9.2f} {c:11.2f} {r-c:10.2f}")
        r1 = 100 * float(np.mean([np.mean(rows[(wper, k)]["real"]) for k in KS]))
        c1 = 100 * float(np.mean([np.mean(rows[(wper, k)]["ctrl"]) for k in KS]))
        print(f"  {'mean':>3} {r1:9.2f} {c1:11.2f} {r1-c1:10.2f}\n")
    print("  The control's envelopes are band-limited to f0/8 BY CONSTRUCTION, so")
    print("  its reading is pure leakage and estimator artefact. Excess is the part")
    print("  of the real envelope that ENV_RATE=1 cannot represent -- at one node")
    print("  per period the interface's Nyquist is f0/2.")
    print("  LIMIT: nothing above the demodulator's own passband is visible, so")
    print("  this bounds the excess in the stated interval and NOT above it.")


if __name__ == "__main__":
    main()
