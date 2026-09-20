"""Task C: a gate that can see 8-22 kHz, which PESQ cannot.

score_one() resamples both signals to 16 kHz before calling PESQ-wb, so
everything above 8 kHz is invisible to every number this project has optimised.
For an ASMR vocoder that band IS the product -- breath, lip and contact noise
live there -- and the project has already recorded PESQ misranking the same
defect twice.

This measures what PESQ cannot, on the same renders:

  band       per-octave energy error in dB, 0-22 kHz, so upper-band tilt shows
  mod        envelope modulation spectrum match 20-400 Hz (roughness / graininess)
  hf         8-22 kHz energy error alone, the band PESQ discards
  crest      peak-to-rms of the high band, which separates "hiss" from "texture"

It reports each configuration on all of them so a choice that PESQ likes and the
upper band hates becomes visible instead of silently shipping.
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

OCT = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000),
       (4000, 8000), (8000, 16000), (16000, 22050)]


def band_db(a: torch.Tensor, b: torch.Tensor):
    A, B = R.stft(a).abs() ** 2, R.stft(b).abs() ** 2
    T = min(A.shape[-1], B.shape[-1])
    binhz = R.SR / R.NFFT
    out = []
    for lo, hi in OCT:
        i, j = int(lo / binhz), int(hi / binhz)
        ea = float(A[i:j, :T].sum()) + 1e-20
        eb = float(B[i:j, :T].sum()) + 1e-20
        out.append(10 * math.log10(eb / ea))
    return out


def hp(x: torch.Tensor, cut: float = 8000.0) -> torch.Tensor:
    X = R.stft(x)
    fb = torch.arange(R.NB) * R.SR / R.NFFT
    return R.istft(X * (fb > cut).float()[:, None], x.shape[-1])


def mod_spectrum(x: torch.Tensor, lo: float = 20.0, hi: float = 400.0):
    """Envelope modulation spectrum of the HIGH band, where graininess lives."""
    from scipy.signal import hilbert
    e = np.abs(hilbert(hp(x).detach().numpy().astype(np.float64)))
    e = e - e.mean()
    n = len(e)
    E = np.abs(np.fft.rfft(e * np.hanning(n)))
    fr = np.fft.rfftfreq(n, 1 / R.SR)
    m = (fr >= lo) & (fr <= hi)
    v = E[m]
    return v / (np.linalg.norm(v) + 1e-12)


def crest(x: torch.Tensor) -> float:
    h = hp(x).detach().numpy().astype(np.float64)
    r = np.sqrt((h ** 2).mean()) + 1e-12
    return 20 * math.log10(float(np.abs(h).max()) / r)


def evaluate(ids, tag: str, **cfg):
    old = {k: getattr(R, k) for k in cfg}
    for k, v in cfg.items():
        setattr(R, k, v)
    pesq, bands, modc, cr, hfd = [], [], [], [], []
    for uid in ids:
        x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
        gt = torch.tensor(x[: R.SR * 6])
        y, _, _, _ = R.resynthesize(gt)
        n = min(len(y), len(gt))
        gt, y = gt[:n], y[:n]
        y = y * float(torch.sqrt((gt ** 2).mean() / ((y ** 2).mean() + 1e-12)))
        pesq.append(score_one(gt, y))
        d = band_db(gt, y)
        bands.append(d)
        hfd.append(d[-2:])
        modc.append(float(np.dot(mod_spectrum(gt), mod_spectrum(y))))
        cr.append(crest(y) - crest(gt))
    for k, v in old.items():
        setattr(R, k, v)
    b = np.mean(bands, axis=0)
    print(f"{tag:26s} PESQ {np.mean(pesq):6.3f} | hf8-22k {np.mean(hfd):+6.2f}dB | "
          f"mod-match {np.mean(modc):.4f} | crest {np.mean(cr):+5.2f}dB")
    print(f"{'':26s} octaves " + " ".join(f"{v:+5.1f}" for v in b))
    return float(np.mean(pesq)), float(np.mean(hfd)), float(np.mean(modc))


def main() -> None:
    uids = [p.stem for p in sorted(CACHE.glob("*.npz"))]
    ids = uids[40:48]
    print("octave bins: " + " ".join(f"{lo//1000 if lo>=1000 else lo}-"
                                     f"{hi//1000 if hi>=1000 else hi}"
                                     for lo, hi in OCT) + "\n")
    evaluate(ids, "shipping")
    evaluate(ids, "PULSE_MIX 0.45", PULSE_MIX=0.45)
    evaluate(ids, "MVF 3000", MVF_FLOOR=3000.0)
    evaluate(ids, "MVF 5000", MVF_FLOOR=5000.0)
    evaluate(ids, "NOISE_SMOOTH 60", NOISE_SMOOTH=60)
    evaluate(ids, "NOISE_SMOOTH 260", NOISE_SMOOTH=260)
    evaluate(ids, "NOISE_HOPDIV 6", NOISE_HOPDIV=6)


if __name__ == "__main__":
    main()
