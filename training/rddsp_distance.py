"""Turn the ear's "the image is distant" into numbers.

The listening gate on results/rddsp_ab_hf returned one word: 音像が遠い -- the
sound image sits further away than the reference. Level is already matched to gt
per file, so it is not loudness. Four physical correlates can produce that
percept, and they call for different fixes, so measure all four rather than
guess:

  1. BAND LEVEL      a dip in the presence region (2-5 kHz) reads as distance.
  2. MODULATION DEPTH distance fills the gaps between events. If the per-band
                     envelope contrast (std/mean) is lower than gt, the signal
                     has less silence between syllables -- exactly what a room
                     does, and exactly what an over-generous noise floor does.
  3. CREST FACTOR    blunted transients read as far away. Peak/RMS per band.
  4. INTER-HARMONIC FLOOR
                     energy BETWEEN the harmonics, relative to energy AT them.
                     A raised floor is heard as veiling, and this project's core
                     runs one deliberately (FLOOR_K = 0.25). This is the first
                     suspect.

Measured on the same six unseen speakers that were auditioned, against gt, for
both the DSP core and BigVGAN -- so "distant" can be attributed to our core
rather than to the harness, the level matching, or PESQ's blind spots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R

AB = Path("/home/kojirotanaka/kjranyone/LightVC/results/rddsp_ab_hf")
BANDS = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 8000), (8000, 16000)]
NFFT, HOP = 1024, 256


def bandpass_env(x: torch.Tensor, lo: float, hi: float):
    """Band energy envelope at the STFT frame rate."""
    w = torch.hann_window(NFFT)
    S = torch.stft(x, NFFT, HOP, NFFT, w, center=True, return_complex=True).abs()
    f = torch.fft.rfftfreq(NFFT, 1 / R.SR)
    m = (f >= lo) & (f < hi)
    return (S[m] ** 2).sum(0).sqrt()


def crest(x: torch.Tensor, lo: float, hi: float) -> float:
    X = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(x.shape[-1], 1 / R.SR)
    X = X * ((f >= lo) & (f < hi))
    y = torch.fft.irfft(X, n=x.shape[-1])
    r = float(y.pow(2).mean().sqrt())
    return 20 * np.log10(float(y.abs().max()) / max(r, 1e-12))


def harmonic_floor(x: torch.Tensor, f0: torch.Tensor) -> float:
    """dB between the harmonic peaks and the valleys, over voiced frames.

    High value = clean harmonic structure. Low value = filled-in floor, which is
    what veiling sounds like."""
    w = torch.hann_window(NFFT)
    S = torch.stft(x, NFFT, HOP, NFFT, w, center=True, return_complex=True).abs()
    f = torch.fft.rfftfreq(NFFT, 1 / R.SR)
    v = R.frame_upsample((f0 > 50).double(), x.shape[-1])
    fr = torch.nn.functional.interpolate(v[None, None], size=S.shape[-1],
                                         mode="linear", align_corners=False)[0, 0]
    keep = fr > 0.5
    if keep.sum() < 5:
        return float("nan")
    fm = float(R.median_f0(f0))
    band = (f > 300) & (f < 5000)
    Sv = S[:, keep][band]
    fb = f[band]
    # distance from each bin to the nearest multiple of f0, in units of f0
    d = torch.remainder(fb / fm, 1.0)
    d = torch.minimum(d, 1 - d)
    peak = d < 0.15
    vall = d > 0.35
    if peak.sum() < 3 or vall.sum() < 3:
        return float("nan")
    p = float((Sv[peak] ** 2).mean())
    q = float((Sv[vall] ** 2).mean())
    return 10 * np.log10(p / max(q, 1e-20))


def main() -> None:
    ids = sorted({p.name.split("_")[0] for p in AB.glob("*_gt.wav")})
    rows = {k: {b: [] for b in BANDS} for k in ("rddsp", "bigvgan")}
    mod = {k: {b: [] for b in BANDS} for k in ("rddsp", "bigvgan")}
    cr = {k: {b: [] for b in BANDS} for k in ("rddsp", "bigvgan")}
    hnr = {"gt": [], "rddsp": [], "bigvgan": []}

    for uid in ids:
        g, _ = sf.read(AB / f"{uid}_gt.wav", dtype="float32")
        gt = torch.tensor(g)
        sigs = {"gt": gt}
        for tag, suf in (("rddsp", "b_rddsp"), ("bigvgan", "c_bigvgan")):
            y, _ = sf.read(AB / f"{uid}_{suf}.wav", dtype="float32")
            sigs[tag] = torch.tensor(y)
        n = min(len(s) for s in sigs.values())
        sigs = {k: v[:n] for k, v in sigs.items()}
        try:
            _, _, _, prm = R.resynthesize(sigs["gt"])
            f0 = prm["f0"]
        except Exception:
            f0 = None

        for lo, hi in BANDS:
            eg = bandpass_env(sigs["gt"], lo, hi)
            lg = 20 * np.log10(max(float(eg.pow(2).mean().sqrt()), 1e-12))
            mg = float(eg.std() / eg.mean().clamp(min=1e-12))
            cg = crest(sigs["gt"], lo, hi)
            for tag in ("rddsp", "bigvgan"):
                e = bandpass_env(sigs[tag], lo, hi)
                l = 20 * np.log10(max(float(e.pow(2).mean().sqrt()), 1e-12))
                rows[tag][(lo, hi)].append(l - lg)
                mod[tag][(lo, hi)].append(float(e.std() / e.mean().clamp(min=1e-12)) - mg)
                cr[tag][(lo, hi)].append(crest(sigs[tag], lo, hi) - cg)
        if f0 is not None:
            for tag in ("gt", "rddsp", "bigvgan"):
                hnr[tag].append(harmonic_floor(sigs[tag], f0))

    print(f"  {len(ids)} auditioned speakers, level already matched to gt.\n")
    print("  BAND LEVEL vs gt (dB) -- a presence dip reads as distance")
    print(f"  {'band Hz':>12} {'rddsp':>9} {'bigvgan':>9}")
    for b in BANDS:
        print(f"  {f'{b[0]}-{b[1]}':>12} {np.mean(rows['rddsp'][b]):+9.2f} "
              f"{np.mean(rows['bigvgan'][b]):+9.2f}")

    print("\n  ENVELOPE MODULATION DEPTH vs gt (std/mean) -- less contrast = more room")
    print(f"  {'band Hz':>12} {'rddsp':>9} {'bigvgan':>9}")
    for b in BANDS:
        print(f"  {f'{b[0]}-{b[1]}':>12} {np.mean(mod['rddsp'][b]):+9.3f} "
              f"{np.mean(mod['bigvgan'][b]):+9.3f}")

    print("\n  CREST FACTOR vs gt (dB) -- blunted transients = further away")
    print(f"  {'band Hz':>12} {'rddsp':>9} {'bigvgan':>9}")
    for b in BANDS:
        print(f"  {f'{b[0]}-{b[1]}':>12} {np.mean(cr['rddsp'][b]):+9.2f} "
              f"{np.mean(cr['bigvgan'][b]):+9.2f}")

    print("\n  HARMONIC PEAK-TO-VALLEY, 300-5000 Hz, voiced frames (dB)")
    print("  higher = cleaner harmonic structure; lower = filled-in floor = veiled")
    for tag in ("gt", "rddsp", "bigvgan"):
        v = np.array([x for x in hnr[tag] if np.isfinite(x)])
        if v.size:
            print(f"  {tag:9s} {v.mean():7.2f}   (vs gt "
                  f"{v.mean() - np.mean([x for x in hnr['gt'] if np.isfinite(x)]):+.2f})")


if __name__ == "__main__":
    main()
