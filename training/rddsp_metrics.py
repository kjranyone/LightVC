"""Single source of truth for the rddsp numbers.

Everything here goes through resynthesize(), i.e. exactly the code path that
renders the audio, and is averaged over the whole eval set. An earlier harness
called synthesize_v2() directly with unrefined GCIs and reported crest 25.3 for
a configuration that actually renders at 19.2 -- numbers measured off the
shipping path are worse than no numbers."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from kansei_train import CACHE, DATA

BANDS = [0, 500, 1000, 2000, 4000, 8000, 16000, 22050]


def lsd(a, b):
    A, B = R.stft(a).abs(), R.stft(b).abs()
    T = min(A.shape[-1], B.shape[-1])
    return float((torch.log(A[..., :T] + 1e-5) - torch.log(B[..., :T] + 1e-5)).abs().mean())


def band_worst(a, b):
    bh = R.SR / R.NFFT
    A, B = R.stft(a).abs() ** 2, R.stft(b).abs() ** 2
    T = min(A.shape[-1], B.shape[-1])
    out = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        i, j = int(lo / bh), int(hi / bh)
        out.append(10 * math.log10(float(B[i:j, :T].sum() + 1e-12) / float(A[i:j, :T].sum() + 1e-12)))
    return max(abs(v) for v in out)


def crest(s):
    """p99.9 peak over rms. max/rms is decided by a single sample and is not a
    stable statistic -- it disagreed by 40% between adjacent runs of the same
    configuration."""
    return float(np.percentile(s.abs().numpy(), 99.9) / float(s.std().clamp(min=1e-9)))


def env_mod(sig, f0m):
    from scipy.signal import hilbert
    e = np.abs(hilbert(sig.numpy().astype(np.float64)))
    e = e - e.mean()
    E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    fr = np.fft.rfftfreq(len(e), 1 / R.SR)
    return float(E[(fr > f0m * .85) & (fr < f0m * 1.15)].max()
                 / (np.median(E[(fr > 20) & (fr < f0m * 3)]) + 1e-12))


def cycle_corr(x, y, gci, f0m):
    T0 = int(R.SR / f0m)
    g = gci[(gci > T0) & (gci < len(x) - 2 * T0)][:400]
    if len(g) < 20:
        return float("nan")
    A = torch.stack([x[i - T0 // 2: i - T0 // 2 + 2 * T0] for i in g.tolist()]).mean(0)
    B = torch.stack([y[i - T0 // 2: i - T0 // 2 + 2 * T0] for i in g.tolist()]).mean(0)
    A, B = A - A.mean(), B - B.mean()
    return float((A @ B) / (A.norm() * B.norm() + 1e-9))


def main():
    import librosa
    rows = []
    for uid in [p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]]:
        x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
        gt = torch.tensor(x[: R.SR * 8])
        y, h, nz, p = R.resynthesize(gt)
        n = min(len(y), len(gt))
        gt, y, nz = gt[:n], y[:n], nz[:n]
        f0 = p["f0"]
        fm = float(f0[f0 > 50].median())
        # floor: same noise renderer given the PERFECT target magnitude
        M = R.stft(gt).abs()
        cn = R.cyclic_noise(p["gci"], n, fm)
        g2 = torch.Generator().manual_seed(1)
        src = 0.7 * cn / cn.std().clamp(min=1e-9) + 0.3 * torch.randn(n, generator=g2)
        floor_lsd = lsd(gt, R._shaped_noise(M, src, n, 0))
        rows.append(dict(uid=uid, lsd=lsd(gt, y), floor=floor_lsd,
                         band=band_worst(gt, y), crest=crest(y), crest_gt=crest(gt),
                         # compare like with like: the FULL output against the
                         # full reference. Measuring the noise branch alone
                         # against the whole of gt is not a comparison -- gt's
                         # harmonics dominate its envelope modulation at f0.
                         mod=env_mod(y, fm), mod_gt=env_mod(gt, fm),
                         cyc=cycle_corr(gt, h, p["gci"], fm)))
    print(f"{'metric':26s} {'value':>9s} {'gt/target':>10s} {'floor':>8s}")
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"{'LSD':26s} {m('lsd'):9.3f} {0.0:10.3f} {m('floor'):8.3f}  <- floor is architectural")
    print(f"{'band energy worst (dB)':26s} {m('band'):9.2f} {0.0:10.2f} {'-':>8s}")
    print(f"{'crest factor':26s} {m('crest'):9.2f} {m('crest_gt'):10.2f} {'-':>8s}")
    print(f"{'breath env mod @f0':26s} {m('mod'):9.2f} {m('mod_gt'):10.2f} {'-':>8s}")
    print(f"{'glottal cycle corr':26s} {m('cyc'):9.3f} {1.0:10.3f} {'-':>8s}")
    print()
    for r in rows:
        print(f"  {r['uid'][-8:]}  LSD {r['lsd']:.3f} (floor {r['floor']:.3f})  band {r['band']:.2f}dB  "
              f"crest {r['crest']:.1f}/{r['crest_gt']:.1f}  mod {r['mod']:.2f}/{r['mod_gt']:.2f}  cyc {r['cyc']:.3f}")


if __name__ == "__main__":
    main()
