import sys, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from kansei_train import CACHE, DATA


def cycle_corr(x, y, gci, f0m):
    """GCI-aligned average glottal cycle: correlation of the pulse SHAPE."""
    T0 = int(R.SR / f0m)
    g = gci[(gci > T0) & (gci < len(x) - 2 * T0)]
    if len(g) < 20:
        return float("nan")
    A = torch.stack([x[i - T0 // 2: i - T0 // 2 + 2 * T0] for i in g[:400].tolist()])
    B = torch.stack([y[i - T0 // 2: i - T0 // 2 + 2 * T0] for i in g[:400].tolist()])
    a, b = A.mean(0), B.mean(0)
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def env_mod_at_f0(sig, f0m):
    """Peak of the noise ENVELOPE spectrum at f0, normalised by its own floor.
    Cyclic noise is re-excited every glottal closure -> envelope modulated at f0."""
    from scipy.signal import hilbert
    e = np.abs(hilbert(sig.numpy().astype(np.float64)))
    e = e - e.mean()
    E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    fr = np.fft.rfftfreq(len(e), 1 / R.SR)
    band = (fr > f0m * 0.85) & (fr < f0m * 1.15)
    ref = (fr > 20) & (fr < f0m * 3)
    return float(E[band].max() / (np.median(E[ref]) + 1e-12))


def main():
    import librosa
    uid = sorted(p.stem for p in CACHE.glob("*.npz"))[-1]
    x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
    gt = torch.tensor(x[: R.SR * 4])
    p = R.analyze(gt)
    f0 = p["f0"]
    fm = float(f0[f0 > 50].median())
    gci, _ = R.zff_gci(gt, fm)
    f0_at = f0[(gci // R.HOP).clamp(0, len(f0) - 1)]
    Xg = R.harmonic_analysis_at(gt, gci, f0_at)
    n = gt.shape[-1]

    y1, h1, n1 = R.synthesize(f0, p["mvf"], p["apbins"], p["amp"], p["noisemag"], n)
    y2, h2, n2 = R.synthesize_v2(f0, p["mvf"], p["apbins"], p["noisemag"], n, gci, Xg, f0_at)

    print(f"{'':22s} {'v1 (sin, random-phase)':>24s} {'v2 (glottal, cyclic)':>22s}   gt")
    c1, c2 = cycle_corr(gt, h1, gci, fm), cycle_corr(gt, h2, gci, fm)
    print(f"{'glottal cycle corr':22s} {c1:24.3f} {c2:22.3f}   1.000")
    m1, m2 = env_mod_at_f0(n1, fm), env_mod_at_f0(n2, fm)
    mg = env_mod_at_f0(gt, fm)
    print(f"{'breath env mod @f0':22s} {m1:24.2f} {m2:22.2f}   {mg:.2f}")
    def crest(s):
        return float(s.abs().max() / s.std().clamp(min=1e-9))
    print(f"{'crest factor':22s} {crest(y1):24.2f} {crest(y2):22.2f}   {crest(gt):.2f}")
    def lsd(a, b):
        A, B = R.stft(a).abs(), R.stft(b).abs()
        T = min(A.shape[-1], B.shape[-1])
        return float((torch.log(A[..., :T] + 1e-5) - torch.log(B[..., :T] + 1e-5)).abs().mean())
    print(f"{'LSD vs gt':22s} {lsd(gt,y1):24.3f} {lsd(gt,y2):22.3f}   0.000")


if __name__ == "__main__":
    main()
