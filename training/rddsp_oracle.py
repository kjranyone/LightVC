import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from kansei_train import CACHE, DATA, octave_correct

SR = 44100
HOP = 512
NFFT = 2048
OUT = Path("../results/rddsp_oracle")
EPS = 1e-8


def analyze(x: np.ndarray, f0: np.ndarray, kmax: int = 400):
    w = torch.hann_window(NFFT)
    X = torch.stft(torch.tensor(x), NFFT, HOP, NFFT, w, center=True, return_complex=True)
    mag = X.abs().numpy() * 2.0 / w.sum().item()
    nb, T = mag.shape
    T = min(T, len(f0))
    mag, f0 = mag[:, :T], f0[:T]
    binhz = SR / NFFT
    amp = np.zeros((kmax, T), dtype=np.float32)
    voiced = f0 > 50.0
    for k in range(1, kmax + 1):
        fk = k * f0
        ok = voiced & (fk < SR / 2 - binhz)
        if not ok.any():
            break
        b = fk / binhz
        lo = np.clip(np.floor(b).astype(int), 0, nb - 2)
        fr = b - lo
        a = mag[lo, np.arange(T)] * (1 - fr) + mag[lo + 1, np.arange(T)] * fr
        amp[k - 1] = np.where(ok, a, 0.0)
    return amp, f0


def upsample(v: np.ndarray, n: int) -> np.ndarray:
    t = np.arange(n) / HOP
    i = np.clip(t.astype(int), 0, v.shape[-1] - 2)
    fr = t - i
    return v[..., i] * (1 - fr) + v[..., i + 1] * fr


def synth_harmonic(amp: np.ndarray, f0: np.ndarray, n: int) -> np.ndarray:
    f0u = upsample(f0.astype(np.float64), n)
    y = np.zeros(n, dtype=np.float64)
    kmax = amp.shape[0]
    phase = 2 * np.pi * np.cumsum(f0u) / SR
    for k in range(1, kmax + 1):
        if amp[k - 1].max() <= 0:
            continue
        au = upsample(amp[k - 1].astype(np.float64), n)
        au[k * f0u >= SR / 2] = 0.0
        y += au * np.sin(k * phase)
    return y


def synth_noise(gt: np.ndarray, h: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Noise floor by MAGNITUDE-domain subtraction. The additive model rebuilds
    phase, so gt-h in the time domain is not a residual (it carries the harmonics
    twice) -- subtract |H| from |GT| instead and give the remainder random phase."""
    w = torch.hann_window(NFFT)
    kw = dict(n_fft=NFFT, hop_length=HOP, win_length=NFFT, window=w, center=True)
    G = torch.stft(torch.tensor(gt, dtype=torch.float32), return_complex=True, **kw)
    H = torch.stft(torch.tensor(h, dtype=torch.float32), return_complex=True, **kw)
    N = (G.abs() - H.abs()).clamp(min=0.0)
    g = torch.Generator().manual_seed(seed)
    ph = torch.rand(N.shape, generator=g) * 2 * np.pi
    S = N * (torch.cos(ph) + 1j * torch.sin(ph))
    return torch.istft(S, length=n, **kw).numpy()


def logspec_dist(a: np.ndarray, b: np.ndarray) -> float:
    w = torch.hann_window(NFFT)
    kw = dict(n_fft=NFFT, hop_length=HOP, win_length=NFFT, window=w, center=True)
    A = torch.stft(torch.tensor(a, dtype=torch.float32), return_complex=True, **kw).abs()
    B = torch.stft(torch.tensor(b, dtype=torch.float32), return_complex=True, **kw).abs()
    return float((torch.log(A + 1e-5) - torch.log(B + 1e-5)).abs().mean())


def gain_match(y: np.ndarray, gt: np.ndarray) -> np.ndarray:
    g = np.sqrt((gt ** 2).mean() / ((y ** 2).mean() + EPS))
    y = y * g
    p = np.abs(y).max()
    return (y * (0.95 / p) if p > 0.95 else y).astype(np.float32)


def main() -> None:
    import librosa

    OUT.mkdir(parents=True, exist_ok=True)
    uids = [p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]]
    for uid in uids:
        x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=SR, mono=True)
        f0 = octave_correct(np.load(CACHE / (uid + ".npz"))["f0"].astype(np.float32))
        n = min(len(x), len(f0) * HOP, int(8.0 * SR))
        gt = x[:n].astype(np.float64)
        amp, f0c = analyze(gt.astype(np.float32), f0)
        h = synth_harmonic(amp, f0c, n)
        nz = synth_noise(gt, h, n)
        y = h + nz
        hv = float(np.sqrt((h ** 2).mean()))
        nv = float(np.sqrt((nz ** 2).mean()))
        sf.write(OUT / f"{uid}_gt.wav", gt.astype(np.float32), SR)
        sf.write(OUT / f"{uid}_ddsp.wav", gain_match(y, gt), SR)
        sf.write(OUT / f"{uid}_harmonly.wav", gain_match(h, gt), SR)
        print(f"  {uid}  {n/SR:.1f}s  LSD gt-vs-ddsp {logspec_dist(gt, y):.3f} "
              f"(harm-only {logspec_dist(gt, h):.3f})  rms h {hv:.4f} / noise {nv:.4f} "
              f"| voiced {100*(f0c>50).mean():4.1f}%  f0 med {np.median(f0c[f0c>50]):.0f}Hz")
    print(f"wrote -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
