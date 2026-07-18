"""R-oracle mixed-phase gate (RESEARCH.md R-oracle, 2026-07-16).

Decisive kill-switch for the source-filter ceiling. The recorded negative result
(current/vocoder.md:6) is min-phase-specific: WORLD and our NSF-LTV oracle (GT
perfect envelope, MIN-phase) both ear-FAIL, istft (71dB recon) ear-PASSES. The
one UNTESTED variable is phase: glottal open phase is MAXIMUM phase, impossible
in a min-phase filter. kansei broke the same ceiling with a non-min-phase neural
phase (circumstantial). This gate isolates phase as the single variable.

Both arms share ONE synthetic harmonic+noise excitation and ONE liftered
magnitude envelope (from the GT log-spectrum). They differ ONLY in the filter
phase:
  arm 'min'  = min-phase(magnitude)               [the known ear-FAIL]
  arm 'mix'  = GT liftered complex-cepstrum phase  [the untested candidate]
Because Re(complex cepstrum) is the real cepstrum, symmetric low-quefrency
liftering gives BOTH arms an identical magnitude response by construction; only
the phase differs. Linear (integer-delay) phase is removed so no per-frame
timing jitter is injected. Anchors: 'gt' (real) and 'istft' (recon, ear-PASS).

Judge = human ear (listen_gui.py on results/roracle_mix/). No training.

Usage:
  cd training
  uv run python roracle_mixphase.py            # 4 e2_triage af1ad utts
  uv run python roracle_mixphase.py --files a.wav b.wav
  uv run python roracle_mixphase.py --order-scale 1.0 --k 1024
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from ltv_render import HOP, NFFT, SR, HarmonicSource, ltv_ola

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/roracle_mix"
EPS = 1e-8
CENTER = NFFT // 2

DEFAULT_UIDS = ["af1ad5575a3fa383_00035936", "af1ad5575a3fa383_00027042",
                "af1ad5575a3fa383_00040870", "af1ad5575a3fa383_00036578"]


def load_wav(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    peak = np.abs(y).max() + EPS
    if peak > 0.95:
        y = y * (0.95 / peak)
    return y.astype(np.float64)


def gt_f0(y: np.ndarray, n_frames: int) -> np.ndarray:
    import pyworld as pw
    fp = 1000.0 * HOP / SR
    f0, t = pw.harvest(y, SR, frame_period=fp)
    f0 = pw.stonemask(y, f0, t, SR)
    ft = np.arange(n_frames) * HOP / SR
    f0i = np.interp(ft, t, f0)
    f0i[f0i < 1.0] = 0.0
    return f0i


def _full_spectrum(col: np.ndarray) -> np.ndarray:
    return np.concatenate([col, np.conj(col[-2:0:-1])])


def _lifter_win(order: int) -> np.ndarray:
    w = np.zeros(NFFT)
    o = min(order, CENTER)
    w[:o + 1] = 1.0
    if o > 0:
        w[NFFT - o:] = 1.0
    return w


def _min_fold(rceps: np.ndarray) -> np.ndarray:
    c = np.zeros(NFFT)
    c[0] = rceps[0]
    c[1:CENTER] = 2.0 * rceps[1:CENTER]
    c[CENTER] = rceps[CENTER]
    return c


def build_firs(S: np.ndarray, f0: np.ndarray, k: int, order_scale: float,
               clamp_nats: float) -> tuple[np.ndarray, np.ndarray]:
    T = S.shape[1]
    b_min = np.zeros((T, k), dtype=np.float32)
    b_mix = np.zeros((T, k), dtype=np.float32)
    half = k // 2
    for t in range(T):
        X = _full_spectrum(S[:, t])
        logmag = np.log(np.abs(X) + EPS)
        phase = np.unwrap(np.angle(X))
        ndelay = np.round(phase[CENTER] / np.pi)
        phase = phase - np.pi * ndelay * np.arange(NFFT) / CENTER
        f = f0[t] if f0[t] > 1.0 else 250.0
        order = int(round(order_scale * SR / (2.0 * max(f, 70.0))))
        win = _lifter_win(order)

        rceps = np.fft.ifft(logmag).real
        cmin = _min_fold(rceps * win)
        logmin = np.fft.fft(cmin)
        logmin = np.clip(logmin.real, -clamp_nats, clamp_nats) + 1j * logmin.imag
        hmin = np.fft.ifft(np.exp(logmin)).real
        b_min[t] = hmin[:k]

        cceps = np.fft.ifft(logmag + 1j * phase).real
        cl = cceps * win
        logmix = np.fft.fft(cl)
        logmix = np.clip(logmix.real, -clamp_nats, clamp_nats) + 1j * logmix.imag
        hmix = np.fft.ifft(np.exp(logmix)).real
        b_mix[t] = np.concatenate([hmix[NFFT - half:], hmix[:half]])
    return b_min, b_mix


def excitation(f0: np.ndarray, n: int, disp: str = "none",
               disp_c: float = 0.0) -> torch.Tensor:
    f0t = torch.from_numpy(f0.astype(np.float32)).unsqueeze(0)
    src = HarmonicSource(causal=True, disp=disp, disp_c=disp_c)
    e, _ = src(f0t)
    e = e[:, :n]
    voiced = (f0t > 1.0).float().unsqueeze(1)
    vs = torch.nn.functional.interpolate(voiced, size=n, mode="nearest").squeeze(1)
    nz = 0.05 + 0.5 * (1.0 - vs)
    e = e + torch.randn(1, n) * nz
    return e


def render(uid: str, path: Path, k: int, order_scale: float,
           clamp_nats: float) -> None:
    y = load_wav(path)
    S = librosa.stft(y.astype(np.float32), n_fft=NFFT, hop_length=HOP,
                     win_length=NFFT, center=True)
    T = S.shape[1]
    n = T * HOP
    f0 = gt_f0(y, T)
    e = excitation(f0, n)
    ed = excitation(f0, n, disp="hfrand", disp_c=2000.0)
    b_min, b_mix = build_firs(S, f0, k, order_scale, clamp_nats)
    bm = torch.from_numpy(b_min).unsqueeze(0)
    bx = torch.from_numpy(b_mix).unsqueeze(0)
    y_min = ltv_ola(e, bm, hop=HOP, backend="mm")[0].numpy()
    y_mix = ltv_ola(e, bx, hop=HOP, backend="mm")[0].numpy()
    y_mixd = ltv_ola(ed, bx, hop=HOP, backend="mm")[0].numpy()
    y_ist = librosa.istft(S, hop_length=HOP, win_length=NFFT, center=True,
                          length=len(y))

    gt_rms = np.sqrt((y ** 2).mean()) + EPS

    def match(a: np.ndarray) -> np.ndarray:
        a = a * (gt_rms / (np.sqrt((a ** 2).mean()) + EPS))
        p = np.abs(a).max() + EPS
        if p > 0.98:
            a = a * (0.98 / p)
        return a.astype(np.float32)

    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(OUT / f"{uid}_gt.wav", match(y), SR)
    sf.write(OUT / f"{uid}_istft.wav", match(y_ist), SR)
    sf.write(OUT / f"{uid}_min.wav", match(y_min), SR)
    sf.write(OUT / f"{uid}_mix.wav", match(y_mix), SR)
    sf.write(OUT / f"{uid}_mixd.wav", match(y_mixd), SR)
    print(f"  {uid}: T={T} rms min={np.sqrt((y_min**2).mean()):.3f} "
          f"mix={np.sqrt((y_mix**2).mean()):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--order-scale", type=float, default=1.0)
    ap.add_argument("--clamp-nats", type=float, default=9.0)
    args = ap.parse_args()

    print("R-oracle mixed-phase: min-phase(FAIL anchor) vs GT mixed-phase")
    print(f"  arms per utt: gt, istft(PASS anchor), min, mix | k={args.k} "
          f"order_scale={args.order_scale}")
    if args.files:
        items = [(Path(f).stem, Path(f)) for f in args.files]
    else:
        tri = ROOT / "results/e2_triage"
        items = [(u, tri / f"{u}_gt.wav") for u in DEFAULT_UIDS]
    for uid, path in items:
        if not path.exists():
            print(f"  {uid}: MISSING {path}")
            continue
        render(uid, path, args.k, args.order_scale, args.clamp_nats)
    print(f"listen: uv run python listen_gui.py --dir {OUT} --port 8773")


if __name__ == "__main__":
    main()
