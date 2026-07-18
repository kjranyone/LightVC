"""Look at the ACTUAL freeuniv output spectrum vs gt vs bigvgan, to verify (not
assume) the claimed deficit (HF under-production) and excess (inter-harmonic
valley filling). Renders already exist in results/e2_triage.
"""
from __future__ import annotations
import numpy as np
import librosa
from pathlib import Path

TRI = Path(__file__).resolve().parent.parent / "results/e2_triage"
SR, NFFT, HOP = 44100, 2048, 512
DF = SR / NFFT
UIDS = ["af1ad5575a3fa383_00035936", "af1ad5575a3fa383_00027042",
        "af1ad5575a3fa383_00040870", "af1ad5575a3fa383_00036578"]
ARMS = ["gt", "freeuniv", "bigvgan", "freetgt2"]
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000),
         (6000, 8000), (8000, 12000), (12000, 16000)]


def logspec(y):
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP)) + 1e-7
    return np.log(S)  # nats; ×(20/ln10) for dB later


def f0_track(y, T):
    import pyworld as pw
    f0, t = pw.harvest(y.astype(np.float64), SR, frame_period=1000 * HOP / SR)
    f0 = pw.stonemask(y.astype(np.float64), f0, t, SR)
    ft = np.arange(T) * HOP / SR
    return np.interp(ft, t, f0)


def main():
    freqs = np.arange(NFFT // 2 + 1) * DF
    ltas = {a: [] for a in ARMS}
    hv = {a: {"harm": [], "val": []} for a in ARMS}
    for uid in UIDS:
        ys = {}
        for a in ARMS:
            p = TRI / f"{uid}_{a}.wav"
            if not p.exists():
                continue
            ys[a], _ = librosa.load(str(p), sr=SR)
        if "gt" not in ys:
            continue
        n = min(len(v) for v in ys.values())
        # RMS-match to gt so LTAS compares SHAPE not level
        gr = np.sqrt((ys["gt"][:n] ** 2).mean()) + 1e-9
        specs = {}
        for a, y in ys.items():
            y = y[:n] * (gr / (np.sqrt((y[:n] ** 2).mean()) + 1e-9))
            specs[a] = logspec(y)
        T = min(s.shape[1] for s in specs.values())
        f0 = f0_track(ys["gt"][:n], T)
        for a in specs:
            L = specs[a][:, :T]
            ltas[a].append(L.mean(1))
            for ti in range(T):
                f = f0[ti]
                if f < 80 or f > 400:
                    continue
                for k in range(3, int(6000 / f)):  # 2-6kHz-ish harmonics
                    hb = int(round(k * f / DF))
                    vb = int(round((k + 0.5) * f / DF))
                    if vb < L.shape[0]:
                        hv[a]["harm"].append(L[hb, ti])
                        hv[a]["val"].append(L[vb, ti])

    k = 20.0 / np.log(10)  # nats -> dB
    print("=== LTAS (dB, RMS-matched to gt), and Δ vs gt ===")
    print(f"{'band':>12} " + " ".join(f"{a:>9}" for a in ARMS))
    gt_ltas = np.mean(ltas["gt"], 0)
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        row = f"{lo//1000}-{hi//1000}kHz".rjust(12) + " "
        for a in ARMS:
            if not ltas[a]:
                row += f"{'--':>9} "
                continue
            v = np.mean(ltas[a], 0)[m].mean() * k
            g = gt_ltas[m].mean() * k
            row += (f"{v-g:+8.1f}d" if a != "gt" else f"{v:8.1f}d") + " "
        print(row)
    print("\n=== harmonic peak-to-valley contrast (dB, voiced 2-6kHz; higher=cleaner) ===")
    for a in ARMS:
        if not hv[a]["harm"]:
            continue
        h = np.mean(hv[a]["harm"]) * k
        v = np.mean(hv[a]["val"]) * k
        print(f"  {a:9} harm {h:7.1f}dB  valley {v:7.1f}dB  contrast {h-v:6.1f}dB")


if __name__ == "__main__":
    main()
