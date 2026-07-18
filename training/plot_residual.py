"""Time-domain truth: since freeuniv/bigvgan reconstruct from gt's mel they are
sample-aligned to gt (no CC align). Look at few-period waveform overlay and the
RESIDUAL (arm - gt) in time and in the STFT domain -> shows exactly where each
arm departs from gt.
"""
from __future__ import annotations
import sys
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

TRI = Path(__file__).resolve().parent.parent / "results/e2_triage"
SR, NFFT, HOP = 44100, 2048, 512
UID = sys.argv[1] if len(sys.argv) > 1 else "af1ad5575a3fa383_00035936"
OUT = Path("/tmp/claude-1000/-home-kojirotanaka-kjranyone-LightVC/2ed9836e-e8de-4c00-b91a-ee2ae45093f0/scratchpad/resid.png")


def load(a):
    y, _ = librosa.load(str(TRI / f"{UID}_{a}.wav"), sr=SR)
    return y


def main():
    arms = ["gt", "freeuniv", "bigvgan"]
    ys = {a: load(a) for a in arms}
    n = min(len(v) for v in ys.values())
    gr = np.sqrt((ys["gt"][:n] ** 2).mean()) + 1e-9
    for a in ys:
        y = ys[a][:n]
        ys[a] = y * (gr / (np.sqrt((y ** 2).mean()) + 1e-9))
    g = ys["gt"]
    w = int(0.5 * SR)
    e = np.array([g[i:i + w].std() for i in range(0, n - w, HOP)])
    c = int(np.argmax(e) * HOP) + w // 2

    fig, ax = plt.subplots(3, 2, figsize=(16, 10))
    # 6 ms overlay (few pitch periods)
    z0, z1 = c, c + int(0.006 * SR)
    t = np.arange(z0, z1) / SR * 1000
    ax[0, 0].plot(t, g[z0:z1], "k", lw=1.2, label="gt")
    ax[0, 0].plot(t, ys["freeuniv"][z0:z1], "C1", lw=1.0, label="freeuniv")
    ax[0, 0].set_title("6ms overlay: gt vs freeuniv"); ax[0, 0].legend()
    ax[0, 1].plot(t, g[z0:z1], "k", lw=1.2, label="gt")
    ax[0, 1].plot(t, ys["bigvgan"][z0:z1], "C2", lw=1.0, label="bigvgan")
    ax[0, 1].set_title("6ms overlay: gt vs bigvgan"); ax[0, 1].legend()
    # residual waveform over 30ms
    z1b = c + int(0.03 * SR)
    tb = np.arange(c, z1b) / SR * 1000
    for j, a in enumerate(["freeuniv", "bigvgan"]):
        r = ys[a][c:z1b] - g[c:z1b]
        ax[1, j].plot(tb, g[c:z1b], "k", lw=0.5, alpha=0.4, label="gt")
        ax[1, j].plot(tb, r, "C3", lw=0.8, label=f"{a}-gt residual")
        rms_r = np.sqrt((ys[a][:n] - g) ** 2).mean() ** 0.5
        ax[1, j].set_title(f"{a}: residual  (whole-utt resid-RMS/gt-RMS = "
                           f"{np.sqrt(((ys[a][:n]-g)**2).mean())/gr:.2f})")
        ax[1, j].legend()
    # residual STFT (arm - gt) magnitude-of-complex-difference, dB
    G = librosa.stft(g, n_fft=NFFT, hop_length=HOP)
    for j, a in enumerate(["freeuniv", "bigvgan"]):
        A = librosa.stft(ys[a][:n], n_fft=NFFT, hop_length=HOP)
        R = librosa.amplitude_to_db(np.abs(A - G) + 1e-6)
        im = ax[2, j].imshow(R, origin="lower", aspect="auto", cmap="viridis",
                             extent=[0, n / SR, 0, SR / 2000], vmin=-60, vmax=10)
        ax[2, j].set_title(f"complex STFT residual |{a}-gt| (dB)")
        ax[2, j].set_ylabel("kHz"); ax[2, j].set_xlabel("s")
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=90)
    # numeric: banded complex-residual energy ratio vs gt energy
    print(f"center {c/SR:.2f}s")
    fb = np.linspace(0, SR / 2, G.shape[0])
    for a in ["freeuniv", "bigvgan"]:
        A = librosa.stft(ys[a][:n], n_fft=NFFT, hop_length=HOP)
        print(f"  {a}:")
        for lo, hi in [(0, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, 22050)]:
            m = (fb >= lo) & (fb < hi)
            num = (np.abs(A[m] - G[m]) ** 2).sum()
            den = (np.abs(G[m]) ** 2).sum() + 1e-9
            print(f"    {lo//1000:2d}-{hi//1000:2d}kHz  resid/gt = {np.sqrt(num/den):.2f}")


if __name__ == "__main__":
    main()
