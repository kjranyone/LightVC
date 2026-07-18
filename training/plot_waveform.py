"""Actually LOOK: render waveform (time domain) + spectrogram + single-frame
spectrum for gt vs freeuniv vs bigvgan, save PNG to inspect visually.
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
ARMS = ["gt", "freeuniv", "bigvgan"]
OUT = Path("/tmp/claude-1000/-home-kojirotanaka-kjranyone-LightVC/2ed9836e-e8de-4c00-b91a-ee2ae45093f0/scratchpad/wave.png")


def load(a):
    y, _ = librosa.load(str(TRI / f"{UID}_{a}.wav"), sr=SR)
    return y


def main():
    ys = {a: load(a) for a in ARMS}
    n = min(len(v) for v in ys.values())
    gr = np.sqrt((ys["gt"][:n] ** 2).mean()) + 1e-9
    for a in ys:
        y = ys[a][:n]
        ys[a] = y * (gr / (np.sqrt((y ** 2).mean()) + 1e-9))

    # pick a strongly voiced window (max short-time energy)
    w = int(0.5 * SR)
    e = np.array([ys["gt"][i:i + w].std() for i in range(0, n - w, HOP)])
    c = int(np.argmax(e) * HOP) + w // 2
    z0, z1 = c, min(c + int(0.03 * SR), n)  # 30 ms zoom

    fig, ax = plt.subplots(4, 3, figsize=(16, 12))
    for j, a in enumerate(ARMS):
        y = ys[a]
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP)) + 1e-6)
        ax[0, j].imshow(S, origin="lower", aspect="auto", cmap="magma",
                        extent=[0, n / SR, 0, SR / 2000], vmin=-60, vmax=20)
        ax[0, j].set_title(f"{a}  spectrogram (dB)")
        ax[0, j].set_ylabel("kHz")
        ax[1, j].plot(np.arange(z0, z1) / SR, y[z0:z1], lw=0.6)
        ax[1, j].set_title(f"{a}  waveform 30ms voiced")
        ax[1, j].set_ylim(-np.abs(ys["gt"][z0:z1]).max() * 1.5,
                          np.abs(ys["gt"][z0:z1]).max() * 1.5)
        # HF-only waveform (>4kHz) to see the HF floor/haze directly
        yhf = librosa.effects.preemphasis(y)  # rough HF emphasis
        b = librosa.stft(y, n_fft=NFFT, hop_length=HOP)
        f = np.linspace(0, SR / 2, b.shape[0])
        b[f < 4000] = 0
        yh = librosa.istft(b, hop_length=HOP, length=n)
        ax[2, j].plot(np.arange(z0, z1) / SR, yh[z0:z1], lw=0.6, color="C3")
        ax[2, j].set_title(f"{a}  >4kHz only, 30ms")
        ax[2, j].set_ylim(-np.abs(yh[z0:z1]).max() * 1.5 - 1e-6,
                          np.abs(yh[z0:z1]).max() * 1.5 + 1e-6)

    # single-frame spectrum overlay at center
    fr = c // HOP
    for a in ARMS:
        S = np.abs(librosa.stft(ys[a], n_fft=NFFT, hop_length=HOP)) + 1e-7
        f = np.linspace(0, SR / 2000, S.shape[0])
        ax[3, 0].plot(f, librosa.amplitude_to_db(S[:, fr]), lw=0.7, label=a)
    ax[3, 0].set_title("voiced-frame spectrum"); ax[3, 0].legend()
    ax[3, 0].set_xlabel("kHz"); ax[3, 0].set_ylabel("dB")
    # zoom HF spectrum 4-16kHz
    for a in ARMS:
        S = np.abs(librosa.stft(ys[a], n_fft=NFFT, hop_length=HOP)) + 1e-7
        f = np.linspace(0, SR / 2000, S.shape[0])
        ax[3, 1].plot(f, librosa.amplitude_to_db(S[:, fr]), lw=0.7, label=a)
    ax[3, 1].set_xlim(4, 16); ax[3, 1].set_ylim(-70, 0)
    ax[3, 1].set_title("voiced-frame spectrum 4-16kHz zoom"); ax[3, 1].set_xlabel("kHz")
    # unvoiced/gap frame spectrum (min energy frame)
    fr2 = int(np.argmin(e))
    for a in ARMS:
        S = np.abs(librosa.stft(ys[a], n_fft=NFFT, hop_length=HOP)) + 1e-7
        f = np.linspace(0, SR / 2000, S.shape[0])
        ax[3, 2].plot(f, librosa.amplitude_to_db(S[:, min(fr2, S.shape[1]-1)]), lw=0.7, label=a)
    ax[3, 2].set_title("low-energy (gap) frame spectrum"); ax[3, 2].set_xlabel("kHz")
    ax[3, 2].legend()

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=90)
    print(f"saved {OUT}  voiced center {c/SR:.2f}s")


if __name__ == "__main__":
    main()
