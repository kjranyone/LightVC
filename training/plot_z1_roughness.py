"""LOOK at the roughness (I can't hear; use vision). Compare gt/ceiling/z1/z1gan
spectrograms + a voiced-frame spectrum (harmonic-vs-noise floor) + waveform zoom
to SEE the roughness signature (inter-harmonic noise, HF grain, spectral flicker)."""
from __future__ import annotations
import numpy as np, librosa, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path

D = Path("/home/kojirotanaka/kjranyone/LightVC/results/z1")
STEM = "04_af1ad5575a3fa383_af1ad5575a3fa383_00032753"
ARMS = ["gt", "ceiling", "z1", "z1gan"]
SR, NFFT, HOP = 44100, 2048, 512
OUT = Path("/tmp/claude-1000/-home-kojirotanaka-kjranyone-LightVC/2ed9836e-e8de-4c00-b91a-ee2ae45093f0/scratchpad/z1_rough.png")

ys = {}
for a in ARMS:
    y, _ = librosa.load(str(D / f"{STEM}_{a}.wav"), sr=SR)
    ys[a] = y
n = min(len(v) for v in ys.values())
gr = np.sqrt((ys["gt"][:n] ** 2).mean()) + 1e-9
for a in ys:
    y = ys[a][:n]; ys[a] = y * (gr / (np.sqrt((y ** 2).mean()) + 1e-9))

# pick a strongly voiced window
w = int(0.4 * SR)
e = np.array([ys["gt"][i:i+w].std() for i in range(0, n - w, HOP)])
c = int(np.argmax(e) * HOP) + w // 2
fr = c // HOP

fig, ax = plt.subplots(3, 4, figsize=(20, 11))
for j, a in enumerate(ARMS):
    S = librosa.amplitude_to_db(np.abs(librosa.stft(ys[a], n_fft=NFFT, hop_length=HOP)) + 1e-6)
    ax[0, j].imshow(S, origin="lower", aspect="auto", cmap="magma",
                    extent=[0, n/SR, 0, SR/2000], vmin=-55, vmax=15)
    ax[0, j].set_title(f"{a}  spectrogram"); ax[0, j].set_ylabel("kHz")
    # voiced-frame spectrum 0-8kHz (harmonic peaks vs inter-harmonic noise floor)
    sp = np.abs(librosa.stft(ys[a], n_fft=NFFT, hop_length=HOP)) + 1e-7
    f = np.linspace(0, SR/2000, sp.shape[0])
    ax[1, j].plot(f, librosa.amplitude_to_db(sp[:, fr]), lw=0.6, color="C0")
    ax[1, j].set_xlim(0, 6); ax[1, j].set_ylim(-60, 15)
    ax[1, j].set_title(f"{a}  voiced spectrum 0-6kHz"); ax[1, j].set_xlabel("kHz")
    # waveform zoom 12ms
    z0 = c; z1 = c + int(0.012 * SR)
    ax[2, j].plot(np.arange(z0, z1)/SR*1000, ys[a][z0:z1], lw=0.7, color="C3")
    ax[2, j].set_title(f"{a}  waveform 12ms"); ax[2, j].set_xlabel("ms")
    ax[2, j].set_ylim(-np.abs(ys["gt"][z0:z1]).max()*1.6, np.abs(ys["gt"][z0:z1]).max()*1.6)
plt.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True); plt.savefig(OUT, dpi=85)
# numeric: inter-harmonic noise floor (voiced), HF energy
print("=== objective (harmonic peak-to-valley = higher=cleaner; less=rougher) ===")
import pyworld as pw
f0, t = pw.harvest(ys["gt"].astype(np.float64), SR, frame_period=1000*HOP/SR)
for a in ARMS:
    sp = np.abs(librosa.stft(ys[a], n_fft=NFFT, hop_length=HOP)) + 1e-7
    df = SR / NFFT; hv, vv = [], []
    for ti in range(min(sp.shape[1], len(f0))):
        ff = f0[ti]
        if ff < 80 or ff > 400: continue
        for k in range(2, int(5000/ff)):
            hb, vb = int(round(k*ff/df)), int(round((k+0.5)*ff/df))
            if vb < sp.shape[0]:
                hv.append(20*np.log10(sp[hb, ti])); vv.append(20*np.log10(sp[vb, ti]))
    print(f"  {a:8} harm {np.mean(hv):6.1f}dB valley {np.mean(vv):6.1f}dB  contrast {np.mean(hv)-np.mean(vv):5.1f}dB")
print(f"saved {OUT}")
