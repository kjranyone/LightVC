"""SEE the s_art effect (I can't hear). Overlay z3neu vs z3moe vs gt: spectrogram
0-4kHz (formant bands), LTAS envelope (F1 rise? F2 concentrate?), centroid drag
(timbre disentangle check)."""
from __future__ import annotations
import numpy as np, librosa, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path

D = Path("/home/kojirotanaka/kjranyone/LightVC/results/z3")
OUT = Path("/tmp/claude-1000/-home-kojirotanaka-kjranyone-LightVC/2ed9836e-e8de-4c00-b91a-ee2ae45093f0/scratchpad/z3.png")
SR, NFFT, HOP = 44100, 2048, 512
stems = sorted({"_".join(Path(p).name.split("_")[:-1]) for p in glob.glob(str(D/"*_z3neu.wav"))})[:4]
ARMS = ["gt", "z3neu", "z3moe"]

def load(stem, a):
    y, _ = librosa.load(str(D/f"{stem}_{a}.wav"), sr=SR); return y

# LTAS averaged over the 4 utts (voiced), per arm
ltas = {a: [] for a in ARMS}
cent = {a: [] for a in ARMS}
for stem in stems:
    ys = {a: load(stem, a) for a in ARMS}
    n = min(len(v) for v in ys.values())
    gr = np.sqrt((ys["gt"][:n]**2).mean()) + 1e-9
    for a in ARMS:
        y = ys[a][:n] * (gr / (np.sqrt((ys[a][:n]**2).mean())+1e-9))
        S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP)) + 1e-6
        v = S.mean(0) > np.percentile(S.mean(0), 40)  # voiced-ish frames
        ltas[a].append(np.log(S[:, v]).mean(1))
        f = np.linspace(0, SR/2, S.shape[0])
        cent[a].append((S[:, v]*f[:,None]).sum(0).mean() / (S[:, v].sum(0).mean()+1e-9))

freqs = np.linspace(0, SR/2000, NFFT//2+1)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
k = 20/np.log(10)
for a, col in zip(ARMS, ["k", "C0", "C3"]):
    m = np.mean(ltas[a], 0) * k
    ax[0].plot(freqs, m, col, lw=1.4, label=f"{a} (centroid {np.mean(cent[a]):.0f}Hz)")
ax[0].set_xlim(0, 4); ax[0].set_title("LTAS 0-4kHz (F1~0.3-0.9k: moe=F1↑ ; F2~1-2.5k spread)")
ax[0].set_xlabel("kHz"); ax[0].set_ylabel("dB"); ax[0].legend(); ax[0].grid(alpha=0.3)
# difference z3moe - z3neu (what s_art moved)
dm = (np.mean(ltas["z3moe"],0) - np.mean(ltas["z3neu"],0)) * k
ax[1].plot(freqs, dm, "C2", lw=1.4); ax[1].axhline(0, color="gray", lw=0.5)
ax[1].set_xlim(0, 6); ax[1].set_title("z3moe - z3neu envelope diff (s_art moe effect)")
ax[1].set_xlabel("kHz"); ax[1].set_ylabel("dB"); ax[1].grid(alpha=0.3)
plt.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True); plt.savefig(OUT, dpi=90)
print(f"centroid drag z3neu->z3moe: {np.mean(cent['z3neu']):.0f} -> {np.mean(cent['z3moe']):.0f} Hz "
      f"(+{np.mean(cent['z3moe'])-np.mean(cent['z3neu']):.0f}); gt {np.mean(cent['gt']):.0f}")
print(f"saved {OUT}")
