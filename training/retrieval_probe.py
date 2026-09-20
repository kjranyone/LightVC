"""Retrieval sharpness premise-test (non-parametric pivot). For a held target
speaker: source utt X, pool = the speaker's OTHER utts. For each source frame,
kNN-match by ContentVec against pool frames, retrieve the REAL mel of the matched
pool frame(s) (mean over k), vocode with freebig. If retrieval output harmonic
sharpness ~ gt/ceiling (>> parametric-G self ~2.81), the premise holds: retrieval
supplies sharpness that regression cannot (no conditional-mean blur). k=1 = pure
real frame (upper sharpness), k>1 = averaged (coherence vs sharpness trade)."""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F, soundfile as sf, librosa
sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, DEV
from train_z1 import load_freebig
from f0leak_probe import load_wav

SRC_N = 5


def frames(d):
    c = d["content"].float()
    w = load_wav(d["path"])
    m = mel_of(torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0).to(DEV))[0]  # [128, T]
    T = m.shape[-1]
    cf = F.interpolate(c.t().unsqueeze(0), size=T, mode="linear", align_corners=False)[0].t().to(DEV)  # [T,768]
    return cf, m.t(), w[: T * HOP]  # content[T,768], mel[T,128], wav


def sharp(y):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) + 1e-7
    f = np.linspace(0, SR / 2, S.shape[0]); v = S.mean(0) > np.percentile(S.mean(0), 50)
    band = (f >= 500) & (f <= 3000); sub = np.log(S[band][:, v])
    return float((np.percentile(sub, 90, 0) - np.percentile(sub, 10, 0)).mean())


def main():
    voc, _ = load_freebig("checkpoints/freebig/foundation_bigvgan_parity.pt")
    held = sorted(Path(x.strip()) for x in open("/tmp/e5eval_held.txt") if x.strip())
    by = defaultdict(list)
    for f in held:
        by[f.parent.name].append(f)
    spk = [s for s in by if len(by[s]) >= 4][:SRC_N]
    out = Path("../results/retr"); out.mkdir(parents=True, exist_ok=True)
    R = {"gt": [], "ceiling": [], "retr_k1": [], "retr_k4": []}
    for i, s in enumerate(spk):
        fs = by[s]
        ds = torch.load(fs[0], weights_only=False)
        if "content" not in ds:
            continue
        cs, ms, ws = frames(ds)                       # source
        pc, pm = [], []
        for pf in fs[1:]:
            dp = torch.load(pf, weights_only=False)
            cf, mf, _ = frames(dp); pc.append(cf); pm.append(mf)
        pc = torch.cat(pc, 0); pm = torch.cat(pm, 0)  # pool [Np,768], [Np,128]
        csn = F.normalize(cs, dim=-1); pcn = F.normalize(pc, dim=-1)
        sim = csn @ pcn.t()                           # [Ts, Np]
        with torch.no_grad():
            for k, tag in [(1, "retr_k1"), (4, "retr_k4")]:
                idx = sim.topk(k, dim=-1).indices           # [Ts,k]
                rm = pm[idx].mean(1)                         # [Ts,128] retrieved mel
                y = voc(rm.t().unsqueeze(0))[0].cpu().numpy()
                R[tag].append(sharp(y))
                if i < 3:
                    sf.write(out / f"{i:02d}_{s[:8]}_{tag}.wav", np.clip(y, -1, 1), SR, subtype="PCM_16")
            yc = voc(ms.t().unsqueeze(0))[0].cpu().numpy()
        R["ceiling"].append(sharp(yc)); R["gt"].append(sharp(np.asarray(ws)))
        if i < 3:
            sf.write(out / f"{i:02d}_{s[:8]}_gt.wav", np.clip(np.asarray(ws), -1, 1), SR, subtype="PCM_16")
            sf.write(out / f"{i:02d}_{s[:8]}_ceiling.wav", np.clip(yc, -1, 1), SR, subtype="PCM_16")
    print("harmonic sharpness (higher=sharper), n=", len(R["gt"]))
    for k in ["gt", "ceiling", "retr_k1", "retr_k4"]:
        print(f"  {k:10s} {np.mean(R[k]):.2f}")
    print("  [parametric-G self ~2.81 参照]  wavs ->", out)


if __name__ == "__main__":
    main()
