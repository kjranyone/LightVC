"""Z0-C ear gate render: same held utterances through gt / freec_F(centered) /
freec_b(causal, per window). Gain-matched. The ACTUAL Z0-C decision is the ear
(does causal path B degrade vs centered path A?); harmonic-sharp is only a guide.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np, torch, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import SR, DEV
from train_z1 import load_freebig
from f0leak_probe import load_wav
from causal_mel import causal_mel, centered_mel


def gmatch(y, gt):
    n = min(len(y), len(gt)); y, gt = y[:n], gt[:n]
    g = np.sqrt((gt ** 2).mean() / ((y ** 2).mean() + 1e-9))
    return np.clip(y * g, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vf", default="checkpoints/freeC/foundation_lowlatency_5p8ms.pt")
    ap.add_argument("--windows", nargs="+", default=["512", "1024", "2048"])
    ap.add_argument("--held", default="../results/interpretable_vc/held_multispk.txt")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default="../results/interpretable_vc/ear_z0c")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    VF = load_freebig(args.vf)[0]
    VB = {}
    for w in args.windows:
        p = Path(f"checkpoints/freec_b_nfft{w}/last.pt")
        if p.exists():
            VB[w] = load_freebig(str(p))[0]
    held = sorted(Path(x.strip()) for x in open(args.held) if x.strip())
    # one utt per distinct speaker, first k
    seen, picks = set(), []
    for f in held:
        d = torch.load(f, weights_only=False)
        spk = Path(d["path"]).parent.name
        if spk in seen:
            continue
        seen.add(spk); picks.append((spk, d))
        if len(picks) >= args.k:
            break

    for spk, d in picks:
        w = load_wav(d["path"]); gt = torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0).to(DEV)
        sf.write(out / f"{spk[:8]}_0gt.wav", np.clip(w, -1, 1), SR, subtype="PCM_16")
        with torch.no_grad():
            yf = VF(centered_mel(gt, n_fft=2048, hop=128)).squeeze().cpu().numpy()
        sf.write(out / f"{spk[:8]}_1freecF_centered.wav", gmatch(yf, w), SR, subtype="PCM_16")
        for wnd, V in VB.items():
            with torch.no_grad():
                yb = V(causal_mel(gt, n_fft=int(wnd), hop=128)).squeeze().cpu().numpy()
            sf.write(out / f"{spk[:8]}_2causal_nfft{wnd}.wav", gmatch(yb, w), SR, subtype="PCM_16")
    print(f"ear renders -> {out} | speakers {[s[:8] for s,_ in picks]} | windows {list(VB)}")


if __name__ == "__main__":
    main()
