"""Ear A/B/X for Vocoder A vs freeC: same held utterances through
gt / vocoderA(causal mel 2048/hop128/look0) / freecF(centered mel 2048/hop128).
Gain-matched. Judge = ear (mid-range muddiness): does vocoderA beat freeC?
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
from vocoder_a import FreeVocoderA


def gmatch(y, gt):
    n = min(len(y), len(gt)); y, gt = y[:n], gt[:n]
    g = np.sqrt((gt ** 2).mean() / ((y ** 2).mean() + 1e-9))
    return np.clip(y * g, -1, 1)


def load_vfa(path, dim=256):
    net = FreeVocoderA(dim=dim, causal=True).to(DEV).eval()
    ck = torch.load(path, map_location=DEV, weights_only=False)
    net.load_state_dict({k: v.to(next(net.parameters()).dtype) for k, v in ck["gen"].items()})
    return net, ck.get("step", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vfa", default="checkpoints/vocoderA_v10/last.pt")
    ap.add_argument("--vf", default="checkpoints/freeC/foundation_lowlatency_5p8ms.pt")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--held", default="../results/interpretable_vc/held_multispk.txt")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="../results/interpretable_vc/ear_vocoderA_v10")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    VFA, step = load_vfa(args.vfa, args.dim)
    VF = load_freebig(args.vf)[0]
    held = sorted(Path(x.strip()) for x in open(args.held) if x.strip())
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
        with torch.no_grad():
            ya = VFA(causal_mel(gt, n_fft=2048, hop=128)).squeeze().cpu().numpy()
            yf = VF(centered_mel(gt, n_fft=2048, hop=128)).squeeze().cpu().numpy()
        sf.write(out / f"{spk[:8]}_0gt.wav", np.clip(w, -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"{spk[:8]}_1vocoderA_v10.wav", gmatch(ya, w), SR, subtype="PCM_16")
        sf.write(out / f"{spk[:8]}_2freecF.wav", gmatch(yf, w), SR, subtype="PCM_16")
    print(f"ear A/B/X (step {step}) -> {out} | speakers {[s[:8] for s,_ in picks]}", flush=True)


if __name__ == "__main__":
    main()
