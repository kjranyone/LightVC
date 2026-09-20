"""Product ear comparison: gt / w1(causal direct-wave) / freec_B(causal, the
SHIPPABLE freeC) / freecF(centered, non-shippable reference). Each vocoder renders
with its own trained mel. Judge=ear: does w1's clean direct-waveform beat freec_B's
mid muddiness (contrast is tied ~2.8, but proxy misses muddiness)?
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
from vocoder_a import WaveVocoderA


def gmatch(y, gt):
    n = min(len(y), len(gt)); y, gt = y[:n], gt[:n]
    g = np.sqrt((gt ** 2).mean() / ((y ** 2).mean() + 1e-9))
    return np.clip(y * g, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", default="checkpoints/vocoderA_w1/w1_dim256_ref.pt")
    ap.add_argument("--fb", default="checkpoints/freec_b512_full/last.pt")
    ap.add_argument("--ff", default="checkpoints/freeC/foundation_lowlatency_5p8ms.pt")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--held", default="../results/interpretable_vc/held_multispk.txt")
    ap.add_argument("--out", default="../results/interpretable_vc/ear_product")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    W1 = WaveVocoderA(dim=256, causal=True).to(DEV).eval()
    ck = torch.load(args.w1, map_location=DEV, weights_only=False)
    W1.load_state_dict({k: v.to(next(W1.parameters()).dtype) for k, v in ck["gen"].items()})
    FB = load_freebig(args.fb)[0]        # freec_B (causal, product)
    FF = load_freebig(args.ff)[0]        # freecF (centered, reference)
    held = sorted(Path(x.strip()) for x in open(args.held) if x.strip())
    seen, picks = set(), []
    for f in held:
        d = torch.load(f, weights_only=False); spk = Path(d["path"]).parent.name
        if spk in seen:
            continue
        seen.add(spk); picks.append((spk, d))
        if len(picks) >= args.k:
            break
    for spk, d in picks:
        w = load_wav(d["path"]); gt = torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0).to(DEV)
        with torch.no_grad():
            yw = W1(causal_mel(gt, n_fft=2048, hop=128)).squeeze().cpu().numpy()
            yb = FB(causal_mel(gt, n_fft=512, hop=128)).squeeze().cpu().numpy()
            yf = FF(centered_mel(gt, n_fft=2048, hop=128)).squeeze().cpu().numpy()
        s = spk[:8]
        sf.write(out / f"{s}_0gt.wav", np.clip(w, -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"{s}_1w1_causal.wav", gmatch(yw, w), SR, subtype="PCM_16")
        sf.write(out / f"{s}_2freecB_causal_SHIPPABLE.wav", gmatch(yb, w), SR, subtype="PCM_16")
        sf.write(out / f"{s}_3freecF_centered_ref.wav", gmatch(yf, w), SR, subtype="PCM_16")
    print(f"product ear -> {out} | speakers {[s[:8] for s,_ in picks]}", flush=True)


if __name__ == "__main__":
    main()
