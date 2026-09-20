"""Stable resolution diagnostic (no churn): female self-recon (has gt), N=20+
utts averaged, two independent sharpness metrics. Decomposes where fine detail is
lost: gt -> ceiling(FreeC on real mel) = FreeC realtime cost ; ceiling -> self
(parametric G) = G's mel. Fixed held set, deterministic, averaged -> trustworthy
numbers to base a single lever on."""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, librosa
sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, DEV
from train_m2 import TimbreEncoder, TIMBRE_DIM, ART_DIM
from mel_gen import MelGen
from train_z1 import load_freebig
from f0leak_probe import load_wav, build_cond


def metrics(y):
    S = np.abs(librosa.stft(y.astype(np.float64), n_fft=2048, hop_length=512)) + 1e-7
    f = np.linspace(0, SR / 2, S.shape[0]); v = S.mean(0) > np.percentile(S.mean(0), 60)
    b = (f >= 500) & (f <= 3000); sub = np.log(S[b][:, v])
    sharp = float((np.percentile(sub, 90, 0) - np.percentile(sub, 10, 0)).mean())   # harmonic def
    hf = (f >= 4000) & (f <= 12000); lf = (f >= 300) & (f <= 3000)
    L = np.log(S[:, v]).mean(1)
    hfr = float(L[hf].mean() - L[lf].mean())                                          # HF presence
    return sharp, hfr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", default="checkpoints/e6/g_24000.pt")
    ap.add_argument("--v", default="checkpoints/e6/snap_24000.pt")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    gck = torch.load(args.g, map_location=DEV, weights_only=False); ga = gck["args"]
    g = MelGen(cond_dim=770, dim=ga["dim"], n_layers=ga["layers"], timbre_dim=TIMBRE_DIM,
               art_dim=ART_DIM, f0_fourier=ga.get("f0_fourier", 0)).to(DEV)
    g.load_state_dict(gck["g"]); g.eval()
    t = TimbreEncoder().to(DEV); t.load_state_dict(gck["t"]); t.eval()
    vE = load_freebig(args.v)[0]; vF = load_freebig("checkpoints/freebig/foundation_bigvgan_parity.pt")[0]
    up = gck.get("upsample", 4)
    held = sorted(Path(x.strip()) for x in open("/tmp/e5eval_held.txt") if x.strip())
    R = {"gt": [], "ceiling": [], "self": []}
    done = 0
    for f in held:
        if done >= args.n:
            break
        d = torch.load(f, weights_only=False)
        if "f0" not in d:
            continue
        cond, tmel = build_cond(d); cond = cond.unsqueeze(0).to(DEV)
        w = load_wav(d["path"])[: tmel * HOP]
        gt = torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0).to(DEV)
        with torch.no_grad():
            s = t(mel_of(gt[:, : int(3 * SR)]))
            cond4 = F.interpolate(cond, scale_factor=up, mode="linear", align_corners=False)
            y_self = vE(g(cond4, s, None)).squeeze().cpu().numpy()
            y_ceil = vF(mel_of(gt))[0].cpu().numpy()
        for tag, y in [("gt", np.asarray(w)), ("ceiling", y_ceil), ("self", y_self)]:
            R[tag].append(metrics(y))
        done += 1
    print(f"stable resolution eval | n={done} female self-recon | E6 G {args.g.split('/')[-1]}")
    print(f"{'arm':9s} {'harmonic-sharp':>15s} {'HF-presence':>12s}")
    for k in ["gt", "ceiling", "self"]:
        a = np.array(R[k]); print(f"{k:9s} {a[:,0].mean():8.2f}±{a[:,0].std():.2f}    {a[:,1].mean():+8.2f}")
    gt, ce, se = np.array(R["gt"]), np.array(R["ceiling"]), np.array(R["self"])
    print(f"loss: gt->ceiling(FreeC) {gt[:,0].mean()-ce[:,0].mean():+.2f}sharp | ceiling->self(G-mel) {ce[:,0].mean()-se[:,0].mean():+.2f}sharp")


if __name__ == "__main__":
    main()
