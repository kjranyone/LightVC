"""Cross-VC render for the ECAPA-fooling check: for held pairs (A-content,
timbre-B) dump gt-A / self-A / cross-AB so the spectrogram shows whether id_out
kept speech structure (real conversion) or garbled it (numeric fooling)."""
from __future__ import annotations
import sys, argparse, random
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, DEV
from train_m2 import TimbreEncoder, TIMBRE_DIM, ART_DIM
from mel_gen import MelGen
from train_z1 import load_freebig
from f0leak_probe import load_wav, build_cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--out", default="../results/e2_cipt")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--upsample", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    ga = ck.get("args", {"dim": 384, "layers": 6})
    g = MelGen(cond_dim=770, dim=ga["dim"], n_layers=ga["layers"],
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM, f0_fourier=ga.get("f0_fourier", 0)).to(DEV)
    g.load_state_dict(ck["g"]); g.eval()
    t = TimbreEncoder().to(DEV); t.load_state_dict(ck["t"]); t.eval()
    voc, _ = load_freebig(args.freebig)
    held = [Path(x) for x in (Path(args.ckpt).parent / "heldout.txt").read_text().split("\n") if x]
    held = sorted(f for f in held if f.exists())
    by_spk = defaultdict(list)
    for f in held:
        by_spk[f.parent.name].append(f)
    spks = list(by_spk)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.shuffle(held)
    done = 0
    for f in held:
        if done >= args.n:
            break
        dA = torch.load(f, weights_only=False)
        if "f0" not in dA:
            continue
        others = [s for s in spks if s != f.parent.name]
        fB = random.choice(by_spk[random.choice(others)])
        dB = torch.load(fB, weights_only=False)
        wB = load_wav(dB["path"])
        cond, tmel = build_cond(dA, 0.0); cond = cond.unsqueeze(0).to(DEV)
        wA = load_wav(dA["path"])[: tmel * HOP]
        with torch.no_grad():
            zB = t(mel_of(torch.from_numpy(np.ascontiguousarray(wB[: int(3*SR)])).float().unsqueeze(0).to(DEV)))
            zA = t(mel_of(torch.from_numpy(np.ascontiguousarray(wA)).float().unsqueeze(0).to(DEV)))
            def synth(z):
                c = cond
                if args.upsample > 1:
                    c = torch.nn.functional.interpolate(cond, scale_factor=args.upsample, mode="linear", align_corners=False)
                return voc(g(c, z, None))[0].cpu().numpy()
            y_self = synth(zA)
            y_cross = synth(zB)
        stem = f"{done:02d}_{f.parent.name[:8]}_to_{fB.parent.name[:8]}"
        sf.write(out / f"{stem}_gtA.wav", np.clip(np.asarray(wA), -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"{stem}_selfA.wav", np.clip(y_self, -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"{stem}_crossAB.wav", np.clip(y_cross, -1, 1), SR, subtype="PCM_16")
        print(f"  [{done}] {stem} tmel {tmel}", flush=True)
        done += 1
    print(f"cross render -> {out} ({done})", flush=True)


if __name__ == "__main__":
    main()
