"""Clean E5 eval: render with the EXACT training path (gain_match, no [0]/load_wav
drift) so ceiling matches the clean training-time FreeC output (verified identical
to results/e2_triage evidence). self = FreeC_e5(G(cond x4)); ceiling =
FreeC_foundation(get_mel_spectrogram(gt)). Saves wavs + prints SI-SDR/sq-PESQ."""
from __future__ import annotations
import sys, os, json, argparse, random
from pathlib import Path
from collections import defaultdict
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch, torch.nn.functional as F, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, DEV
from train_m2 import TimbreEncoder, TIMBRE_DIM, ART_DIM
from mel_gen import MelGen
from train_z1 import load_freebig
from f0leak_probe import load_wav, build_cond
from bigvgan.env import AttrDict
from bigvgan.meldataset import get_mel_spectrogram
from free_train_universal import SNAP, gain_match
import torchaudio.functional as AF
from torchaudio.pipelines import SQUIM_OBJECTIVE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gckpt", required=True)
    ap.add_argument("--vckpt", required=True)
    ap.add_argument("--out", default="../results/e5_clean")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--up", type=int, default=4)
    args = ap.parse_args()
    obj = SQUIM_OBJECTIVE.get_model().to(DEV).eval()
    gck = torch.load(args.gckpt, map_location=DEV, weights_only=False)
    ga = gck["args"]
    g = MelGen(cond_dim=770, dim=ga["dim"], n_layers=ga["layers"], timbre_dim=TIMBRE_DIM,
               art_dim=ART_DIM, f0_fourier=ga.get("f0_fourier", 0)).to(DEV)
    g.load_state_dict(gck["g"]); g.eval()
    t = TimbreEncoder().to(DEV); t.load_state_dict(gck["t"]); t.eval()
    freeE, _ = load_freebig(args.vckpt)                         # E5-trained FreeC
    freeF, _ = load_freebig("checkpoints/freeC/foundation_lowlatency_5p8ms.pt")
    h = AttrDict(json.loads((SNAP / "config.json").read_text())); h["hop_size"] = 128
    held = sorted(Path(x.strip()) for x in open("/tmp/e5eval_held.txt") if x.strip())
    held = [f for f in held if f.exists()]
    random.seed(0); random.shuffle(held)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def sisdr(y):
        with torch.no_grad():
            _, pe, si = obj(AF.resample(torch.from_numpy(np.ascontiguousarray(y)).float().unsqueeze(0).to(DEV), SR, 16000))
        return si.item(), pe.item()
    S = {"self": [], "ceiling": []}
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
            cond4 = F.interpolate(cond, scale_factor=args.up, mode="linear", align_corners=False)
            y_self = freeE(g(cond4, s, None)).squeeze().cpu().numpy()
            y_ceil = freeF(get_mel_spectrogram(gt.cpu(), h).to(DEV)).squeeze().cpu().numpy()
        wn = w.numpy() if hasattr(w, "numpy") else np.asarray(w)
        y_self = gain_match(y_self, wn[: len(y_self)]); y_ceil = gain_match(y_ceil, wn[: len(y_ceil)])
        ss, sp = sisdr(y_self); cs, cp = sisdr(y_ceil)
        S["self"].append((ss, sp)); S["ceiling"].append((cs, cp))
        stem = f"{done:02d}_{f.parent.name[:8]}"
        sf.write(out / f"{stem}_gt.wav", np.clip(wn, -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"{stem}_ceiling.wav", np.clip(y_ceil, -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"{stem}_self.wav", np.clip(y_self, -1, 1), SR, subtype="PCM_16")
        done += 1
    m = lambda a, i: float(np.mean([x[i] for x in a]))
    print(f"CLEAN E5 eval (train-path render, gain_match) | n={done}")
    print(f"  ceiling: SI-SDR {m(S['ceiling'],0):.2f}  sq-PESQ {m(S['ceiling'],1):.3f}")
    print(f"  self:    SI-SDR {m(S['self'],0):.2f}  sq-PESQ {m(S['self'],1):.3f}")
    print(f"  wavs -> {out}")


if __name__ == "__main__":
    main()
