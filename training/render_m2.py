from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import pyworld
from transformers import HubertModel

sys.path.insert(0, str(Path(__file__).parent))
from nsf_hn import NsfHifiGan
from train_m1 import mel_of, SR, HOP, DEV
from train_m2 import TimbreEncoder, TIMBRE_DIM, ContentScrub

CV_SR = 16000


def load_cv() -> HubertModel:
    return HubertModel.from_pretrained("lengyue233/content-vec-best").to(DEV).eval()


@torch.no_grad()
def content_of(cv: HubertModel, wav16: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(wav16).float().view(1, -1).to(DEV)
    return cv(x).last_hidden_state.squeeze(0)


def harvest_f0(wav44: np.ndarray) -> np.ndarray:
    w = wav44.astype(np.float64)
    f0, t = pyworld.harvest(w, SR, f0_floor=65, f0_ceil=1000, frame_period=HOP / SR * 1000)
    return pyworld.stonemask(w, f0, t, SR).astype(np.float32)


def logstats(f0: np.ndarray) -> tuple:
    v = f0[f0 > 1.0]
    lg = np.log(v + 1e-6)
    return float(lg.mean()), float(lg.std() + 1e-5)


def moe_remap(f0_src: np.ndarray, ref_f0: np.ndarray, exaggerate: float,
              register_st: float = 0.0) -> np.ndarray:
    mu_s, sd_s = logstats(f0_src)
    mu_m, sd_m = logstats(ref_f0)
    out = f0_src.copy()
    v = f0_src > 1.0
    z = (np.log(f0_src[v] + 1e-6) - mu_s) / sd_s
    out[v] = np.exp(mu_m + z * sd_m * exaggerate + register_st * np.log(2.0) / 12.0)
    return out


def f0_energy(wav44: np.ndarray, tmel: int, shift: float, ref_f0: np.ndarray = None,
              exaggerate: float = 1.0, register_st: float = 0.0) -> tuple:
    f0 = harvest_f0(wav44)
    if ref_f0 is not None:
        f0 = moe_remap(f0, ref_f0, exaggerate, register_st)
    else:
        f0 = f0 * shift
    f0 = np.pad(f0, (0, max(0, tmel - len(f0))))[:tmel]
    seg = wav44[:tmel * HOP].reshape(tmel, HOP)
    e = np.sqrt((seg ** 2).mean(1) + 1e-9).astype(np.float32)
    return torch.from_numpy(f0), torch.from_numpy(e)


SWEEP = [("flat", 0.0, 1.0), ("lively", 0.0, 2.0), ("high", 4.0, 1.5), ("moe", 2.0, 2.4)]
PITCH_SWEEP = [("pitch-4", -4.0, 1.3), ("pitch-2", -2.0, 1.3), ("pitch0", 0.0, 1.3),
               ("pitch+2", 2.0, 1.3), ("pitch+4", 4.0, 1.3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/m2_vc/last.pt")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("MALE", "MOEREF"), required=True)
    ap.add_argument("--out", default="../results/diag_m2")
    ap.add_argument("--f0-shift", type=float, default=1.8)
    ap.add_argument("--exaggerate", type=float, default=1.4)
    ap.add_argument("--register", type=float, default=0.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--pitch-sweep", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    g = NsfHifiGan(cond_dim=768 + 2, timbre_dim=TIMBRE_DIM).to(DEV)
    g.load_state_dict(ck["g"])
    g.eval()
    t = TimbreEncoder().to(DEV)
    t.load_state_dict(ck["t"])
    t.eval()
    scrub = ContentScrub().to(DEV)
    scrub.load_state_dict(ck["scrub"])
    scrub.eval()
    cv = load_cv()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.pitch_sweep:
        settings = PITCH_SWEEP
    elif args.sweep:
        settings = SWEEP
    else:
        settings = [("converted", args.register, args.exaggerate)]
    print(f"render M2 | step {ck.get('step')} | {len(args.pair)} pairs | knobs={[s[0] for s in settings]}")

    for i, (male, moeref) in enumerate(args.pair):
        m16, _ = librosa.load(male, sr=CV_SR, mono=True)
        m44, _ = librosa.load(male, sr=SR, mono=True)
        content = content_of(cv, m16).float()
        tmel = len(m44) // HOP
        m44 = m44[:tmel * HOP]
        c = F.interpolate(content.t().unsqueeze(0), size=tmel, mode="linear",
                          align_corners=False).squeeze(0).to(DEV)
        with torch.no_grad():
            c = scrub(c.unsqueeze(0)).squeeze(0)
        ref44, _ = librosa.load(moeref, sr=SR, mono=True)
        ref = torch.from_numpy(ref44[: 3 * SR]).float().unsqueeze(0).to(DEV)
        ref_f0 = harvest_f0(ref44)
        with torch.no_grad():
            s = t(mel_of(ref))
        sf.write(out / f"diag{i}_source.wav", np.clip(m44, -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"diag{i}_moeref.wav", np.clip(ref44[: 3 * SR], -1, 1), SR, subtype="PCM_16")

        for tag, reg, exa in settings:
            f0, energy = f0_energy(m44, tmel, args.f0_shift, ref_f0=ref_f0,
                                   exaggerate=exa, register_st=reg)
            logf0 = torch.log(f0.clamp(min=1.0)) / 7.0
            eng = torch.log(energy.clamp(min=1e-4)) * 0.2
            cond = torch.cat([c, logf0.unsqueeze(0).to(DEV), eng.unsqueeze(0).to(DEV)],
                             dim=0).unsqueeze(0)
            with torch.no_grad():
                y = g(cond, f0.unsqueeze(0).to(DEV), s)[0, 0].cpu().numpy()
            sf.write(out / f"diag{i}_{tag}.wav", np.clip(y, -1, 1), SR, subtype="PCM_16")
        print(f"  [{i}] {Path(male).name} -> {Path(moeref).parent.name}", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
