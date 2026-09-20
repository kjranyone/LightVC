"""Z1 render — self-reconstruction, 3 arms per utt (zeroshot_vc.md §9 Z1).

For each utterance renders:
  *_gt.wav       original waveform (the reference/ceiling of the ear test)
  *_ceiling.wav  freebig(mel_of(gt))  = vocoder ceiling (best a perfect mel gives)
  *_z1.wav       freebig(MelGen(...))  = what G actually paints, self-recon

The gap z1 vs ceiling = how far G's mel is from the true mel (muffle shows here).
The gap ceiling vs gt = the frozen vocoder's own ceiling (not G's fault).
No moe_remap / register knobs (Z1 is self-recon疎通, not cross-identity).

Reads feats straight from the cache .pt (content / f0 / energy / path already
there), so it renders the exact conditioning the trainer saw.
"""
from __future__ import annotations

import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, DEV
from train_m2 import TimbreEncoder, ArticEncoder, ContentScrub, TIMBRE_DIM, ART_DIM
from free_vocoder import FreeVocoder
from mel_gen import MelGen
from train_z1 import load_freebig


def load_wav(path: str) -> torch.Tensor:
    w, _ = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    return torch.from_numpy(np.ascontiguousarray(w)).float()


def build_cond(d: dict) -> tuple:
    content = d["content"].float()
    f0 = d["f0"].float()
    energy = d["energy"].float()
    tmel = f0.shape[0]
    c = F.interpolate(content.t().unsqueeze(0), size=tmel, mode="linear",
                      align_corners=False).squeeze(0)
    logf0 = torch.log(f0.clamp(min=1.0)) / 7.0
    eng = torch.log(energy.clamp(min=1e-4)) * 0.2
    cond = torch.cat([c, logf0.unsqueeze(0), eng.unsqueeze(0)], dim=0)
    return cond, tmel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/z1/last.pt")
    ap.add_argument("--feat", default="../data/z1_overfit_feat")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--out", default="../results/z1")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    ga = ck.get("args", {"dim": 384, "layers": 6})
    mode = ga.get("content_mode", "scrub")
    arm = "z1" + mode
    cond_dim = 768 + 2 + (128 if mode == "oracle" else 0)
    g = MelGen(cond_dim=cond_dim, dim=ga["dim"], n_layers=ga["layers"],
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM).to(DEV)
    g.load_state_dict(ck["g"]); g.eval()
    t = TimbreEncoder().to(DEV); t.load_state_dict(ck["t"]); t.eval()
    ea = ArticEncoder().to(DEV); ea.load_state_dict(ck["ea"]); ea.eval()
    scrub = None
    if mode == "scrub":
        scrub = ContentScrub().to(DEV); scrub.load_state_dict(ck["scrub"]); scrub.eval()
    voc, voc_step = load_freebig(args.freebig)

    files = sorted(Path(args.feat).rglob("*.pt"))
    by_spk = defaultdict(list)
    for f in files:
        by_spk[f.parent.name].append(f)
    pick = files[:] if len(files) <= args.n else random.sample(files, args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"render Z1 | ckpt step {ck.get('step')} | content-mode {mode} (arm {arm}) | "
          f"freebig step {voc_step} | {len(pick)} utts -> {out}", flush=True)

    for i, f in enumerate(pick):
        d = torch.load(f, weights_only=False)
        cond, tmel = build_cond(d)
        cond = cond.unsqueeze(0).to(DEV)
        y = load_wav(d["path"])[: tmel * HOP]
        # reference for z_spk / s_art = another utt of the same speaker (matches
        # training); falls back to self if the speaker has a single utt.
        pool = [x for x in by_spk[f.parent.name] if x != f] or [f]
        rd = torch.load(random.choice(pool), weights_only=False)
        rw = load_wav(rd["path"])
        rn = int(3 * SR)
        rw = rw[:rn] if rw.shape[0] >= rn else F.pad(rw, (0, rn - rw.shape[0]))
        with torch.no_grad():
            mel_ref = mel_of(rw.unsqueeze(0).to(DEV))
            s = t(mel_ref)
            s_art, _ = ea(mel_ref)
            mel_gt = mel_of(y.unsqueeze(0).to(DEV))
            if mode == "scrub":
                cond2 = torch.cat([scrub(cond[:, :768]), cond[:, 768:]], dim=1)
            elif mode == "raw":
                cond2 = cond
            else:  # oracle: raw CV768 + prosody + source mel_t
                Tc = min(cond.shape[-1], mel_gt.shape[-1])
                cond2 = torch.cat([cond[..., :Tc], mel_gt[..., :Tc]], dim=1)
            m_hat = g(cond2, s, s_art)
            y_z1 = voc(m_hat)[0].cpu().numpy()
            y_ceil = voc(mel_gt)[0].cpu().numpy()
        stem = f"{f.parent.name}_{f.stem}"
        sf.write(out / f"{i:02d}_{stem}_gt.wav", np.clip(y.numpy(), -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"{i:02d}_{stem}_ceiling.wav", np.clip(y_ceil, -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"{i:02d}_{stem}_{arm}.wav", np.clip(y_z1, -1, 1), SR, subtype="PCM_16")
        print(f"  [{i}] {stem} tmel {tmel} ref {rd['speaker'] if 'speaker' in rd else '?'}", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
