"""Z3 swap test — THE frontier measurement (zeroshot_vc.md §9 Z3).

Take neutral-style input utterances, keep timbre (z_spk) fixed, and swap only
s_art between s_art(neutral) and s_art(moe). Render through freebig and MEASURE
the output-audio formants:
  ΔF1      = F1_moe - F1_neu   (expect > 0  = moe: higher F1)
  ΔF2range = F2range_moe - F2range_neu (expect < 0 = moe: narrower F2 spread)
If both move the moe way, the articulatory-style clone works on REAL audio (the
PoC's formant-domain ΔF1 reproduced end-to-end). Disentanglement seed: output
spectral centroid should barely move (s_art must not drag timbre).

Arms in results/z3/: gt / z3neu / z3moe (neutral input, s_art neutral vs moe).
Judge = ears/eyes; ΔF1/ΔF2range = mechanism proxy.
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
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, DEV
from train_m2 import TimbreEncoder, ArticEncoder, ContentScrub, TIMBRE_DIM, ART_DIM
from mel_gen import MelGen
from train_z1 import load_freebig
from build_formant_cache import formant_track, style_name

MOE_STYLES = {"cute_high", "intimate_close", "young_bright"}
FSR = 16000  # formant_track sample rate


def load_wav(path: str) -> np.ndarray:
    w, _ = sf.read(path, dtype="float32")
    return w.mean(1) if w.ndim > 1 else w


def build_cond(d: dict) -> torch.Tensor:
    content = d["content"].float()
    f0 = d["f0"].float(); energy = d["energy"].float()
    tmel = f0.shape[0]
    c = F.interpolate(content.t().unsqueeze(0), size=tmel, mode="linear",
                      align_corners=False).squeeze(0)
    logf0 = torch.log(f0.clamp(min=1.0)) / 7.0
    eng = torch.log(energy.clamp(min=1e-4)) * 0.2
    return torch.cat([c, logf0.unsqueeze(0), eng.unsqueeze(0)], dim=0), tmel


@torch.no_grad()
def s_art_of(ea, path, sec=4.0):
    w = load_wav(path)[: int(sec * SR)]
    mel = mel_of(torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0).to(DEV))
    return ea(mel)[0]


def formant_stats(wav44: np.ndarray) -> tuple:
    w16 = librosa.resample(wav44, orig_sr=SR, target_sr=FSR)
    ft = formant_track(w16.astype(np.float64))
    f1, f2 = ft[:, 0], ft[:, 1]
    m = np.isfinite(f1) & np.isfinite(f2)
    if m.sum() < 8:
        return np.nan, np.nan
    f1, f2 = f1[m], f2[m]
    return float(np.median(f1)), float(np.percentile(f2, 90) - np.percentile(f2, 10))


def centroid(wav44: np.ndarray) -> float:
    S = np.abs(librosa.stft(wav44, n_fft=2048, hop_length=512)) + 1e-7
    f = np.linspace(0, SR / 2, S.shape[0])
    P = (S ** 2).mean(1)
    return float((f * P).sum() / P.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/z3/last.pt")
    ap.add_argument("--feat", default="../data/z3_feat")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--out", default="../results/z3")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n-ref", type=int, default=40, help="utts per style for s_art mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-audio", type=int, default=6)
    args = ap.parse_args()

    random.seed(args.seed)
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    ga = ck.get("args", {"dim": 384, "layers": 6})
    g = MelGen(cond_dim=768 + 2, dim=ga["dim"], n_layers=ga["layers"],
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM).to(DEV)
    g.load_state_dict(ck["g"]); g.eval()
    t = TimbreEncoder().to(DEV); t.load_state_dict(ck["t"]); t.eval()
    ea = ArticEncoder().to(DEV); ea.load_state_dict(ck["ea"]); ea.eval()
    scrub = ContentScrub().to(DEV); scrub.load_state_dict(ck["scrub"]); scrub.eval()
    voc, voc_step = load_freebig(args.freebig)

    files = sorted(Path(args.feat).rglob("*.pt"))
    by_style = defaultdict(list)
    for f in files:
        d = torch.load(f, weights_only=False)
        by_style[style_name(d.get("style", ""))].append((f, d["path"]))
    neu = by_style.get("neutral", [])
    moe = [x for st in MOE_STYLES for x in by_style.get(st, [])]
    print(f"Z3 swap | ckpt step {ck.get('step')} art_scrub_w {ga.get('art_scrub_w')} | "
          f"neutral {len(neu)} moe {len(moe)} | freebig {voc_step}", flush=True)

    # style s_art means (speaker-averaged style code)
    with torch.no_grad():
        s_neu = torch.stack([s_art_of(ea, p) for _, p in random.sample(neu, min(args.n_ref, len(neu)))]).mean(0)
        s_moe = torch.stack([s_art_of(ea, p) for _, p in random.sample(moe, min(args.n_ref, len(moe)))]).mean(0)
    print(f"  |s_moe - s_neu| L2 = {torch.norm(s_moe - s_neu).item():.3f}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    inputs = random.sample(neu, min(args.n, len(neu)))
    dF1, dF2r, dCent = [], [], []
    written = 0
    for i, (f, path) in enumerate(inputs):
        d = torch.load(f, weights_only=False)
        cond, tmel = build_cond(d)
        cond = cond.unsqueeze(0).to(DEV)
        y = load_wav(path)[: tmel * HOP]
        with torch.no_grad():
            z_spk = t(mel_of(torch.from_numpy(np.ascontiguousarray(y)).float().unsqueeze(0).to(DEV)))
            cond2 = torch.cat([scrub(cond[:, :768]), cond[:, 768:]], dim=1)
            w_neu = voc(g(cond2, z_spk, s_neu))[0].cpu().numpy()
            w_moe = voc(g(cond2, z_spk, s_moe))[0].cpu().numpy()
        f1n, f2rn = formant_stats(w_neu)
        f1m, f2rm = formant_stats(w_moe)
        if np.isfinite(f1n) and np.isfinite(f1m):
            dF1.append(f1m - f1n); dF2r.append(f2rm - f2rn)
            dCent.append(centroid(w_moe) - centroid(w_neu))
        if written < args.write_audio:
            stem = f"{i:02d}_{f.parent.name}_{f.stem}"
            sf.write(out / f"{stem}_gt.wav", np.clip(y, -1, 1), SR, subtype="PCM_16")
            sf.write(out / f"{stem}_z3neu.wav", np.clip(w_neu, -1, 1), SR, subtype="PCM_16")
            sf.write(out / f"{stem}_z3moe.wav", np.clip(w_moe, -1, 1), SR, subtype="PCM_16")
            written += 1

    dF1, dF2r, dCent = np.array(dF1), np.array(dF2r), np.array(dCent)
    print(f"\n=== Z3 articulatory swap (neutral input, s_art neu->moe), {len(dF1)} utts ===")
    print(f"  ΔF1       {np.mean(dF1):+7.1f} Hz  (>0 = moe direction)   per-utt+:{int((dF1>0).sum())}/{len(dF1)}")
    print(f"  ΔF2_range {np.mean(dF2r):+7.1f} Hz  (<0 = moe direction)   per-utt-:{int((dF2r<0).sum())}/{len(dF2r)}")
    moe_dir = (np.mean(dF1) > 0) and (np.mean(dF2r) < 0)
    print(f"  => both moe direction: {moe_dir}")
    print(f"  disentangle: Δcentroid {np.mean(dCent):+7.1f} Hz (small = timbre held by s_art swap)")
    print(f"  render -> {out} ({written} trios)", flush=True)


if __name__ == "__main__":
    main()
