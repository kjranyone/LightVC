"""R-proto-A: train the F0-free FreeVocoder to transparency (same harness as
kansei_train, minus F0). Single-speaker self-recon, held-out 3 eval utts +
render on 6 unseen -> ear-AB vs bigvgan (ceiling) / kansei (F0-driven reference).

Hypothesis (one): dropping F0 removes the unseen-hoarse root cause (mistracked
F0) while a Vocos-style free-phase ISTFT head still matches BigVGAN by ear.
Judge = human ear only.

Usage: cd training && HF_HUB_OFFLINE=1 uv run python free_train.py --steps 30000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import bigvgan
from bigvgan.env import AttrDict
from bigvgan.meldataset import get_mel_spectrogram
from bigvgan.discriminators import (MultiPeriodDiscriminator,
                                    MultiResolutionDiscriminator)
from bigvgan.loss import (MultiScaleMelSpectrogramLoss, discriminator_loss,
                          feature_loss, generator_loss)
from free_vocoder import FreeVocoder
from kansei_train import mrstft

_PHWIN = {}


def phase_loss(y, y_hat, nfft=2048, hop=512):
    """Anti-wrapping phase supervision (APNet, survey §7-0): IP + GD + IAF on the
    STFT phase of y_hat vs y, magnitude-weighted toward harmonic peaks. Directly
    targets the residual かすれ = inter-harmonic phase incoherence of the free-phase
    ISTFT head (STFT(ISTFT(S))~=S so this supervises the head's predicted phase).
    Weighting by target magnitude emphasizes peak coherence -> clean valleys."""
    pi = torch.pi
    key = (nfft, y.device)
    if key not in _PHWIN:
        _PHWIN[key] = torch.hann_window(nfft, device=y.device)
    w = _PHWIN[key]
    y, y_hat = y.squeeze(1), y_hat.squeeze(1)
    Y = torch.stft(y, nfft, hop, nfft, w, return_complex=True)
    Yh = torch.stft(y_hat, nfft, hop, nfft, w, return_complex=True)
    P, Ph = torch.angle(Y), torch.angle(Yh)
    mag = Y.abs()
    wt = mag / (mag.mean(dim=(1, 2), keepdim=True) + 1e-6)

    def aw(x):
        return torch.abs(x - 2 * pi * torch.round(x / (2 * pi)))

    l_ip = (aw(Ph - P) * wt).mean()
    l_gd = (aw((Ph[:, 1:] - Ph[:, :-1]) - (P[:, 1:] - P[:, :-1]))
            * wt[:, 1:]).mean()
    l_iaf = (aw((Ph[..., 1:] - Ph[..., :-1]) - (P[..., 1:] - P[..., :-1]))
             * wt[..., 1:]).mean()
    return l_ip + l_gd + l_iaf

SNAP = Path("/home/kojirotanaka/.cache/huggingface/hub/models--nvidia--"
            "bigvgan_v2_44khz_128band_512x/snapshots/"
            "95a9d1dcb12906c03edd938d77b9333d6ded7dfb")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "female-dataset/af1ad5575a3fa383"
CACHE = ROOT / "data/e2_ltv_cache/af1ad5575a3fa383"
TRIAGE = ROOT / "results/e2_triage"
SR = 44100
HOP = 512
EPS = 1e-8
DEV = "cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")


def load_data():
    eval_uids = [p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]]
    train, ev = [], []
    for npz in sorted(CACHE.glob("*.npz")):
        uid = npz.stem
        w = DATA / (uid + ".wav")
        if not w.exists():
            continue
        x, _ = librosa.load(str(w), sr=SR, mono=True)
        item = (uid, x.astype(np.float32))
        (ev if uid in eval_uids else train).append(item)
    return train, ev


def gain_match(y, gt):
    g = np.sqrt((gt ** 2).mean() / ((y ** 2).mean() + EPS))
    y = y * g
    peak = np.abs(y).max()
    if peak > 0.95:
        y = y * (0.95 / peak)
    return y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--tag", default="free")
    ap.add_argument("--init", default="")
    ap.add_argument("--lam-mrstft", type=float, default=2.0)
    ap.add_argument("--lam-phase", type=float, default=0.0)
    args = ap.parse_args()

    h = AttrDict(json.loads((SNAP / "config.json").read_text()))
    h["resolutions"] = [[1024, 256, 1024], [2048, 512, 2048], [512, 128, 512]]
    gen = FreeVocoder(dim=args.dim, n_layers=args.layers, causal=args.causal).to(DEV)
    if args.init:
        gen.load_state_dict(torch.load(args.init, map_location=DEV)["gen"])
        print(f"warm-start gen from {args.init}", flush=True)
    mpd = MultiPeriodDiscriminator(h).to(DEV)
    mrd = MultiResolutionDiscriminator(h).to(DEV)
    fn_mel = MultiScaleMelSpectrogramLoss(sampling_rate=SR).to(DEV)
    og = torch.optim.AdamW(gen.parameters(), args.lr, betas=(0.8, 0.99))
    od = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()),
                           args.lr, betas=(0.8, 0.99))
    print(f"FreeVocoder {sum(p.numel() for p in gen.parameters())/1e6:.1f}M "
          f"causal={args.causal} on {DEV}", flush=True)

    train, ev = load_data()
    print(f"train {len(train)} / eval {len(ev)}", flush=True)
    rng = np.random.default_rng(0)
    ckdir = ROOT / f"training/checkpoints/{args.tag}"
    ckdir.mkdir(parents=True, exist_ok=True)

    def sample():
        mels, ys = [], []
        for i in rng.integers(0, len(train), args.batch):
            _, x = train[i]
            nf = len(x) // HOP
            if nf <= args.frames:
                continue
            s = rng.integers(0, nf - args.frames)
            xw = x[s * HOP:(s + args.frames) * HOP]
            mel = get_mel_spectrogram(torch.tensor(xw).unsqueeze(0), h)[0]
            mels.append(mel[..., :args.frames])
            ys.append(torch.tensor(xw[:args.frames * HOP]))
        return torch.stack(mels).to(DEV), torch.stack(ys).to(DEV)

    @torch.no_grad()
    def evaluate(step):
        gen.eval()
        for uid, x in ev:
            nf = min(len(x) // HOP, int(8.0 * SR) // HOP)
            gt = x[:nf * HOP]
            mel = get_mel_spectrogram(torch.tensor(gt).unsqueeze(0), h).to(DEV)
            y = gen(mel).squeeze().cpu().numpy()
            sf.write(TRIAGE / f"{uid}_{args.tag}.wav", gain_match(y, gt[:len(y)]), SR)
        gen.train()
        print(f"[eval {step}] wrote {len(ev)} -> {args.tag}", flush=True)

    t0 = time.time()
    evaluate(0)
    for it in range(1, args.steps + 1):
        mel, y = sample()
        y = y.unsqueeze(1)
        y_hat = gen(mel).unsqueeze(1)
        n = min(y.shape[-1], y_hat.shape[-1])
        y, y_hat = y[..., :n], y_hat[..., :n]

        od.zero_grad()
        yr, yg, _, _ = mpd(y, y_hat.detach())
        ldf, _, _ = discriminator_loss(yr, yg)
        yr, yg, _, _ = mrd(y, y_hat.detach())
        ldr, _, _ = discriminator_loss(yr, yg)
        (ldf + ldr).backward()
        od.step()

        og.zero_grad()
        loss_mel = fn_mel(y, y_hat) * 15.0
        yr, yg, fr, fg = mpd(y, y_hat)
        lfm_f, (lg_f, _) = feature_loss(fr, fg), generator_loss(yg)
        yr, yg, fr, fg = mrd(y, y_hat)
        lfm_r, (lg_r, _) = feature_loss(fr, fg), generator_loss(yg)
        loss_mr = mrstft(y, y_hat) * args.lam_mrstft
        loss_ph = phase_loss(y, y_hat) * args.lam_phase if args.lam_phase > 0 else 0.0
        g_all = lg_f + lg_r + lfm_f + lfm_r + loss_mel + loss_mr + loss_ph
        g_all.backward()
        og.step()

        if it % 100 == 0:
            ph = loss_ph.item() if args.lam_phase > 0 else 0.0
            print(f"it {it:6d} mel {loss_mel.item()/15:.3f} "
                  f"gen {(lg_f+lg_r).item():.3f} fm {(lfm_f+lfm_r).item():.3f} "
                  f"ph {ph:.3f} d {(ldf+ldr).item():.3f} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if it % args.eval_every == 0:
            evaluate(it)
            torch.save({"gen": gen.state_dict(), "step": it,
                        "args": vars(args)}, ckdir / "last.pt")
    torch.save({"gen": gen.state_dict(), "step": args.steps,
                "args": vars(args)}, ckdir / "last.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
