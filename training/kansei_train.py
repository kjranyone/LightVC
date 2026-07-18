"""Gate V-1: train the own-architecture KanseiVocoder to transparency.

mel->waveform reconstruction of one speaker (af1ad5575a3fa383), F0 from cache.
Loss = MultiScaleMel(BigVGAN teacher)*15 + MPD/MRD adversarial + feature matching
(non-regressive texture, P3). Held-out 3 utts -> ear-AB vs bigvgan ceiling / gt.
Judge = human ear only. Own generator, borrowed nothing in the shipping graph.

Usage: cd training && HF_HUB_OFFLINE=1 uv run python kansei_train.py --steps 30000
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
from kansei_vocoder import KanseiVocoder

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


_MRWIN = {}


def mrstft(y, y_hat, cfgs=((512, 128), (1024, 256), (2048, 512))):
    """Multi-resolution linear-magnitude STFT L1. Linear (not log-mel) so the
    strong low band (F0/F1) is supervised directly -> fixes the ~10% F0/F1
    under-production that log-mel is too soft to correct."""
    loss = 0.0
    for nfft, hop in cfgs:
        key = (nfft, y.device)
        if key not in _MRWIN:
            _MRWIN[key] = torch.hann_window(nfft, device=y.device)
        w = _MRWIN[key]
        Y = torch.stft(y.squeeze(1), nfft, hop, window=w, return_complex=True).abs()
        Yh = torch.stft(y_hat.squeeze(1), nfft, hop, window=w, return_complex=True).abs()
        loss = loss + F.l1_loss(Yh, Y)
    return loss


def octave_correct(f0, thr=330.0):
    """Cached F0 is octave-doubled (dominant 2nd harmonic fools the tracker on
    this weak-fundamental voice) so the harmonic source misses the true
    fundamental (~220Hz) + odd harmonics -> thin body.

    Correct PER-UTTERANCE and UNIFORMLY (halve all voiced iff the utt's voiced
    median > thr), NOT per-frame: a per-frame threshold halves only the frames
    that stray above thr, creating 2x jumps between adjacent frames near the
    boundary -> pitch instability -> hoarse/かすれ voice. Uniform per-utt is
    jump-free (octave-jumps 5-10% -> ~0%). Verified: all 66 utts -> 165-318Hz."""
    f = f0.copy()
    v = f > 1
    for _ in range(2):
        if v.sum() > 0 and np.median(f[v]) > thr:
            f[v] = f[v] / 2.0
    return f


def load_data():
    eval_uids = [p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]]
    train, ev = [], []
    for npz in sorted(CACHE.glob("*.npz")):
        uid = npz.stem
        w = DATA / (uid + ".wav")
        if not w.exists():
            continue
        x, _ = librosa.load(str(w), sr=SR, mono=True)
        f0 = octave_correct(np.load(npz)["f0"].astype(np.float32))
        item = (uid, x.astype(np.float32), f0)
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
    ap.add_argument("--tag", default="kansei")
    ap.add_argument("--init", default="")
    ap.add_argument("--lam-mrstft", type=float, default=2.0)
    args = ap.parse_args()

    h = AttrDict(json.loads((SNAP / "config.json").read_text()))
    h["resolutions"] = [[1024, 256, 1024], [2048, 512, 2048], [512, 128, 512]]
    gen = KanseiVocoder(dim=args.dim, n_layers=args.layers, causal=args.causal).to(DEV)
    if args.init:
        gen.load_state_dict(torch.load(args.init, map_location=DEV)["gen"])
        print(f"warm-start gen from {args.init}", flush=True)
    mpd = MultiPeriodDiscriminator(h).to(DEV)
    mrd = MultiResolutionDiscriminator(h).to(DEV)
    fn_mel = MultiScaleMelSpectrogramLoss(sampling_rate=SR).to(DEV)
    og = torch.optim.AdamW(gen.parameters(), args.lr, betas=(0.8, 0.99))
    od = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()),
                           args.lr, betas=(0.8, 0.99))
    print(f"KanseiVocoder {sum(p.numel() for p in gen.parameters())/1e6:.1f}M "
          f"causal={args.causal} on {DEV}", flush=True)

    train, ev = load_data()
    print(f"train {len(train)} / eval {len(ev)}", flush=True)
    rng = np.random.default_rng(0)
    ckdir = ROOT / f"training/checkpoints/{args.tag}"
    ckdir.mkdir(parents=True, exist_ok=True)
    seg = args.frames * HOP

    def sample():
        mels, f0s, ys = [], [], []
        for i in rng.integers(0, len(train), args.batch):
            _, x, f0 = train[i]
            nf = min(len(f0), len(x) // HOP)
            if nf <= args.frames:
                continue
            s = rng.integers(0, nf - args.frames)
            xw = x[s * HOP:(s + args.frames) * HOP]
            mel = get_mel_spectrogram(torch.tensor(xw).unsqueeze(0), h)[0]
            F = mel.shape[-1]
            mels.append(mel[..., :args.frames])
            f0s.append(torch.tensor(f0[s:s + args.frames]))
            ys.append(torch.tensor(xw[:args.frames * HOP]))
        return (torch.stack(mels).to(DEV), torch.stack(f0s).to(DEV),
                torch.stack(ys).to(DEV))

    @torch.no_grad()
    def evaluate(step):
        gen.eval()
        for uid, x, f0 in ev:
            nf = min(len(f0), len(x) // HOP, int(8.0 * SR) // HOP)
            gt = x[:nf * HOP]
            mel = get_mel_spectrogram(torch.tensor(gt).unsqueeze(0), h).to(DEV)
            f0t = torch.tensor(f0[:nf]).unsqueeze(0).to(DEV)
            y = gen(mel, f0t).squeeze().cpu().numpy()
            sf.write(TRIAGE / f"{uid}_{args.tag}.wav", gain_match(y, gt[:len(y)]), SR)
        gen.train()
        print(f"[eval {step}] wrote {len(ev)} -> {args.tag}", flush=True)

    t0 = time.time()
    evaluate(0)
    for it in range(1, args.steps + 1):
        mel, f0, y = sample()
        y = y.unsqueeze(1)
        y_hat = gen(mel, f0).unsqueeze(1)
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
        g_all = lg_f + lg_r + lfm_f + lfm_r + loss_mel + loss_mr
        g_all.backward()
        og.step()

        if it % 100 == 0:
            print(f"it {it:6d} mel {loss_mel.item()/15:.3f} "
                  f"gen {(lg_f+lg_r).item():.3f} fm {(lfm_f+lfm_r).item():.3f} "
                  f"d {(ldf+ldr).item():.3f} ({(time.time()-t0)/60:.1f}m)", flush=True)
        if it % args.eval_every == 0:
            evaluate(it)
            torch.save({"gen": gen.state_dict(), "step": it,
                        "args": vars(args)}, ckdir / "last.pt")
    torch.save({"gen": gen.state_dict(), "step": args.steps,
                "args": vars(args)}, ckdir / "last.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
