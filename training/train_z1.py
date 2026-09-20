"""Z1 — MelGen self-reconstruction (zeroshot_vc.md §6.1 Phase B / §9 Z1).

G reconstructs mel_of(X) from (content, prosody, target z_spk/s_art), the mel is
fed to the FROZEN freebig vocoder to hear it. Acceptance is the ear (does the
AdaIN clone follow the target timbre without muffling); the overfit gate here is
the疎通 check: mel-L1 must drop hard on a few utterances.

Losses (§6.1 / §8.1):
  - mel-L1 direct on m_hat vs mel_of(y): the manifold anchor. Warm-up high then
    ANNEAL to a floor (never 0, never a constant 45 — that route muffles).
  - mrstft through the FROZEN freebig vocoder: the anti-muffle audio signal.
    freebig weights are frozen (never updated) but kept in the autograd graph so
    crisp-texture gradient reaches G's mel. Ramps in after warm-up. NO GAN in Z1.

Reuses (unchanged): MoeVCSet / TimbreEncoder / ArticEncoder / ContentScrub from
train_m2; mel_of / mrstft_loss from train_m1; FreeVocoder (freebig) frozen.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, mrstft_loss, fm_loss, SR, HOP, N_MELS, DEV
from train_m2 import (MoeVCSet, TimbreEncoder, ArticEncoder, ContentScrub,
                      Discriminator2, TIMBRE_DIM, ART_DIM, N_ARTIC, ARTIC_PATH)
from free_vocoder import FreeVocoder
from mel_gen import MelGen


def load_freebig(path: str):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    a = ck["args"]
    voc = FreeVocoder(dim=a["dim"], n_layers=a["layers"], causal=bool(a["causal"]),
                      nfft=a["nfft"], win=a["win"], hop=a["hop"]).to(DEV)
    voc.load_state_dict(ck["gen"])
    voc.eval()
    for p in voc.parameters():
        p.requires_grad_(False)
    return voc, ck.get("step")


def mel_weight(step: int, warmup: int, hi: float, floor: float, anneal: int) -> float:
    if step < warmup:
        return hi
    t = min(1.0, (step - warmup) / max(1, anneal))
    return hi + (floor - hi) * t


def mrs_weight(step: int, warmup: int, target: float, ramp: int) -> float:
    if step < warmup:
        return 0.0
    return target * min(1.0, (step - warmup) / max(1, ramp))


@torch.no_grad()
def init_head_bias(g: MelGen, dl: DataLoader, n_batches: int = 8) -> np.ndarray:
    acc = torch.zeros(N_MELS, device=DEV)
    cnt = 0
    for i, batch in enumerate(dl):
        y = batch[2].to(DEV)
        m = mel_of(y.squeeze(1))
        acc += m.mean(dim=(0, 2))
        cnt += 1
        if i + 1 >= n_batches:
            break
    mean = (acc / max(1, cnt)).cpu().numpy()
    g.head.bias.data.copy_(torch.from_numpy(mean).to(DEV))
    return mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/z1_overfit_feat")
    ap.add_argument("--out", default="checkpoints/z1")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--mel-hi", type=float, default=45.0)
    ap.add_argument("--mel-floor", type=float, default=10.0)
    ap.add_argument("--mel-anneal", type=int, default=1500)
    ap.add_argument("--mrs-w", type=float, default=2.0)
    ap.add_argument("--mrs-ramp", type=int, default=1000)
    ap.add_argument("--no-voc-loss", action="store_true",
                    help="disable through-vocoder mrstft (pure-mel overfit smoke)")
    ap.add_argument("--content-mode", choices=["scrub", "raw", "oracle"], default="scrub",
                    help="G content conditioning richness: scrub=ContentScrub'd CV768 "
                         "(current), raw=CV768 no scrub (+info), oracle=CV768 + source "
                         "mel_t concatenated (info ceiling, self-recon only)")
    ap.add_argument("--gan", action="store_true",
                    help="through-vocoder texture GAN (MPD+MSD+MRD on freebig(m_hat))")
    ap.add_argument("--gan-after", type=int, default=1000)
    ap.add_argument("--fm-w", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--render-every", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    artic = torch.load(ARTIC_PATH, weights_only=False) if Path(ARTIC_PATH).exists() else {}
    if artic:
        av = np.stack(list(artic.values()))
        amu, asd = av.mean(0), av.std(0) + 1e-6
    else:
        amu, asd = np.zeros(N_ARTIC, np.float32), np.ones(N_ARTIC, np.float32)
    ds = MoeVCSet(args.feat, args.seg, artic, amu, asd)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0)
    print(f"Z1 MelGen self-recon | {len(ds)} utts | {ds.n_spk} spk | seg={args.seg} "
          f"| artic-cache {len(artic)} | steps={args.steps}", flush=True)

    voc, voc_step = load_freebig(args.freebig)
    print(f"freebig frozen (step {voc_step}) {sum(p.numel() for p in voc.parameters())/1e6:.1f}M", flush=True)

    cond_dim = 768 + 2 + (N_MELS if args.content_mode == "oracle" else 0)
    g = MelGen(cond_dim=cond_dim, dim=args.dim, n_layers=args.layers, n_mels=N_MELS,
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM, causal=False).to(DEV)
    t = TimbreEncoder().to(DEV)
    ea = ArticEncoder().to(DEV)
    scrub = ContentScrub().to(DEV) if args.content_mode == "scrub" else None
    print(f"content-mode={args.content_mode} | G cond_dim={cond_dim}", flush=True)
    mel_mean = init_head_bias(g, dl)
    print(f"MelGen {sum(p.numel() for p in g.parameters())/1e6:.2f}M | head bias<-data mean "
          f"log-mel [{mel_mean.min():.2f},{mel_mean.max():.2f}]", flush=True)

    params = (list(g.parameters()) + list(t.parameters()) + list(ea.parameters()))
    if scrub is not None:
        params += list(scrub.parameters())
    opt = torch.optim.AdamW(params, args.lr, betas=(0.8, 0.99))
    disc = None
    if args.gan:
        disc = Discriminator2().to(DEV)
        od = torch.optim.AdamW(disc.parameters(), args.lr, betas=(0.8, 0.99))
        print(f"through-vocoder GAN | Discriminator2 (MPD+MSD+MRD) "
              f"{sum(p.numel() for p in disc.parameters())/1e6:.1f}M | gan_after={args.gan_after} "
              f"(freebig frozen but in autograd graph)", flush=True)

    def save(tag: str) -> None:
        blob = {"g": g.state_dict(), "t": t.state_dict(), "ea": ea.state_dict(),
                "mel_mean": mel_mean, "step": step,
                "args": {"dim": args.dim, "layers": args.layers,
                         "content_mode": args.content_mode}}
        if scrub is not None:
            blob["scrub"] = scrub.state_dict()
        if disc is not None:
            blob["d"] = disc.state_dict()
        torch.save(blob, out / tag)

    step = 0
    best = 1e9
    while step < args.steps:
        for batch in dl:
            cond, f0, y, rw = (batch[0].to(DEV), batch[1].to(DEV), batch[2].to(DEV),
                               batch[3].to(DEV))
            mel_ref = mel_of(rw)
            s = t(mel_ref)
            s_art, _ = ea(mel_ref)
            mel_t = mel_of(y.squeeze(1))
            if args.content_mode == "scrub":
                cond2 = torch.cat([scrub(cond[:, :768]), cond[:, 768:]], dim=1)
            elif args.content_mode == "raw":
                cond2 = cond
            else:  # oracle: raw CV768 + prosody + source mel_t (info ceiling)
                Tc = min(cond.shape[-1], mel_t.shape[-1])
                cond2 = torch.cat([cond[..., :Tc], mel_t[..., :Tc]], dim=1)
            m_hat = g(cond2, s, s_art)
            T = min(m_hat.shape[-1], mel_t.shape[-1])
            mel_l = F.l1_loss(m_hat[..., :T], mel_t[..., :T])

            w_mel = mel_weight(step, args.warmup, args.mel_hi, args.mel_floor, args.mel_anneal)
            w_mrs = 0.0 if args.no_voc_loss else mrs_weight(step, args.warmup, args.mrs_w, args.mrs_ramp)
            use_gan = args.gan and step >= args.gan_after
            need_voc = use_gan or w_mrs > 0
            mrs = torch.tensor(0.0, device=DEV)
            g_adv = torch.tensor(0.0, device=DEV)
            d_loss_v = 0.0
            if need_voc:
                y_hat = voc(m_hat)                        # frozen weights, grad -> G
                L = min(y_hat.shape[-1], y.shape[-1])
                y_hat = y_hat[..., :L]
                y_real = y.squeeze(1)[..., :L]
            if w_mrs > 0:
                mrs = mrstft_loss(y_hat, y_real)

            if use_gan:                                   # D step (LSGAN, MPD+MSD+MRD)
                od.zero_grad(set_to_none=True)
                dr, _ = disc(y_real.unsqueeze(1))
                dg, _ = disc(y_hat.detach().unsqueeze(1))
                d_loss = sum(((r - 1) ** 2).mean() + (gg ** 2).mean() for r, gg in zip(dr, dg))
                if torch.isfinite(d_loss):
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), 10.0)
                    od.step()
                    d_loss_v = d_loss.item()

            g_loss = w_mel * mel_l + w_mrs * mrs
            if use_gan:                                   # G adversarial + feature matching
                dg, fg = disc(y_hat.unsqueeze(1))
                dr, fr = disc(y_real.unsqueeze(1))
                g_adv = sum(((gg - 1) ** 2).mean() for gg in dg)
                g_loss = g_loss + g_adv + args.fm_w * fm_loss(fr, fg)

            if not torch.isfinite(g_loss):
                print(f"step {step} NON-FINITE g_loss, skip", flush=True)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue

            opt.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()

            if step % 25 == 0:
                msg = (f"step {step} mel_l1 {mel_l.item():.4f} mrs {float(mrs.detach()):.3f} "
                       f"w_mel {w_mel:.1f} w_mrs {w_mrs:.2f}")
                if use_gan:
                    msg += f" g_adv {float(g_adv.detach()):.3f} d {d_loss_v:.3f}"
                print(msg + f" g_loss {g_loss.item():.3f}", flush=True)
            if mel_l.item() < best:
                best = mel_l.item()
            if step % args.render_every == 0 and step > 0:
                save("last.pt")
            step += 1
            if step >= args.steps:
                break
    save("last.pt")
    print(f"done | best mel_l1 {best:.4f} -> {out/'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
