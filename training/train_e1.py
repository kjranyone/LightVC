"""E1 — intelligibility foundation + scrub balance (zeroshot_vc.md content path).

Z3 output was unintelligible (ASR-impossible; formants crushed to noise). Suspects:
(a) undertrained (4500 steps), (b) over-aggressive scrub (GRL) destroying phonemes,
(c) ContentVec content floor. E1 isolates them: proper self-reconstruction training
(~20k+ steps, train/held-out split), scrub-strength ablation, gate = ASR-CER
(Whisper) — the proxy that does NOT diverge from the ear like harmonic metrics do.

Setup: G reconstructs mel_of(X) from ContentVec content + prosody, timbre AdaIN
(z_spk from a same-speaker ref), NO s_art. mel-L1 anneal + mrstft (through frozen
freebig), NO GAN. Scrub modes:
  none  : raw ContentVec, no ContentScrub, no GRL  = intelligibility ceiling.
  light : ContentScrub + weak speaker-GRL (small alpha).
  heavy : ContentScrub + strong speaker-GRL + articulatory-GRL (Z3-equivalent).
"""
from __future__ import annotations

import sys
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, mrstft_loss, SR, HOP, N_MELS, DEV
from train_m2 import (MoeVCSet, TimbreEncoder, ContentScrub, grad_reverse,
                      TIMBRE_DIM, ART_DIM, N_ARTIC, ARTIC_PATH)
from mel_gen import MelGen
from train_z1 import load_freebig, mel_weight, mrs_weight, init_head_bias

SCRUB = {"none": (0.0, 0.0, 0.0), "light": (0.05, 0.0, 0.2), "heavy": (0.1, 0.1, 0.5)}
#         mode : (spk_grl_w, art_grl_w, alpha_peak)


def is_held(key, mod: int = 12) -> bool:
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16) % mod == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrub", choices=["none", "light", "heavy"], required=True)
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--out", default="checkpoints/e1")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--mel-hi", type=float, default=45.0)
    ap.add_argument("--mel-floor", type=float, default=10.0)
    ap.add_argument("--mel-anneal", type=int, default=3000)
    ap.add_argument("--mrs-w", type=float, default=2.0)
    ap.add_argument("--mrs-ramp", type=int, default=1000)
    ap.add_argument("--no-voc-loss", action="store_true")
    ap.add_argument("--grl-ramp", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--render-every", type=int, default=5000)
    ap.add_argument("--held-by", choices=["utt", "speaker"], default="utt")
    args = ap.parse_args()
    spk_grl_w, art_grl_w, alpha_peak = SCRUB[args.scrub]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    artic = torch.load(ARTIC_PATH, weights_only=False) if Path(ARTIC_PATH).exists() else {}
    if artic:
        av = np.stack(list(artic.values())); amu, asd = av.mean(0), av.std(0) + 1e-6
    else:
        amu, asd = np.zeros(N_ARTIC, np.float32), np.ones(N_ARTIC, np.float32)
    ds = MoeVCSet(args.feat, args.seg, artic, amu, asd)
    # held-out split: by utt (held utts unseen) or by speaker (zero-shot: whole
    # speaker unseen, needed to measure identity injection on held pairs).
    hk = (lambda f: f.parent.name) if args.held_by == "speaker" else (lambda f: f)
    held = [f for f in ds.files if is_held(hk(f))]
    for spk in list(ds.by_spk):
        ds.by_spk[spk] = [f for f in ds.by_spk[spk] if not is_held(hk(f))]
    ds.files = [f for f in ds.files if not is_held(hk(f)) and len(ds.by_spk[f.parent.name]) >= 2]
    (out / "heldout.txt").write_text("\n".join(str(f) for f in held))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0)
    print(f"E1 scrub={args.scrub} (spk_grl {spk_grl_w} art_grl {art_grl_w} alpha {alpha_peak}) "
          f"| train {len(ds.files)} held {len(held)} | {ds.n_spk} spk | seg={args.seg} "
          f"| steps={args.steps}", flush=True)

    voc, voc_step = load_freebig(args.freebig)
    g = MelGen(cond_dim=768 + 2, dim=args.dim, n_layers=args.layers, n_mels=N_MELS,
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM, causal=False).to(DEV)
    t = TimbreEncoder().to(DEV)
    spk_clf = nn.Linear(TIMBRE_DIM, ds.n_spk).to(DEV)
    use_scrub = args.scrub != "none"
    scrub = ContentScrub().to(DEV) if use_scrub else None
    cspk_clf = (nn.Sequential(nn.LayerNorm(768), nn.Linear(768, ds.n_spk)).to(DEV)
                if spk_grl_w > 0 else None)
    cart_pred = (nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.LeakyReLU(0.1),
                               nn.Linear(128, N_ARTIC)).to(DEV) if art_grl_w > 0 else None)
    mel_mean = init_head_bias(g, dl)

    params = list(g.parameters()) + list(t.parameters()) + list(spk_clf.parameters())
    for m in (scrub, cspk_clf, cart_pred):
        if m is not None:
            params += list(m.parameters())
    opt = torch.optim.AdamW(params, args.lr, betas=(0.8, 0.99))

    def save(tag):
        blob = {"g": g.state_dict(), "t": t.state_dict(), "mel_mean": mel_mean,
                "step": step, "scrub_mode": args.scrub,
                "args": {"dim": args.dim, "layers": args.layers}}
        if scrub is not None:
            blob["scrub"] = scrub.state_dict()
        torch.save(blob, out / tag)

    step = 0
    while step < args.steps:
        for batch in dl:
            cond, f0, y, rw, spk = (batch[0].to(DEV), batch[1].to(DEV), batch[2].to(DEV),
                                    batch[3].to(DEV), batch[4].to(DEV))
            iartic, ivalid = batch[7].to(DEV), batch[8].to(DEV)
            mel_in = mel_of(y.squeeze(1))
            s = t(mel_of(rw))
            spk_ce = F.cross_entropy(spk_clf(s), spk)
            content_raw = cond[:, :768]
            content_c = scrub(content_raw) if use_scrub else content_raw
            alpha = alpha_peak * min(1.0, step / max(1, args.grl_ramp))
            c_adv = torch.tensor(0.0, device=DEV); c_art_adv = torch.tensor(0.0, device=DEV)
            if cspk_clf is not None:
                c_adv = F.cross_entropy(cspk_clf(grad_reverse(content_c.mean(-1), alpha)), spk)
            if cart_pred is not None:
                iw = ivalid.unsqueeze(-1)
                cart = cart_pred(grad_reverse(content_c.mean(-1), alpha))
                c_art_adv = (F.mse_loss(cart * iw, iartic * iw, reduction="sum")
                             / (iw.sum() * N_ARTIC + 1e-6))
            cond2 = torch.cat([content_c, cond[:, 768:]], dim=1)
            m_hat = g(cond2, s, None)          # timbre AdaIN only, no s_art
            T = min(m_hat.shape[-1], mel_in.shape[-1])
            mel_l = F.l1_loss(m_hat[..., :T], mel_in[..., :T])

            w_mel = mel_weight(step, args.warmup, args.mel_hi, args.mel_floor, args.mel_anneal)
            w_mrs = 0.0 if args.no_voc_loss else mrs_weight(step, args.warmup, args.mrs_w, args.mrs_ramp)
            mrs = torch.tensor(0.0, device=DEV)
            if w_mrs > 0:
                y_hat = voc(m_hat)
                L = min(y_hat.shape[-1], y.shape[-1])
                mrs = mrstft_loss(y_hat[..., :L], y.squeeze(1)[..., :L])
            loss = (w_mel * mel_l + w_mrs * mrs + spk_ce
                    + spk_grl_w * c_adv + art_grl_w * c_art_adv)
            if not torch.isfinite(loss):
                print(f"step {step} NON-FINITE, skip", flush=True)
                opt.zero_grad(set_to_none=True); step += 1; continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            if step % 100 == 0:
                print(f"step {step} mel {mel_l.item():.3f} mrs {float(mrs.detach()):.2f} "
                      f"spk {spk_ce.item():.2f} cadv {float(c_adv):.2f} "
                      f"cartadv {float(c_art_adv):.3f} alpha {alpha:.2f}", flush=True)
            if step % args.render_every == 0 and step > 0:
                save("last.pt")
            step += 1
            if step >= args.steps:
                break
    save("last.pt")
    print(f"E1 {args.scrub} done -> {out/'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
