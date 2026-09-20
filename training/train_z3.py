"""Z3 — dynamic articulatory-style clone on the real audio path (zeroshot_vc.md
§3.4/§3.5, memory: moe-articulatory-clone). THE frontier claim: RVC/So-VITS/
Beatrice stop at static timbre; we clone the ARTICULATORY STYLE s_art so a neutral
voice becomes "moe" (F1↑ / F2-range↓). Formant-domain PoC already worked
(articulatory_clone_poc.py: neutral→moe s_art swap gave ΔF1 +0.201). Z3 lifts it
to G→mel→freebig and MEASURES the formant shift on real output audio.

Mechanism (do NOT violate — memory/design §3.4):
  - s_art from the INPUT utt itself (self-articulation) so the single AdaIN learns
    s_art -> formant structure (swap it at test → formants move).
  - z_spk (timbre) from a same-speaker ref (static, speaker-constant).
  - single AdaIN fuses z_spk γ/β + s_art γ/β by ADDITION (mel_gen, art zero-init).
    NO two-stage AdaIN (cancels timbre), NO additive FiLM (clone fails).
  - ARTICULATORY-SCRUB (load-bearing): GRL removes the input's own formant summary
    from content_s (cart_pred regresses it, grad reversed) so content CANNOT supply
    articulation → G is forced to read it from s_art. Without this, s_art is ignored
    (the PoC only worked because PCA-content forced s_art necessity).
  - s_art supervised by the 5-dim moe formant signature (ArticEncoder.sup → iartic).
  - self-reconstruction, GT = real audio only (no VC teacher).
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
from train_m1 import mel_of, mrstft_loss, SR, HOP, N_MELS, DEV
from train_m2 import (MoeVCSet, TimbreEncoder, ArticEncoder, ContentScrub, grad_reverse,
                      TIMBRE_DIM, ART_DIM, N_ARTIC, ARTIC_PATH)
from mel_gen import MelGen
from train_z1 import load_freebig, mel_weight, mrs_weight, init_head_bias


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/z3_feat")
    ap.add_argument("--out", default="checkpoints/z3")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--mel-hi", type=float, default=45.0)
    ap.add_argument("--mel-floor", type=float, default=10.0)
    ap.add_argument("--mel-anneal", type=int, default=1500)
    ap.add_argument("--mrs-w", type=float, default=2.0)
    ap.add_argument("--mrs-ramp", type=int, default=800)
    ap.add_argument("--no-voc-loss", action="store_true")
    ap.add_argument("--art-scrub-w", type=float, default=0.1,
                    help="articulatory-scrub GRL weight (LOAD-BEARING; ablation=0.0)")
    ap.add_argument("--spk-scrub-w", type=float, default=0.1)
    ap.add_argument("--artic-sup-w", type=float, default=1.0)
    ap.add_argument("--spk-ce-w", type=float, default=1.0)
    ap.add_argument("--grl-ramp", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--render-every", type=int, default=2000)
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
    print(f"Z3 articulatory-clone | {len(ds)} utts | {ds.n_spk} spk | seg={args.seg} "
          f"| artic-cache {len(artic)} | art_scrub={args.art_scrub_w} | steps={args.steps}",
          flush=True)

    voc, voc_step = load_freebig(args.freebig)
    print(f"freebig frozen (step {voc_step})", flush=True)

    g = MelGen(cond_dim=768 + 2, dim=args.dim, n_layers=args.layers, n_mels=N_MELS,
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM, causal=False).to(DEV)
    t = TimbreEncoder().to(DEV)
    ea = ArticEncoder().to(DEV)
    scrub = ContentScrub().to(DEV)
    spk_clf = nn.Linear(TIMBRE_DIM, ds.n_spk).to(DEV)
    cspk_clf = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, ds.n_spk)).to(DEV)
    cart_pred = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.LeakyReLU(0.1),
                              nn.Linear(128, N_ARTIC)).to(DEV)
    mel_mean = init_head_bias(g, dl)
    print(f"MelGen {sum(p.numel() for p in g.parameters())/1e6:.2f}M | "
          f"s_art=self, z_spk=ref, single-AdaIN(add), artic-scrub GRL", flush=True)

    params = (list(g.parameters()) + list(t.parameters()) + list(ea.parameters())
              + list(scrub.parameters()) + list(spk_clf.parameters())
              + list(cspk_clf.parameters()) + list(cart_pred.parameters()))
    opt = torch.optim.AdamW(params, args.lr, betas=(0.8, 0.99))

    def save(tag: str) -> None:
        torch.save({"g": g.state_dict(), "t": t.state_dict(), "ea": ea.state_dict(),
                    "scrub": scrub.state_dict(), "mel_mean": mel_mean, "step": step,
                    "amu": amu, "asd": asd,
                    "args": {"dim": args.dim, "layers": args.layers,
                             "art_scrub_w": args.art_scrub_w}}, out / tag)

    step = 0
    while step < args.steps:
        for batch in dl:
            cond, f0, y, rw, spk = (batch[0].to(DEV), batch[1].to(DEV), batch[2].to(DEV),
                                    batch[3].to(DEV), batch[4].to(DEV))
            iartic, ivalid = batch[7].to(DEV), batch[8].to(DEV)
            mel_in = mel_of(y.squeeze(1))
            mel_ref = mel_of(rw)
            s = t(mel_ref)                       # z_spk (timbre) from ref
            s_art, artic_pred = ea(mel_in)       # s_art from INPUT (self-articulation)
            spk_ce = F.cross_entropy(spk_clf(s), spk)
            iw = ivalid.unsqueeze(-1)
            artic_l = (F.l1_loss(artic_pred * iw, iartic * iw, reduction="sum")
                       / (iw.sum() * N_ARTIC + 1e-6))
            content_s = scrub(cond[:, :768])
            alpha = 0.5 * min(1.0, step / max(1, args.grl_ramp))
            c_adv = F.cross_entropy(cspk_clf(grad_reverse(content_s.mean(-1), alpha)), spk)
            cart = cart_pred(grad_reverse(content_s.mean(-1), alpha))
            c_art_adv = (F.mse_loss(cart * iw, iartic * iw, reduction="sum")
                         / (iw.sum() * N_ARTIC + 1e-6))
            cond2 = torch.cat([content_s, cond[:, 768:]], dim=1)
            m_hat = g(cond2, s, s_art)
            T = min(m_hat.shape[-1], mel_in.shape[-1])
            mel_l = F.l1_loss(m_hat[..., :T], mel_in[..., :T])

            w_mel = mel_weight(step, args.warmup, args.mel_hi, args.mel_floor, args.mel_anneal)
            w_mrs = 0.0 if args.no_voc_loss else mrs_weight(step, args.warmup, args.mrs_w, args.mrs_ramp)
            mrs = torch.tensor(0.0, device=DEV)
            if w_mrs > 0:
                y_hat = voc(m_hat)
                L = min(y_hat.shape[-1], y.shape[-1])
                mrs = mrstft_loss(y_hat[..., :L], y.squeeze(1)[..., :L])
            loss = (w_mel * mel_l + w_mrs * mrs + args.spk_ce_w * spk_ce
                    + args.artic_sup_w * artic_l + args.spk_scrub_w * c_adv
                    + args.art_scrub_w * c_art_adv)

            if not torch.isfinite(loss):
                print(f"step {step} NON-FINITE, skip", flush=True)
                opt.zero_grad(set_to_none=True); step += 1; continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()

            if step % 50 == 0:
                print(f"step {step} mel {mel_l.item():.3f} mrs {float(mrs.detach()):.2f} "
                      f"spk {spk_ce.item():.2f} artic {artic_l.item():.3f} "
                      f"cadv {c_adv.item():.2f} cartadv {c_art_adv.item():.3f} "
                      f"alpha {alpha:.2f}", flush=True)
            if step % args.render_every == 0 and step > 0:
                save("last.pt")
            step += 1
            if step >= args.steps:
                break
    save("last.pt")
    print(f"done -> {out/'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
