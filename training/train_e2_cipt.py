"""E2-CIPT: output-side identity supervision on the current front-end (MelGen G +
frozen freebig). E1 showed self-recon+AdaIN injects only ~23% of target identity
(swap-sens 0.073) and scaling to 2775 real speakers made it WORSE (0.012) with
mean-collapse: the model is never asked to put identity Y onto content X, so it
ignores the timbre ref. This adds a cross-speaker id_out loss: for (A-content,
timbre-B) the OUTPUT waveform's ECAPA embedding is pushed toward B, differentiably
through freebig. A self-recon arm (A-content, timbre-A -> mel-L1 + mrstft) anchors
quality/content. Warm-start from e1_none on the same rcav held split so the
identity_probe swap-sensitivity is directly comparable to the 0.073 baseline.

Recipe from [[cipt-cross-identity-plan]] (broke the SECS 0.5 ceiling, ear-verified).
GAN + ContentVec-match anti-fooling anchors deferred to C2 unless the spectrogram
shows ECAPA fooling (numbers up, structure garbled).
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
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, mrstft_loss, SR, HOP, N_MELS, DEV
from train_m2 import (MoeVCSet, TimbreEncoder, TIMBRE_DIM, ART_DIM, N_ARTIC)
from mel_gen import MelGen
from train_z1 import load_freebig, mrs_weight
from train_e1 import is_held


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/e1_none/last.pt")
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--out", default="checkpoints/e2_cipt")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-mel", type=float, default=10.0)
    ap.add_argument("--mrs-w", type=float, default=2.0)
    ap.add_argument("--w-id", type=float, default=1.5)
    ap.add_argument("--id-ramp", type=int, default=800)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = MoeVCSet(args.feat, args.seg, {}, np.zeros(N_ARTIC, np.float32), np.ones(N_ARTIC, np.float32))
    held = [f for f in ds.files if is_held(f)]
    for spk in list(ds.by_spk):
        ds.by_spk[spk] = [f for f in ds.by_spk[spk] if not is_held(f)]
    ds.files = [f for f in ds.files if not is_held(f) and len(ds.by_spk[f.parent.name]) >= 2]
    (out / "heldout.txt").write_text("\n".join(str(f) for f in held))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0)

    ck = torch.load(args.init, map_location=DEV, weights_only=False)
    ga = ck.get("args", {"dim": 384, "layers": 6})
    g = MelGen(cond_dim=770, dim=ga["dim"], n_layers=ga["layers"], n_mels=N_MELS,
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM, causal=False).to(DEV)
    g.load_state_dict(ck["g"])
    t = TimbreEncoder().to(DEV); t.load_state_dict(ck["t"])
    mel_mean = ck.get("mel_mean")
    voc, _ = load_freebig(args.freebig)

    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="hf_models/spkrec-ecapa",
                                           run_opts={"device": "cuda:0"})
    for p in ecapa.mods.parameters():
        p.requires_grad_(False)

    def ecapa_emb(y44):
        y16 = AF.resample(y44, SR, 16000)
        e = ecapa.encode_batch(y16).squeeze(1)
        return e / (e.norm(dim=-1, keepdim=True) + 1e-6)

    spk_clf = nn.Linear(TIMBRE_DIM, ds.n_spk).to(DEV)
    params = list(g.parameters()) + list(t.parameters()) + list(spk_clf.parameters())
    opt = torch.optim.AdamW(params, args.lr, betas=(0.8, 0.99))

    def save(tag):
        torch.save({"g": g.state_dict(), "t": t.state_dict(), "mel_mean": mel_mean,
                    "step": step, "scrub_mode": "none",
                    "args": {"dim": ga["dim"], "layers": ga["layers"]}}, out / tag)

    print(f"E2-CIPT warm {args.init} | train {len(ds.files)} held {len(held)} | {ds.n_spk} spk "
          f"| seg {args.seg} batch {args.batch} steps {args.steps} w_id {args.w_id}", flush=True)

    step = 0
    while step < args.steps:
        for batch in dl:
            cond, y, rw, spk = (batch[0].to(DEV), batch[2].to(DEV),
                                batch[3].to(DEV), batch[4].to(DEV))
            mel_in = mel_of(y.squeeze(1))
            s_A = t(mel_of(rw))
            spk_ce = F.cross_entropy(spk_clf(s_A), spk)

            m_self = g(cond, s_A, None)
            T = min(m_self.shape[-1], mel_in.shape[-1])
            mel_l = F.l1_loss(m_self[..., :T], mel_in[..., :T])
            w_mrs = mrs_weight(step, 0, args.mrs_w, 1)
            y_self = voc(m_self)
            L = min(y_self.shape[-1], y.shape[-1])
            mrs = mrstft_loss(y_self[..., :L], y.squeeze(1)[..., :L])

            idx = torch.roll(torch.arange(cond.shape[0], device=DEV), 1)
            rw_B, spk_B = rw[idx], spk[idx]
            valid = (spk_B != spk).float()
            s_B = t(mel_of(rw_B))
            with torch.no_grad():
                tgt_B = ecapa_emb(rw_B)
            m_cross = g(cond, s_B, None)
            y_cross = voc(m_cross)
            emb_cross = ecapa_emb(y_cross)
            idcos = (emb_cross * tgt_B).sum(-1)
            id_loss = ((1 - idcos) * valid).sum() / (valid.sum() + 1e-6)

            w_id = args.w_id * min(1.0, step / max(1, args.id_ramp))
            loss = args.w_mel * mel_l + w_mrs * mrs + spk_ce + w_id * id_loss
            if not torch.isfinite(loss):
                print(f"step {step} NON-FINITE skip", flush=True)
                opt.zero_grad(set_to_none=True); step += 1; continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            if step % 50 == 0:
                print(f"step {step} mel {mel_l.item():.3f} mrs {float(mrs):.2f} "
                      f"idcos {float((idcos*valid).sum()/(valid.sum()+1e-6)):.3f} "
                      f"idloss {id_loss.item():.3f} w_id {w_id:.2f} spk {spk_ce.item():.2f}", flush=True)
            if step % args.save_every == 0 and step > 0:
                save("last.pt")
            step += 1
            if step >= args.steps:
                break
    save("last.pt")
    print(f"E2-CIPT done -> {out/'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
