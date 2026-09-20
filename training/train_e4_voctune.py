"""E4 — vocoder-robustification fine-tune. Diagnosis (2026-07-20, see
quality-gate-pesq memory): the self roughness is NOT blur/info-loss (G's mel has
corr 0.92 & full variance; killing fine detail is harmless). The real cause is
that frozen freebig is BRITTLE to mel-prediction error — a corr-0.92 mel (real+
noise OR G's) crashes it 12.8->1.2 SI-SDR (the classic acoustic-model-mel <->
vocoder mismatch). Fix = fine-tune freebig on G's PREDICTED mels -> gt waveform
with freebig's own MPD/MRD GAN, so it renders G's imperfect mel cleanly. G+t are
FROZEN (identity/content already handled by e2_cipt CIPT); this is purely the
"final texture fine-tune" the CLAUDE.md rules reserve for the vocoder GAN.

Eval: render with --ckpt e2_cipt --freebig checkpoints/e4/snap_*.pt, then
quality_gate.py (fixed N=24 held, SI-SDR/sq-PESQ) + spectrogram; ear is final.
"""
from __future__ import annotations
import sys, os, json, argparse
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, N_MELS, DEV
from train_m2 import TimbreEncoder, TIMBRE_DIM, ART_DIM
from mel_gen import MelGen
from train_z1 import load_freebig
from train_e3_melgan import E3Set
from bigvgan.env import AttrDict
from bigvgan.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from bigvgan.loss import MultiScaleMelSpectrogramLoss, discriminator_loss, feature_loss, generator_loss
from kansei_train import mrstft
from free_train_universal import SNAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["../data/female_real_feat", "../data/female_tts_feat", "../data/male_feat"])
    ap.add_argument("--gckpt", default="checkpoints/e2_cipt/last.pt")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--out", default="checkpoints/e4")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam-mrstft", type=float, default=2.0)
    ap.add_argument("--noise-aug", type=float, default=0.0)
    ap.add_argument("--upsample", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=3000)
    ap.add_argument("--held-mod", type=int, default=24)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ds = E3Set(args.roots, args.seg)
    import hashlib
    from train_e3_melgan import spk_key
    held_spk = {s for s in ds.by_spk if int(hashlib.md5(s.encode()).hexdigest(), 16) % args.held_mod == 0}
    ds.files = [f for f in ds.files if spk_key(f) not in held_spk]
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     drop_last=True, persistent_workers=args.workers > 0)

    gck = torch.load(args.gckpt, map_location=DEV, weights_only=False)
    ga = gck.get("args", {"dim": 384, "layers": 6})
    g = MelGen(cond_dim=770, dim=ga["dim"], n_layers=ga["layers"], n_mels=N_MELS,
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM, causal=False).to(DEV)
    g.load_state_dict(gck["g"]); g.eval()
    t = TimbreEncoder().to(DEV); t.load_state_dict(gck["t"]); t.eval()
    for m in (g, t):
        for p in m.parameters():
            p.requires_grad_(False)

    gen, _ = load_freebig(args.freebig)          # freebig; make trainable
    gen.train()
    for p in gen.parameters():
        p.requires_grad_(True)
    fck = torch.load(args.freebig, map_location=DEV, weights_only=False)
    free_args = fck["args"]

    h = AttrDict(json.loads((SNAP / "config.json").read_text()))
    h["resolutions"] = [[1024, 256, 1024], [2048, 512, 2048], [512, 128, 512]]
    h["hop_size"] = HOP
    mpd = MultiPeriodDiscriminator(h).to(DEV)
    mrd = MultiResolutionDiscriminator(h).to(DEV)
    fn_mel = MultiScaleMelSpectrogramLoss(sampling_rate=SR).to(DEV)
    og = torch.optim.AdamW(gen.parameters(), args.lr, betas=(0.8, 0.99))
    od = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), args.lr, betas=(0.8, 0.99))
    print(f"E4 voctune | roots {args.roots} | train {len(ds.files)} | {ds.n_spk} spk | "
          f"freebig {sum(p.numel() for p in gen.parameters())/1e6:.1f}M | steps {args.steps}", flush=True)

    def save(step):
        torch.save({"gen": gen.state_dict(), "step": step, "args": free_args}, out / f"snap_{step}.pt")
        torch.save({"gen": gen.state_dict(), "step": step, "args": free_args}, out / "last.pt")

    step = 0
    while step < args.steps:
        for cond, y, rw, spk in dl:
            cond, y, rw = cond.to(DEV), y.to(DEV), rw.to(DEV)
            with torch.no_grad():
                s = t(mel_of(rw))
                mel_hat = g(cond, s, None)
                if args.upsample > 1:
                    mel_hat = F.interpolate(mel_hat, scale_factor=args.upsample, mode="linear", align_corners=False)
                if args.noise_aug > 0:
                    mel_hat = mel_hat + torch.randn_like(mel_hat) * args.noise_aug * mel_hat.std()
            y_hat = gen(mel_hat).unsqueeze(1)
            n = min(y.shape[-1], y_hat.shape[-1])
            y, y_hat = y[..., :n], y_hat[..., :n]

            od.zero_grad(set_to_none=True)
            yr, yg, _, _ = mpd(y, y_hat.detach()); ldf, _, _ = discriminator_loss(yr, yg)
            yr, yg, _, _ = mrd(y, y_hat.detach()); ldr, _, _ = discriminator_loss(yr, yg)
            (ldf + ldr).backward()
            torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(mrd.parameters()), 100.0)
            od.step()

            og.zero_grad(set_to_none=True)
            loss_mel = fn_mel(y, y_hat) * 15.0
            yr, yg, fr, fg = mpd(y, y_hat); lfm_f = feature_loss(fr, fg); lg_f, _ = generator_loss(yg)
            yr, yg, fr, fg = mrd(y, y_hat); lfm_r = feature_loss(fr, fg); lg_r, _ = generator_loss(yg)
            loss_mr = mrstft(y, y_hat) * args.lam_mrstft
            g_all = lg_f + lg_r + lfm_f + lfm_r + loss_mel + loss_mr
            if not torch.isfinite(g_all):
                print(f"step {step} NON-FINITE skip", flush=True); step += 1; continue
            g_all.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 100.0)
            og.step()

            if step % 100 == 0:
                print(f"step {step} mel {loss_mel.item()/15:.3f} gen {(lg_f+lg_r).item():.3f} "
                      f"fm {(lfm_f+lfm_r).item():.3f} mr {float(loss_mr):.3f} d {(ldf+ldr).item():.3f}", flush=True)
            if step % args.save_every == 0 and step > 0:
                save(step)
            step += 1
            if step >= args.steps:
                break
    save(step)
    print(f"E4 done -> {out/'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
