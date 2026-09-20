"""E5 — joint end-to-end G + FreeC (realtime vocoder), judged on the WAVEFORM.
Rationale: the roughness came from the mel middleman — G's mel has error and the
frozen vocoder is brittle to it; upsampling G's hop-512 mel to FreeC's hop-128 is
a stretch-hack with no real fine detail. Fix = drop the mel-L1 bottleneck and
train G + FreeC TOGETHER on the output sound: cond is upsampled to FreeC's rate
(hop-128), G outputs a hop-128 mel, FreeC renders it, and the loss is on the
waveform (FreeC's own MPD/MRD GAN + multiscale-mel + mrstft) so G learns mels
FreeC renders well and FreeC adapts to G. A light mel anchor stabilizes; CIPT
keeps target identity. All own weights; FreeC's GAN = the sanctioned final
texture pass. Warm-start G/t from e2_cipt, FreeC from foundation_lowlatency.
Gate: quality_gate (--freebig e5/snap, on the joint output) SI-SDR toward FreeC
ceiling 13.5; identity_probe swap-sens held; ear is final.
"""
from __future__ import annotations
import sys, os, json, argparse, hashlib
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, N_MELS, DEV
from train_m2 import TimbreEncoder, TIMBRE_DIM, ART_DIM
from mel_gen import MelGen
from train_z1 import load_freebig
from train_e3_melgan import E3Set, spk_key
from bigvgan.env import AttrDict
from bigvgan.meldataset import get_mel_spectrogram
from bigvgan.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from bigvgan.loss import MultiScaleMelSpectrogramLoss, discriminator_loss, feature_loss, generator_loss
from kansei_train import mrstft
from free_train_universal import SNAP

UP = 4  # hop-512 (G/content rate) -> hop-128 (FreeC rate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["../data/female_real_feat", "../data/female_tts_feat", "../data/male_feat"])
    ap.add_argument("--gckpt", default="checkpoints/e2_cipt/last.pt")
    ap.add_argument("--freec", default="checkpoints/freeC/foundation_lowlatency_5p8ms.pt")
    ap.add_argument("--out", default="checkpoints/e5")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam-mrstft", type=float, default=2.0)
    ap.add_argument("--lam-melanchor", type=float, default=2.0)
    ap.add_argument("--w-id", type=float, default=1.0)
    ap.add_argument("--id-ramp", type=int, default=3000)
    ap.add_argument("--id-every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=3000)
    ap.add_argument("--held-mod", type=int, default=24)
    ap.add_argument("--f0-fourier", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ds = E3Set(args.roots, args.seg)
    held_spk = {s for s in ds.by_spk if int(hashlib.md5(s.encode()).hexdigest(), 16) % args.held_mod == 0}
    ds.files = [f for f in ds.files if spk_key(f) not in held_spk]
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     drop_last=True, persistent_workers=args.workers > 0)

    gck = torch.load(args.gckpt, map_location=DEV, weights_only=False)
    ga = gck.get("args", {"dim": 384, "layers": 6})
    f0f = args.f0_fourier or ga.get("f0_fourier", 0)
    g = MelGen(cond_dim=770, dim=ga["dim"], n_layers=ga["layers"], n_mels=N_MELS,
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM, causal=False, f0_fourier=f0f).to(DEV)
    msd = g.state_dict()
    sd = {k: v for k, v in gck["g"].items() if k in msd and v.shape == msd[k].shape}
    miss = g.load_state_dict(sd, strict=False)          # in_proj fresh when f0_fourier changes its shape
    print(f"G warm: loaded {len(sd)}/{len(msd)} tensors | fresh(reinit): {list(miss.missing_keys)}", flush=True)
    t = TimbreEncoder().to(DEV); t.load_state_dict(gck["t"])
    gen, _ = load_freebig(args.freec)
    gen.train()
    for p in gen.parameters():
        p.requires_grad_(True)
    fck = torch.load(args.freec, map_location=DEV, weights_only=False)
    free_args = fck["args"]

    h = AttrDict(json.loads((SNAP / "config.json").read_text()))
    h["resolutions"] = [[1024, 256, 1024], [2048, 512, 2048], [512, 128, 512]]
    h["hop_size"] = HOP // UP
    mpd = MultiPeriodDiscriminator(h).to(DEV)
    mrd = MultiResolutionDiscriminator(h).to(DEV)
    fn_mel = MultiScaleMelSpectrogramLoss(sampling_rate=SR).to(DEV)

    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="hf_models/spkrec-ecapa", run_opts={"device": "cuda:0"})
    for p in ecapa.mods.parameters():
        p.requires_grad_(False)

    def emb(y44):
        e = ecapa.encode_batch(AF.resample(y44, SR, 16000)).squeeze(1)
        return e / (e.norm(dim=-1, keepdim=True) + 1e-6)

    gp = list(g.parameters()) + list(t.parameters()) + list(gen.parameters())
    og = torch.optim.AdamW(gp, args.lr, betas=(0.8, 0.99))
    od = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), args.lr, betas=(0.8, 0.99))
    print(f"E5 joint G+FreeC | train {len(ds.files)} | {ds.n_spk} spk | "
          f"G {sum(p.numel() for p in g.parameters())/1e6:.1f}M FreeC {sum(p.numel() for p in gen.parameters())/1e6:.1f}M "
          f"| steps {args.steps}", flush=True)

    def save(step):
        blob_g = {"g": g.state_dict(), "t": t.state_dict(), "scrub_mode": "none", "step": step,
                  "args": {"dim": ga["dim"], "layers": ga["layers"], "f0_fourier": f0f}, "upsample": UP}
        blob_v = {"gen": gen.state_dict(), "step": step, "args": free_args}
        torch.save(blob_g, out / f"g_{step}.pt"); torch.save(blob_v, out / f"snap_{step}.pt")
        torch.save(blob_g, out / "g_last.pt"); torch.save(blob_v, out / "last.pt")

    step = 0
    while step < args.steps:
        for cond, y, rw, spk in dl:
            cond, y, rw, spk = cond.to(DEV), y.to(DEV), rw.to(DEV), spk.to(DEV)
            cond4 = F.interpolate(cond, scale_factor=UP, mode="linear", align_corners=False)
            s = t(mel_of(rw))
            mel_hat = g(cond4, s, None)
            y_hat = gen(mel_hat).unsqueeze(1)
            n = min(y.shape[-1], y_hat.shape[-1]); y1, y_hat = y[..., :n], y_hat[..., :n]

            od.zero_grad(set_to_none=True)
            yr, yg, _, _ = mpd(y1, y_hat.detach()); ldf, _, _ = discriminator_loss(yr, yg)
            yr, yg, _, _ = mrd(y1, y_hat.detach()); ldr, _, _ = discriminator_loss(yr, yg)
            (ldf + ldr).backward()
            torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(mrd.parameters()), 100.0)
            od.step()

            og.zero_grad(set_to_none=True)
            with torch.no_grad():
                tgt_mel = get_mel_spectrogram(y.squeeze(1).cpu(), h).to(DEV)
            Tm = min(mel_hat.shape[-1], tgt_mel.shape[-1])
            mel_anchor = F.l1_loss(mel_hat[..., :Tm], tgt_mel[..., :Tm]) * args.lam_melanchor
            loss_fnmel = fn_mel(y1, y_hat) * 15.0
            yr, yg, fr, fg = mpd(y1, y_hat); lfm_f = feature_loss(fr, fg); lg_f, _ = generator_loss(yg)
            yr, yg, fr, fg = mrd(y1, y_hat); lfm_r = feature_loss(fr, fg); lg_r, _ = generator_loss(yg)
            loss_mr = mrstft(y1, y_hat) * args.lam_mrstft
            id_loss = torch.tensor(0.0, device=DEV)
            w_id = args.w_id * min(1.0, step / max(1, args.id_ramp))
            if w_id > 0 and step % args.id_every == 0:
                idx = torch.roll(torch.arange(cond.shape[0], device=DEV), 1)
                valid = (spk[idx] != spk).float()
                s_B = t(mel_of(rw[idx]))
                with torch.no_grad():
                    tgt_B = emb(rw[idx])
                y_cross = gen(g(cond4, s_B, None))
                idc = (emb(y_cross) * tgt_B).sum(-1)
                id_loss = ((1 - idc) * valid).sum() / (valid.sum() + 1e-6)
            g_all = lg_f + lg_r + lfm_f + lfm_r + loss_fnmel + loss_mr + mel_anchor + w_id * id_loss
            if not torch.isfinite(g_all):
                print(f"step {step} NON-FINITE skip", flush=True); step += 1; continue
            g_all.backward()
            torch.nn.utils.clip_grad_norm_(gp, 100.0)
            og.step()

            if step % 100 == 0:
                print(f"step {step} fnmel {loss_fnmel.item()/15:.3f} anch {mel_anchor.item():.3f} "
                      f"gen {(lg_f+lg_r).item():.3f} fm {(lfm_f+lfm_r).item():.3f} mr {float(loss_mr):.3f} "
                      f"d {(ldf+ldr).item():.3f} id {float(id_loss):.3f} wid {w_id:.2f}", flush=True)
            if step % args.save_every == 0 and step > 0:
                save(step)
            step += 1
            if step >= args.steps:
                break
    save(step)
    print(f"E5 done -> {out}", flush=True)


if __name__ == "__main__":
    main()
