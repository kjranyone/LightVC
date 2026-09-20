"""E3 — mel-domain adversarial + CIPT, FULL data. E1/E2 established: ContentVec
content is intelligible; identity needs output-side ECAPA supervision (CIPT, E2:
swap-sens 0.073->0.145); the quality ceiling is mel-L1 regression -> conditional-
mean blur (validated by PESQ: self 1.1 << ceiling 3.5, step-independent). E3 fixes
the blur with a mel-domain GAN (discriminator on mel, NOT waveform -> avoids the
through-vocoder GAN harmonic destruction) so G's mel leaves the blurry mean and
lands on the real-mel manifold; the frozen freebig (ear-approved on real mels)
then renders it cleanly. Loss = mel-L1(anchor) + mel-GAN(sharpness) + mrstft +
CIPT id_out(identity) + spk aux. FULL corpora per CLAUDE.md: female_real_feat
(2775 real) + female_tts_feat (irodori-tts 669) + male_feat.
Gate: quality_gate.py self-PESQ 1.1->3.5, identity_probe swap-sens held, ear final.
"""
from __future__ import annotations
import sys, argparse, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm
import soundfile as sf
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, mrstft_loss, SR, HOP, N_MELS, DEV
from train_m2 import TimbreEncoder, TIMBRE_DIM, ART_DIM, REF_SEC
from mel_gen import MelGen
from train_z1 import load_freebig, mrs_weight


def spk_key(f: Path) -> str:
    return f"{f.parent.parent.name}/{f.parent.name}"


class E3Set(Dataset):
    def __init__(self, roots, seg=64):
        files = []
        for r in roots:
            files += sorted(Path(r).rglob("*.pt"))
        self.by_spk = defaultdict(list)
        for f in files:
            self.by_spk[spk_key(f)].append(f)
        self.files = [f for f in files if len(self.by_spk[spk_key(f)]) >= 2]
        self.spk_to_idx = {s: i for i, s in enumerate(sorted(self.by_spk))}
        self.n_spk = len(self.spk_to_idx)
        self.seg = seg

    def __len__(self):
        return len(self.files)

    def _wav(self, p):
        w, _ = sf.read(p, dtype="float32")
        return w.mean(1) if w.ndim > 1 else w

    def __getitem__(self, i):
        f = self.files[i]
        d = torch.load(f, weights_only=False)
        if "f0" not in d or "energy" not in d:
            return self.__getitem__((i + 1) % len(self.files))
        content = d["content"].float(); f0 = d["f0"].float(); energy = d["energy"].float()
        tmel = f0.shape[0]
        y = torch.from_numpy(np.ascontiguousarray(self._wav(d["path"]))).float()[: tmel * HOP]
        if y.shape[0] < tmel * HOP:
            y = F.pad(y, (0, tmel * HOP - y.shape[0]))
        c = F.interpolate(content.t().unsqueeze(0), size=tmel, mode="linear",
                          align_corners=False).squeeze(0)
        if tmel <= self.seg:
            s = 0
            c = F.pad(c, (0, self.seg - tmel)); f0 = F.pad(f0, (0, self.seg - tmel))
            energy = F.pad(energy, (0, self.seg - tmel)); y = F.pad(y, (0, self.seg * HOP - y.shape[0]))
        else:
            s = np.random.randint(0, tmel - self.seg)
        c = c[:, s:s + self.seg]; f0 = f0[s:s + self.seg]; energy = energy[s:s + self.seg]
        y = y[s * HOP:(s + self.seg) * HOP]
        cond = torch.cat([c, (torch.log(f0.clamp(min=1.0)) / 7.0).unsqueeze(0),
                          (torch.log(energy.clamp(min=1e-4)) * 0.2).unsqueeze(0)], dim=0)
        spk = spk_key(f)
        rf = self.by_spk[spk][np.random.randint(len(self.by_spk[spk]))]
        rw = torch.from_numpy(np.ascontiguousarray(self._wav(torch.load(rf, weights_only=False)["path"]))).float()
        rn = int(REF_SEC * SR)
        rw = rw[:rn] if rw.shape[0] >= rn else F.pad(rw, (0, rn - rw.shape[0]))
        return cond, y.unsqueeze(0), rw, self.spk_to_idx[spk]


class MelDisc(nn.Module):
    def __init__(self, n_mels=N_MELS):
        super().__init__()
        def blk(ci, co, k, s):
            return spectral_norm(nn.Conv2d(ci, co, k, s, (k[0] // 2, k[1] // 2)))
        self.net = nn.ModuleList([
            blk(1, 32, (3, 9), (1, 1)), blk(32, 64, (3, 9), (1, 2)),
            blk(64, 128, (3, 9), (2, 2)), blk(128, 256, (3, 5), (2, 2)),
            blk(256, 256, (3, 3), (1, 1))])
        self.out = spectral_norm(nn.Conv2d(256, 1, (3, 3), 1, 1))

    def forward(self, mel):
        x = mel.unsqueeze(1)
        fmaps = []
        for l in self.net:
            x = F.leaky_relu(l(x), 0.1); fmaps.append(x)
        return self.out(x), fmaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["../data/female_real_feat", "../data/male_feat"])
    ap.add_argument("--init", default="checkpoints/e2_cipt/last.pt")
    ap.add_argument("--out", default="checkpoints/e3")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--d-lr", type=float, default=1e-4)
    ap.add_argument("--w-mel", type=float, default=8.0)
    ap.add_argument("--w-gan", type=float, default=0.5)
    ap.add_argument("--w-fm", type=float, default=2.0)
    ap.add_argument("--gan-warmup", type=int, default=2000)
    ap.add_argument("--mrs-w", type=float, default=1.0)
    ap.add_argument("--w-id", type=float, default=1.5)
    ap.add_argument("--id-ramp", type=int, default=3000)
    ap.add_argument("--id-every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=4000)
    ap.add_argument("--held-mod", type=int, default=24)
    ap.add_argument("--dim", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ds = E3Set(args.roots, args.seg)
    held_spk = {s for s in ds.by_spk if int(hashlib.md5(s.encode()).hexdigest(), 16) % args.held_mod == 0}
    held = [f for s in held_spk for f in ds.by_spk[s]]
    (out / "heldout.txt").write_text("\n".join(str(f) for f in held))
    ds.files = [f for f in ds.files if spk_key(f) not in held_spk]
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0)
    print(f"E3 | roots {args.roots} | train {len(ds.files)} held {len(held)}({len(held_spk)}spk) "
          f"| {ds.n_spk} spk | seg {args.seg} batch {args.batch} steps {args.steps}", flush=True)

    voc, _ = load_freebig(args.freebig)
    ck = torch.load(args.init, map_location=DEV, weights_only=False) if Path(args.init).exists() else None
    ga = (ck or {}).get("args", {"dim": 384, "layers": 6})
    g = MelGen(cond_dim=770, dim=ga["dim"], n_layers=ga["layers"], n_mels=N_MELS,
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM, causal=False).to(DEV)
    t = TimbreEncoder().to(DEV)
    if ck:
        g.load_state_dict(ck["g"]); t.load_state_dict(ck["t"])
        print(f"warm-start G,t from {args.init}", flush=True)
    D = MelDisc().to(DEV)
    spk_clf = nn.Linear(TIMBRE_DIM, ds.n_spk).to(DEV)

    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="hf_models/spkrec-ecapa", run_opts={"device": "cuda:0"})
    for p in ecapa.mods.parameters():
        p.requires_grad_(False)

    def emb(y44):
        e = ecapa.encode_batch(AF.resample(y44, SR, 16000)).squeeze(1)
        return e / (e.norm(dim=-1, keepdim=True) + 1e-6)

    gp = list(g.parameters()) + list(t.parameters()) + list(spk_clf.parameters())
    opt_g = torch.optim.AdamW(gp, args.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(D.parameters(), args.d_lr, betas=(0.8, 0.99))

    def save(tag, step):
        torch.save({"g": g.state_dict(), "t": t.state_dict(), "D": D.state_dict(),
                    "step": step, "scrub_mode": "none",
                    "args": {"dim": ga["dim"], "layers": ga["layers"]}}, out / tag)
        if tag == "last.pt":
            torch.save({"g": g.state_dict(), "t": t.state_dict(), "mel_mean": None,
                        "step": step, "scrub_mode": "none",
                        "args": {"dim": ga["dim"], "layers": ga["layers"]}}, out / f"snap_{step}.pt")

    step = 0
    while step < args.steps:
        for cond, y, rw, spk in dl:
            cond, y, rw, spk = cond.to(DEV), y.to(DEV), rw.to(DEV), spk.to(DEV)
            mel_r = mel_of(y.squeeze(1))
            s_A = t(mel_of(rw))
            m_fake = g(cond, s_A, None)
            T = min(m_fake.shape[-1], mel_r.shape[-1])
            m_fake, mel_r = m_fake[..., :T], mel_r[..., :T]
            w_gan = args.w_gan * min(1.0, max(0, step - args.gan_warmup) / 2000)

            # D step
            if w_gan > 0:
                d_r, _ = D(mel_r.detach()); d_f, _ = D(m_fake.detach())
                loss_d = (F.relu(1 - d_r).mean() + F.relu(1 + d_f).mean())
                opt_d.zero_grad(set_to_none=True); loss_d.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0); opt_d.step()
            else:
                loss_d = torch.tensor(0.0)

            # G step
            mel_l = F.l1_loss(m_fake, mel_r)
            spk_ce = F.cross_entropy(spk_clf(s_A), spk)
            g_adv = torch.tensor(0.0, device=DEV); fm = torch.tensor(0.0, device=DEV)
            if w_gan > 0:
                d_f, fmap_f = D(m_fake); _, fmap_r = D(mel_r.detach())
                g_adv = -d_f.mean()
                fm = sum(F.l1_loss(a, b.detach()) for a, b in zip(fmap_f, fmap_r)) / len(fmap_f)
            y_self = voc(m_fake); L = min(y_self.shape[-1], y.shape[-1])
            mrs = mrstft_loss(y_self[..., :L], y.squeeze(1)[..., :L])
            # CIPT cross-id (every id_every steps to save compute)
            id_loss = torch.tensor(0.0, device=DEV)
            w_id = args.w_id * min(1.0, step / max(1, args.id_ramp))
            if w_id > 0 and step % args.id_every == 0:
                idx = torch.roll(torch.arange(cond.shape[0], device=DEV), 1)
                valid = (spk[idx] != spk).float()
                s_B = t(mel_of(rw[idx]))
                with torch.no_grad():
                    tgt_B = emb(rw[idx])
                y_cross = voc(g(cond, s_B, None))
                idc = (emb(y_cross) * tgt_B).sum(-1)
                id_loss = ((1 - idc) * valid).sum() / (valid.sum() + 1e-6)
            loss_g = (args.w_mel * mel_l + w_gan * g_adv + args.w_fm * (fm if w_gan > 0 else 0)
                      + args.mrs_w * mrs + spk_ce + w_id * id_loss)
            if not torch.isfinite(loss_g):
                print(f"step {step} NON-FINITE skip", flush=True); step += 1; continue
            opt_g.zero_grad(set_to_none=True); loss_g.backward()
            torch.nn.utils.clip_grad_norm_(gp, 5.0); opt_g.step()

            if step % 100 == 0:
                print(f"step {step} mel {mel_l.item():.3f} gan {float(g_adv):.3f} fm {float(fm):.3f} "
                      f"d {float(loss_d):.3f} mrs {float(mrs):.2f} id {float(id_loss):.3f} "
                      f"wg {w_gan:.2f} wid {w_id:.2f}", flush=True)
            if step % args.save_every == 0 and step > 0:
                save("last.pt", step)
            step += 1
            if step >= args.steps:
                break
    save("last.pt", step)
    print(f"E3 done -> {out/'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
