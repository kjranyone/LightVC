from __future__ import annotations

import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from torch.nn.utils import weight_norm

sys.path.insert(0, str(Path(__file__).parent))
from nsf_hn import NsfHifiGan
from train_m1 import (mel_of, mrstft_loss, fm_loss, stft_mag, PeriodDisc, ScaleDisc,
                      SR, HOP, N_MELS, DEV)

TIMBRE_DIM = 192
ART_DIM = 48
N_ARTIC = 5
ARTIC_PATH = "../data/artic_feats.pt"
REF_SEC = 2.0


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.alpha * g, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


class ContentScrub(nn.Module):
    def __init__(self, dim: int = 768, hidden: int = 512) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(dim, hidden, 1)
        self.c2 = nn.Conv1d(hidden, dim, 1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return c + self.c2(F.leaky_relu(self.c1(c), 0.1))


class TimbreEncoder(nn.Module):
    def __init__(self, n_mels: int = N_MELS, dim: int = TIMBRE_DIM) -> None:
        super().__init__()
        ch = 256
        self.convs = nn.Sequential(
            weight_norm(nn.Conv1d(n_mels, ch, 5, 1, 2)), nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(ch, ch, 5, 2, 2)), nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(ch, ch, 5, 1, 2)), nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(ch, ch, 5, 2, 2)), nn.LeakyReLU(0.1))
        self.proj = nn.Linear(ch * 2, dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        h = self.convs(mel)
        stats = torch.cat([h.mean(-1), h.std(-1) + 1e-5], dim=-1)
        return self.proj(stats)


class ArticEncoder(nn.Module):
    def __init__(self, n_mels: int = N_MELS, dim: int = ART_DIM) -> None:
        super().__init__()
        ch = 192
        self.convs = nn.Sequential(
            weight_norm(nn.Conv1d(n_mels, ch, 5, 1, 2)), nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(ch, ch, 5, 2, 2)), nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(ch, ch, 5, 1, 2)), nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(ch, ch, 5, 2, 2)), nn.LeakyReLU(0.1))
        self.proj = nn.Linear(ch * 2, dim)
        self.sup = nn.Linear(dim, N_ARTIC)

    def forward(self, mel: torch.Tensor) -> tuple:
        h = self.convs(mel)
        stats = torch.cat([h.mean(-1), h.std(-1) + 1e-5], dim=-1)
        code = self.proj(stats)
        return code, self.sup(code)


class SpecDisc(nn.Module):
    def __init__(self, n_fft: int, hop: int, win: int) -> None:
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1)))])
        self.post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> tuple:
        m = stft_mag(x.squeeze(1), self.n_fft, self.hop, self.win)
        m = torch.log(m).unsqueeze(1)
        fmap = []
        for c in self.convs:
            m = F.leaky_relu(c(m), 0.1)
            fmap.append(m)
        m = self.post(m)
        fmap.append(m)
        return m.flatten(1), fmap


class Discriminator2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mpd = nn.ModuleList([PeriodDisc(p) for p in (2, 3, 5, 7, 11)])
        self.msd = nn.ModuleList([ScaleDisc() for _ in range(3)])
        self.pools = nn.ModuleList([nn.AvgPool1d(4, 2, 2) for _ in range(2)])
        self.mrd = nn.ModuleList([SpecDisc(n, h, w) for n, h, w in
                                  [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]])

    def forward(self, x: torch.Tensor) -> tuple:
        outs, fmaps = [], []
        for d in self.mpd:
            o, f = d(x)
            outs.append(o)
            fmaps.append(f)
        y = x
        for i, d in enumerate(self.msd):
            if i > 0:
                y = self.pools[i - 1](y)
            o, f = d(y)
            outs.append(o)
            fmaps.append(f)
        for d in self.mrd:
            o, f = d(x)
            outs.append(o)
            fmaps.append(f)
        return outs, fmaps


class MoeVCSet(Dataset):
    def __init__(self, feat_root: str, seg_frames: int = 32, artic: dict = None,
                 amu: np.ndarray = None, asd: np.ndarray = None) -> None:
        files = sorted(Path(feat_root).rglob("*.pt"))
        self.by_spk = defaultdict(list)
        for f in files:
            self.by_spk[f.parent.name].append(f)
        self.files = [f for f in files if len(self.by_spk[f.parent.name]) >= 2]
        self.spk_to_idx = {s: i for i, s in enumerate(sorted(self.by_spk))}
        self.n_spk = len(self.spk_to_idx)
        self.seg = seg_frames
        self.artic = artic or {}
        self.amu = amu if amu is not None else np.zeros(N_ARTIC, np.float32)
        self.asd = asd if asd is not None else np.ones(N_ARTIC, np.float32)

    def __len__(self) -> int:
        return len(self.files)

    def _load_wav(self, path: str) -> torch.Tensor:
        w, _ = sf.read(path, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        return torch.from_numpy(np.ascontiguousarray(w)).float()

    def __getitem__(self, i: int) -> tuple:
        d = torch.load(self.files[i], weights_only=False)
        if "f0" not in d or "energy" not in d:
            return self.__getitem__((i + 1) % len(self.files))
        if d.get("content_pert"):
            content = random.choice(d["content_pert"]).float()
        else:
            content = d["content"].float()
        f0 = d["f0"].float()
        energy = d["energy"].float()
        y = self._load_wav(d["path"])
        tmel = f0.shape[0]
        y = y[: tmel * HOP]
        if y.shape[0] < tmel * HOP:
            y = F.pad(y, (0, tmel * HOP - y.shape[0]))
        c = F.interpolate(content.t().unsqueeze(0), size=tmel, mode="linear",
                          align_corners=False).squeeze(0)
        if tmel <= self.seg:
            s = 0
            c = F.pad(c, (0, self.seg - tmel))
            f0 = F.pad(f0, (0, self.seg - tmel))
            energy = F.pad(energy, (0, self.seg - tmel))
            y = F.pad(y, (0, self.seg * HOP - y.shape[0]))
        else:
            s = random.randint(0, tmel - self.seg)
        c = c[:, s:s + self.seg]
        f0 = f0[s:s + self.seg]
        energy = energy[s:s + self.seg]
        y = y[s * HOP:(s + self.seg) * HOP]
        logf0 = torch.log(f0.clamp(min=1.0)) / 7.0
        eng = torch.log(energy.clamp(min=1e-4)) * 0.2
        cond = torch.cat([c, logf0.unsqueeze(0), eng.unsqueeze(0)], dim=0)

        spk = self.files[i].parent.name
        ref_file = random.choice([f for f in self.by_spk[spk] if f != self.files[i]])
        rd = torch.load(ref_file, weights_only=False)
        rw = self._load_wav(rd["path"])
        rn = int(REF_SEC * SR)
        if rw.shape[0] > rn:
            rs = random.randint(0, rw.shape[0] - rn)
            rw = rw[rs:rs + rn]
        else:
            rw = F.pad(rw, (0, rn - rw.shape[0]))
        def lookup(path):
            a = self.artic.get(str(path))
            if a is None:
                return torch.zeros(N_ARTIC), 0.0
            return torch.from_numpy(((a - self.amu) / self.asd).astype(np.float32)), 1.0
        artic, avalid = lookup(rd["path"])
        iartic, ivalid = lookup(d["path"])
        return (cond, f0, y.unsqueeze(0), rw, self.spk_to_idx[spk],
                artic, avalid, iartic, ivalid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--out", default="checkpoints/m2_vc")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gan-after", type=int, default=1000)
    ap.add_argument("--render-every", type=int, default=2000)
    ap.add_argument("--init-g", default="checkpoints/m1_v2/last.pt")
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
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=6,
                    drop_last=True, persistent_workers=True)
    print(f"M2 conditional VC | {len(ds)} utts | timbre={TIMBRE_DIM} art={ART_DIM} "
          f"| artic {len(artic)} | steps={args.steps}")

    g = NsfHifiGan(cond_dim=768 + 2, timbre_dim=TIMBRE_DIM, art_dim=ART_DIM).to(DEV)
    if args.init_g and Path(args.init_g).exists():
        sd = torch.load(args.init_g, map_location=DEV, weights_only=False)["g"]
        own = g.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        g.load_state_dict(own)
        print(f"  warm-started G from {args.init_g}: {loaded}/{len(own)} tensors")
    t = TimbreEncoder().to(DEV)
    ea = ArticEncoder().to(DEV)
    spk_clf = nn.Linear(TIMBRE_DIM, ds.n_spk).to(DEV)
    scrub = ContentScrub().to(DEV)
    cspk_clf = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, ds.n_spk)).to(DEV)
    cart_pred = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 128), nn.LeakyReLU(0.1),
                              nn.Linear(128, N_ARTIC)).to(DEV)
    d = Discriminator2().to(DEV)
    g_params = (list(g.parameters()) + list(t.parameters()) + list(ea.parameters())
                + list(spk_clf.parameters()) + list(scrub.parameters())
                + list(cspk_clf.parameters()) + list(cart_pred.parameters()))
    og = torch.optim.AdamW(g_params, args.lr, betas=(0.8, 0.99))
    od = torch.optim.AdamW(d.parameters(), args.lr, betas=(0.8, 0.99))
    print(f"  speaker-aux + content-adv (GRL) + articulatory-style AdaIN(sup)")

    step = 0
    while step < args.steps:
        for cond, f0, y, rw, spk, artic, avalid, iartic, ivalid in dl:
            cond, f0, y, rw, spk = (cond.to(DEV), f0.to(DEV), y.to(DEV),
                                    rw.to(DEV), spk.to(DEV))
            artic, avalid = artic.to(DEV), avalid.to(DEV)
            iartic, ivalid = iartic.to(DEV), ivalid.to(DEV)
            mel_ref = mel_of(rw)
            s = t(mel_ref)
            s_art, artic_pred = ea(mel_ref)
            spk_ce = F.cross_entropy(spk_clf(s), spk)
            aw = avalid.unsqueeze(-1)
            artic_l = (F.l1_loss(artic_pred * aw, artic * aw, reduction="sum")
                       / (aw.sum() * N_ARTIC + 1e-6))
            content_s = scrub(cond[:, :768])
            grl_alpha = 0.5 * min(1.0, step / 10000.0)
            c_adv = F.cross_entropy(cspk_clf(grad_reverse(content_s.mean(-1), grl_alpha)), spk)
            iw = ivalid.unsqueeze(-1)
            cart = cart_pred(grad_reverse(content_s.mean(-1), grl_alpha))
            c_art_adv = (F.mse_loss(cart * iw, iartic * iw, reduction="sum")
                         / (iw.sum() * N_ARTIC + 1e-6))
            cond2 = torch.cat([content_s, cond[:, 768:]], dim=1)
            y_hat = g(cond2, f0, s, s_art)[..., : y.shape[-1]]
            mel_l = F.l1_loss(mel_of(y_hat.squeeze(1)), mel_of(y.squeeze(1)))
            mrs = mrstft_loss(y_hat.squeeze(1), y.squeeze(1))
            use_gan = step >= args.gan_after

            if use_gan:
                od.zero_grad()
                dr, _ = d(y)
                dg, _ = d(y_hat.detach())
                d_loss = sum(((r - 1) ** 2).mean() + (gg ** 2).mean() for r, gg in zip(dr, dg))
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(d.parameters(), 10.0)
                od.step()

            og.zero_grad()
            g_loss = (45.0 * mel_l + 2.0 * mrs + 1.0 * spk_ce + 0.1 * c_adv
                      + 1.0 * artic_l + 0.1 * c_art_adv)
            if use_gan:
                dg, fg = d(y_hat)
                dr, fr = d(y)
                g_adv = sum(((gg - 1) ** 2).mean() for gg in dg)
                g_loss = g_loss + g_adv + 2.0 * fm_loss(fr, fg)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(g_params, 10.0)
            og.step()

            if step % 50 == 0:
                msg = (f"step {step} mel {mel_l.item():.3f} mrs {mrs.item():.3f} "
                       f"spk {spk_ce.item():.2f} cadv {c_adv.item():.2f} "
                       f"artic {artic_l.item():.3f} cartadv {c_art_adv.item():.3f}")
                if use_gan:
                    msg += f" g_adv {g_adv.item():.3f} d {d_loss.item():.3f}"
                print(msg, flush=True)
            if step % args.render_every == 0:
                torch.save({"g": g.state_dict(), "t": t.state_dict(), "scrub": scrub.state_dict(), "ea": ea.state_dict(), "amu": amu, "asd": asd, "step": step}, out / "last.pt")
            step += 1
            if step >= args.steps:
                break
    torch.save({"g": g.state_dict(), "t": t.state_dict(), "scrub": scrub.state_dict(), "ea": ea.state_dict(), "amu": amu, "asd": asd, "step": step}, out / "last.pt")
    print("done")


if __name__ == "__main__":
    main()
