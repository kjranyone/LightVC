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
from train_m1 import mel_of, mrstft_loss, fm_loss, SR, HOP, N_MELS, DEV
from train_m2 import (TimbreEncoder, ContentScrub, Discriminator2,
                      grad_reverse, TIMBRE_DIM)

EMB_DIM = 192
ECAPA_PATH = "../data/ecapa_emb.pt"
REF_SEC = 2.0


class F0Predictor(nn.Module):
    def __init__(self, cdim: int = 768, edim: int = EMB_DIM, h: int = 256) -> None:
        super().__init__()
        self.emb = nn.Linear(edim, h)
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(cdim + h, h, 5, 1, 2)), nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(h, h, 5, 1, 2)), nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(h, h, 5, 1, 2)), nn.LeakyReLU(0.1))
        self.out = nn.Conv1d(h, 2, 1)

    def forward(self, content: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        t = content.shape[-1]
        e = self.emb(emb).unsqueeze(-1).expand(-1, -1, t)
        h = self.net(torch.cat([content, e], 1))
        return self.out(h)


class M3Set(Dataset):
    def __init__(self, feat_root: str, ecapa: dict, seg_frames: int = 32) -> None:
        files = []
        for root in str(feat_root).split(","):
            files.extend(sorted(Path(root).rglob("*.pt")))
        self.by_spk = defaultdict(list)
        for f in files:
            self.by_spk[str(f.parent)].append(f)
        self.files = [f for f in files if len(self.by_spk[str(f.parent)]) >= 2]
        self.seg = seg_frames
        self.ecapa = ecapa

    def __len__(self) -> int:
        return len(self.files)

    def _wav(self, path: str) -> torch.Tensor:
        w, _ = sf.read(path, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        return torch.from_numpy(np.ascontiguousarray(w)).float()

    def __getitem__(self, i: int) -> tuple:
        d = torch.load(self.files[i], weights_only=False)
        if "f0" not in d or "energy" not in d:
            return self.__getitem__((i + 1) % len(self.files))
        _v = d["f0"][d["f0"] > 1]
        if _v.numel() > 5 and float(_v.median()) > 450.0:
            return self.__getitem__((i + 1) % len(self.files))
        if d.get("content_pert") and random.random() < 0.5:
            content = random.choice(d["content_pert"]).float()
        else:
            content = d["content"].float()
        f0 = d["f0"].float()
        energy = d["energy"].float()
        y = self._wav(d["path"])
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
        voiced = (f0 > 1.0).float()
        eng = torch.log(energy.clamp(min=1e-4)) * 0.2
        cond = torch.cat([c, logf0.unsqueeze(0), eng.unsqueeze(0)], dim=0)

        spk = str(self.files[i].parent)
        ref_file = random.choice([f for f in self.by_spk[spk] if f != self.files[i]])
        rd = torch.load(ref_file, weights_only=False)
        rw = self._wav(rd["path"])
        rn = int(REF_SEC * SR)
        if rw.shape[0] > rn:
            rs = random.randint(0, rw.shape[0] - rn)
            rw = rw[rs:rs + rn]
        else:
            rw = F.pad(rw, (0, rn - rw.shape[0]))
        emb = self.ecapa.get(str(rd["path"]))
        if emb is None:
            emb = np.zeros(EMB_DIM, np.float32)
            evalid = 0.0
        else:
            evalid = 1.0
        return (cond, f0, logf0, voiced, y.unsqueeze(0), rw,
                torch.from_numpy(np.asarray(emb, np.float32)), evalid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--out", default="checkpoints/m3")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gan-after", type=int, default=1000)
    ap.add_argument("--init-g", default="checkpoints/m2_vc6/last.pt")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ecapa = torch.load(ECAPA_PATH, weights_only=False)
    ds = M3Set(args.feat, ecapa, args.seg)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=6,
                    drop_last=True, persistent_workers=True)
    print(f"M3 foundation VC | {len(ds)} utts | ecapa {len(ecapa)} | steps={args.steps}")

    g = NsfHifiGan(cond_dim=768 + 2, timbre_dim=TIMBRE_DIM).to(DEV)
    if args.init_g and Path(args.init_g).exists():
        sd = torch.load(args.init_g, map_location=DEV, weights_only=False)["g"]
        own = g.state_dict()
        n = 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                n += 1
        g.load_state_dict(own)
        print(f"  warm-started G: {n}/{len(own)} tensors")
    _ck = (torch.load(args.init_g, map_location=DEV, weights_only=False)
           if args.init_g and Path(args.init_g).exists() else {})
    t = TimbreEncoder(dim=EMB_DIM).to(DEV)
    scrub = ContentScrub().to(DEV)
    if "t" in _ck:
        t.load_state_dict(_ck["t"]); scrub.load_state_dict(_ck["scrub"])
        print("  resumed t/scrub from checkpoint")
    cemb_pred = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.LeakyReLU(0.1),
                              nn.Linear(256, EMB_DIM)).to(DEV)
    f0p = F0Predictor().to(DEV)
    d = Discriminator2().to(DEV)
    gp = (list(g.parameters()) + list(t.parameters()) + list(scrub.parameters())
          + list(cemb_pred.parameters()) + list(f0p.parameters()))
    og = torch.optim.AdamW(gp, args.lr, betas=(0.8, 0.99))
    od = torch.optim.AdamW(d.parameters(), args.lr, betas=(0.8, 0.99))
    print("  ECAPA identity-distill (cosine) + learned-F0 + content-scrub(GRL)")

    step = 0
    while step < args.steps:
        for cond, f0, logf0, voiced, y, rw, emb, evalid in dl:
            cond, f0, logf0, voiced, y, rw, emb, evalid = (
                cond.to(DEV), f0.to(DEV), logf0.to(DEV), voiced.to(DEV), y.to(DEV),
                rw.to(DEV), emb.to(DEV), evalid.to(DEV))
            s = t(mel_of(rw))
            ew = evalid.unsqueeze(-1)
            id_loss = (1.0 - F.cosine_similarity(s, emb, dim=-1)) * evalid
            id_loss = id_loss.sum() / (evalid.sum() + 1e-6)
            content_s = scrub(cond[:, :768])
            alpha = 0.5 * min(1.0, step / 10000.0)
            cadv = cemb_pred(grad_reverse(content_s.mean(-1), alpha))
            c_adv = (F.mse_loss(cadv * ew, emb * ew, reduction="sum")
                     / (ew.sum() * EMB_DIM + 1e-6))
            f0_hat = f0p(content_s, s)
            f0l = (F.smooth_l1_loss(f0_hat[:, 0] * voiced, logf0 * voiced)
                   + F.binary_cross_entropy_with_logits(f0_hat[:, 1], voiced))
            cond2 = torch.cat([content_s, cond[:, 768:]], dim=1)
            y_hat = g(cond2, f0, s)[..., : y.shape[-1]]
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
            g_loss = (45.0 * mel_l + 2.0 * mrs + 3.0 * id_loss + 0.1 * c_adv + 5.0 * f0l)
            if use_gan:
                dg, fg = d(y_hat)
                dr, fr = d(y)
                g_adv = sum(((gg - 1) ** 2).mean() for gg in dg)
                g_loss = g_loss + g_adv + 2.0 * fm_loss(fr, fg)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gp, 10.0)
            og.step()

            if step % 50 == 0:
                msg = (f"step {step} mel {mel_l.item():.3f} mrs {mrs.item():.3f} "
                       f"id {id_loss.item():.3f} cadv {c_adv.item():.3f} f0 {f0l.item():.3f}")
                if use_gan:
                    msg += f" gadv {g_adv.item():.2f}"
                print(msg, flush=True)
            if step % 2000 == 0:
                torch.save({"g": g.state_dict(), "t": t.state_dict(), "scrub": scrub.state_dict(),
                            "f0p": f0p.state_dict(), "step": step}, out / "last.pt")
            step += 1
            if step >= args.steps:
                break
    torch.save({"g": g.state_dict(), "t": t.state_dict(), "scrub": scrub.state_dict(),
                "f0p": f0p.state_dict(), "step": step}, out / "last.pt")
    print("done")


if __name__ == "__main__":
    main()
