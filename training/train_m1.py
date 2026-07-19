from __future__ import annotations

import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import pyworld
import soundfile as sf
from torch.nn.utils import weight_norm

sys.path.insert(0, str(Path(__file__).parent))
from nsf_hn import NsfHifiGan

import bigvgan.meldataset as bm

SR = 44100
HOP = 512
N_FFT = 2048
WIN = 2048
N_MELS = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def mel_of(wav: torch.Tensor) -> torch.Tensor:
    return bm.mel_spectrogram(wav, N_FFT, N_MELS, SR, HOP, WIN, 0, None)


def compute_f0(wav: np.ndarray, n_frames: int) -> np.ndarray:
    w64 = wav.astype(np.float64)
    f0, t = pyworld.harvest(w64, SR, f0_floor=65, f0_ceil=1000,
                            frame_period=HOP / SR * 1000)
    f0 = pyworld.stonemask(w64, f0, t, SR).astype(np.float32)
    if len(f0) >= n_frames:
        return f0[:n_frames]
    return np.pad(f0, (0, n_frames - len(f0)))


class MoeSet(Dataset):
    def __init__(self, feat_root: str, seg_frames: int = 32) -> None:
        self.files = sorted(Path(feat_root).rglob("*.pt"))
        self.seg = seg_frames

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple:
        d = torch.load(self.files[i], weights_only=False)
        if "f0" not in d or "energy" not in d:
            return self.__getitem__((i + 1) % len(self.files))
        content = d["content"].float()
        f0 = d["f0"].float()
        energy = d["energy"].float()
        w, sr = sf.read(d["path"], dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        y = torch.from_numpy(np.ascontiguousarray(w)).float()
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
        return cond, f0, y.unsqueeze(0)


class PeriodDisc(nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, (2, 0)))])
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x: torch.Tensor) -> tuple:
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), "reflect")
        x = x.view(b, c, -1, self.period)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class ScaleDisc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, 1, 7)),
            weight_norm(nn.Conv1d(128, 128, 41, 4, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, 2))])
        self.post = weight_norm(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple:
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mpd = nn.ModuleList([PeriodDisc(p) for p in (2, 3, 5, 7, 11)])
        self.msd = nn.ModuleList([ScaleDisc() for _ in range(3)])
        self.pools = nn.ModuleList([nn.AvgPool1d(4, 2, 2) for _ in range(2)])

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
        return outs, fmaps


def fm_loss(fr: list, fg: list) -> torch.Tensor:
    loss = 0.0
    for dr, dg in zip(fr, fg):
        for r, g in zip(dr, dg):
            loss = loss + F.l1_loss(g, r.detach())
    return loss


STFT_CFG = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]


def stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    w = torch.hann_window(win, device=x.device)
    s = torch.stft(x, n_fft, hop, win, w, return_complex=True)
    return s.abs().clamp(min=1e-7)


def mrstft_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for n_fft, hop, win in STFT_CFG:
        sh = stft_mag(y_hat, n_fft, hop, win)
        st = stft_mag(y, n_fft, hop, win)
        sc = torch.norm(st - sh, p="fro") / (torch.norm(st, p="fro") + 1e-7)
        mag = F.l1_loss(torch.log(sh), torch.log(st))
        loss = loss + sc + mag
    return loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--out", default="checkpoints/m1_nsfhn")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gan-after", type=int, default=1000)
    ap.add_argument("--render-every", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    render_dir = Path("../results/diag_m1")
    render_dir.mkdir(parents=True, exist_ok=True)

    ds = MoeSet(args.feat, args.seg)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=6,
                    drop_last=True, persistent_workers=True)
    print(f"M1 NSF-HN | {len(ds)} utts | steps={args.steps} gan_after={args.gan_after}")

    g = NsfHifiGan(cond_dim=768 + 2).to(DEV)
    d = Discriminator().to(DEV)
    og = torch.optim.AdamW(g.parameters(), args.lr, betas=(0.8, 0.99))
    od = torch.optim.AdamW(d.parameters(), args.lr, betas=(0.8, 0.99))

    step = 0
    while step < args.steps:
        for cond, f0, y in dl:
            cond, f0, y = cond.to(DEV), f0.to(DEV), y.to(DEV)
            y_hat = g(cond, f0)
            y_hat = y_hat[..., : y.shape[-1]]
            mel_hat = mel_of(y_hat.squeeze(1))
            mel_t = mel_of(y.squeeze(1))
            mel_l = F.l1_loss(mel_hat, mel_t)
            use_gan = step >= args.gan_after

            if use_gan:
                od.zero_grad()
                dr, _ = d(y)
                dg, _ = d(y_hat.detach())
                d_loss = sum(((r - 1) ** 2).mean() + (gg ** 2).mean()
                             for r, gg in zip(dr, dg))
                d_loss.backward()
                od.step()

            og.zero_grad()
            mrs = mrstft_loss(y_hat.squeeze(1), y.squeeze(1))
            g_loss = 45.0 * mel_l + 2.0 * mrs
            if use_gan:
                dg, fg = d(y_hat)
                dr, fr = d(y)
                g_adv = sum(((gg - 1) ** 2).mean() for gg in dg)
                g_fm = 2.0 * fm_loss(fr, fg)
                g_loss = g_loss + g_adv + g_fm
            g_loss.backward()
            og.step()

            if step % 50 == 0:
                msg = f"step {step} mel_l1 {mel_l.item():.3f}"
                if use_gan:
                    msg += f" g_adv {g_adv.item():.3f} d {d_loss.item():.3f}"
                print(msg, flush=True)

            if step % args.render_every == 0:
                g.eval()
                with torch.no_grad():
                    yv = g(cond[:1], f0[:1])[0, 0].cpu().numpy()
                sf.write(render_dir / f"diag0_step{step}.wav",
                         np.clip(yv, -1, 1), SR, subtype="PCM_16")
                sf.write(render_dir / "diag0_target.wav",
                         np.clip(y[0, 0].cpu().numpy(), -1, 1), SR, subtype="PCM_16")
                torch.save({"g": g.state_dict(), "step": step}, out / "last.pt")
                g.train()

            step += 1
            if step >= args.steps:
                break
    print("done")


if __name__ == "__main__":
    main()
