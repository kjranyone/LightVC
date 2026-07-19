from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SR = 44100
HOP = 512
N_FFT = 1024
WIN = 1024
NYQ = SR / 2
KMAX = 180


def _window(n: int, device, dtype) -> torch.Tensor:
    return torch.hann_window(n, periodic=True, device=device, dtype=dtype)


def stft_ri(x: torch.Tensor, n_fft: int = N_FFT, hop: int = HOP,
            win: int = WIN, frames: int | None = None) -> torch.Tensor:
    w = _window(win, x.device, x.dtype)
    z = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                   window=w, center=True, return_complex=True)
    ri = torch.stack([z.real, z.imag], dim=1)
    if frames is not None:
        ri = ri[..., :frames]
    return ri


def istft_ri(ri: torch.Tensor, length: int, n_fft: int = N_FFT,
             hop: int = HOP, win: int = WIN) -> torch.Tensor:
    w = _window(win, ri.device, ri.dtype)
    z = torch.complex(ri[:, 0], ri[:, 1])
    return torch.istft(z, n_fft=n_fft, hop_length=hop, win_length=win,
                       window=w, center=True, length=length)


class Excitation(nn.Module):
    def __init__(self, sr: int = SR, hop: int = HOP, kmax: int = KMAX,
                 voiced_threshold: float = 10.0, noise_std: float = 0.01) -> None:
        super().__init__()
        self.sr = sr
        self.hop = hop
        self.kmax = kmax
        self.vt = voiced_threshold
        self.noise_std = noise_std
        self.register_buffer("k", torch.arange(1, kmax + 1).float().view(1, -1, 1))

    @torch.no_grad()
    def forward(self, f0_frame: torch.Tensor, frames: int) -> torch.Tensor:
        n = frames * self.hop
        f0 = F.interpolate(f0_frame.unsqueeze(1), size=n, mode="linear",
                           align_corners=False)
        rad = f0 / self.sr
        cum = torch.cumsum(rad, dim=-1)
        phase = 2.0 * math.pi * cum * self.k
        mask = (self.k * f0 < NYQ).float()
        uv = (f0 > self.vt).float()
        kn = mask.sum(1, keepdim=True).clamp(min=1.0)
        sines = torch.sin(phase) * mask
        harm = sines.sum(1, keepdim=True) * torch.sqrt(0.02 / kn) * uv
        noise = torch.randn_like(harm) * self.noise_std
        e = (harm + noise).squeeze(1)
        return stft_ri(e, frames=frames)


class Block2D(nn.Module):
    def __init__(self, ch: int, timbre_dim: int, kernel: int = 7,
                 depthwise: bool = True) -> None:
        super().__init__()
        groups = ch if depthwise else 1
        self.dw = nn.Conv2d(ch, ch, kernel, padding=kernel // 2, groups=groups)
        self.norm = nn.LayerNorm(ch)
        self.pw1 = nn.Linear(ch, 4 * ch)
        self.pw2 = nn.Linear(4 * ch, ch)
        self.film = nn.Linear(timbre_dim, 2 * ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw2(F.gelu(self.pw1(x)))
        x = x.permute(0, 3, 1, 2)
        g, b = self.film(s).chunk(2, dim=-1)
        x = x * (1.0 + g).unsqueeze(-1).unsqueeze(-1) + b.unsqueeze(-1).unsqueeze(-1)
        return r + x


class AFHN(nn.Module):
    def __init__(self, cond_dim: int = 770, timbre_dim: int = 192,
                 ch: int = 64, cond_proj: int = 32, n_blocks: int = 8,
                 kernel: int = 7, depthwise: bool = True,
                 n_fft: int = N_FFT, hop: int = HOP, use_ckpt: bool = True) -> None:
        super().__init__()
        self.hop = hop
        self.n_fft = n_fft
        self.nbins = n_fft // 2 + 1
        self.use_ckpt = use_ckpt
        self.exc = Excitation(hop=hop)
        self.cond_proj = nn.Conv1d(cond_dim, cond_proj, 1)
        self.env_proj = nn.Conv1d(cond_dim, self.nbins, 1)
        self.conv_pre = nn.Conv2d(2 + 1 + cond_proj, ch, 3, padding=1)
        self.blocks = nn.ModuleList(
            [Block2D(ch, timbre_dim, kernel, depthwise) for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(ch)
        self.head = nn.Linear(ch, 2)
        nn.init.normal_(self.head.weight, 0.0, 0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, cond: torch.Tensor, f0_frame: torch.Tensor,
                s: torch.Tensor) -> torch.Tensor:
        t = cond.shape[-1]
        exc = self.exc(f0_frame, t)
        env = self.env_proj(cond).unsqueeze(1)
        c = self.cond_proj(cond)
        c = c.unsqueeze(2).expand(-1, -1, self.nbins, -1)
        x = torch.cat([exc, env, c], dim=1)
        x = self.conv_pre(x)
        for blk in self.blocks:
            if self.use_ckpt and self.training:
                x = checkpoint(blk, x, s, use_reentrant=False)
            else:
                x = blk(x, s)
        x = x.permute(0, 2, 3, 1)
        x = self.head(self.norm_out(x))
        ri = x.permute(0, 3, 1, 2)
        y = istft_ri(ri, length=t * self.hop, n_fft=self.n_fft, hop=self.hop)
        return y.unsqueeze(1)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = AFHN().to(dev)
    tf = 64
    cond = torch.randn(2, 770, tf, device=dev)
    f0 = (torch.rand(2, tf, device=dev) * 200 + 180)
    s = torch.randn(2, 192, device=dev)
    y = g(cond, f0, s)
    print("params", round(sum(p.numel() for p in g.parameters()) / 1e6, 3), "M")
    print("cond", tuple(cond.shape), "-> wav", tuple(y.shape),
          "expected", tf * HOP)
    print("finite", torch.isfinite(y).all().item(), "range",
          round(y.min().item(), 3), round(y.max().item(), 3))
