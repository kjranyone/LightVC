"""Sample-rate Neural Source-Filter Harmonic+Noise vocoder (NSF-HN2).

Design principle (the fix for the 86Hz breath tremolo):
  The signal path is at SAMPLE RATE throughout. The only signal inputs are a
  harmonic source h[n] (voiced) and a noise source e[n] (breath/aperiodic),
  both generated at sample rate. content/timbre enter ONLY as smooth FiLM
  conditioning (frame-rate params linearly upsampled to sample rate). There is
  NO ConvTranspose upsampling of the signal, so no frame-boundary artifacts.
  Breath = filtered continuous noise -> structurally smooth, no tremolo.

Causal, groups=1 standard conv (XPU-safe), small, streamable.
Interface matches NsfHifiGan: forward(cond, f0_frame, s) -> (B, 1, T*hop).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 44100
HOP = 512


def _up(x: torch.Tensor, hop: int) -> torch.Tensor:
    return F.interpolate(x, scale_factor=hop, mode="linear", align_corners=False)


class HarmonicSource(nn.Module):
    def __init__(self, sr: int = SR, hop: int = HOP, n_harm: int = 160, amp: float = 0.1) -> None:
        super().__init__()
        self.sr = sr
        self.hop = hop
        self.n = n_harm
        self.amp = amp * (8.0 / n_harm) ** 0.5

    @torch.no_grad()
    def forward(self, f0_frame: torch.Tensor) -> torch.Tensor:
        # f0_frame: (B, T) -> h: (B, T*hop)
        # NOTE (M3/Candle port): float32 cumsum * k(=n_harm) accumulates rad-order
        # phase error over minutes of continuous inference. Harmless at 0.74s training
        # segments. At port, use frac(k*frac(p))==frac(k*p) to keep precision.
        f0 = _up(f0_frame.unsqueeze(1), self.hop).squeeze(1)
        base = torch.cumsum(2.0 * np.pi * f0 / self.sr, dim=-1)
        ks = torch.arange(1, self.n + 1, device=f0.device, dtype=f0.dtype).view(1, -1, 1)
        phase = ks * base.unsqueeze(1)
        sines = torch.sin(phase)
        mask = (f0.unsqueeze(1) * ks < self.sr / 2).to(f0.dtype)
        uv = (f0 > 1.0).to(f0.dtype).unsqueeze(1)
        h = (sines * mask * uv).sum(1) * self.amp
        return h


class CausalConv1d(nn.Module):
    def __init__(self, ci: int, co: int, k: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(ci, co, k, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class NsfBlock(nn.Module):
    def __init__(self, ch: int, dilation: int) -> None:
        super().__init__()
        self.conv = CausalConv1d(ch, ch, 3, dilation=dilation)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h * (1.0 + scale) + shift
        return x + self.act(h)


class NsfHn2(nn.Module):
    def __init__(self, cond_dim: int = 770, timbre_dim: int = 192, ch: int = 64,
                 n_blocks: int = 10, sr: int = SR, hop: int = HOP, n_harm: int = 160,
                 base_noise: float = 0.05, uv_noise: float = 0.5) -> None:
        super().__init__()
        self.hop = hop
        self.base_noise = base_noise
        self.uv_noise = uv_noise
        self.harm = HarmonicSource(sr, hop, n_harm)
        self.ap_head = nn.Sequential(
            nn.Conv1d(cond_dim, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1))
        self.cc = 48
        self.cproj = nn.Conv1d(cond_dim, self.cc, 1)
        self.inproj = CausalConv1d(2 + self.cc, ch, 7)
        dils = [1, 3, 9, 27, 81]
        self.blocks = nn.ModuleList([NsfBlock(ch, dils[i % len(dils)]) for i in range(n_blocks)])
        self.films = nn.ModuleList([nn.Conv1d(cond_dim + timbre_dim, ch * 2, 1) for _ in range(n_blocks)])
        for f in self.films:
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)
        self.outproj = CausalConv1d(ch, 1, 7)

    def forward(self, cond: torch.Tensor, f0_frame: torch.Tensor,
                s: torch.Tensor = None, s_art: torch.Tensor = None) -> torch.Tensor:
        B, _, T = cond.shape
        if f0_frame.dim() == 3:
            f0_frame = f0_frame.squeeze(1)
        h = self.harm(f0_frame)                                   # (B, Tn)
        ap = torch.sigmoid(self.ap_head(cond))                    # (B, 1, T) learnable aperiodicity
        ap_s = _up(ap, self.hop).squeeze(1)                       # (B, Tn)
        # No unvoiced gate: ap is active in voiced too -> learnable breathiness knob
        # (breathy-voiced is the ASMR product). Model lowers ap where it wants clean voice.
        noise = torch.randn_like(h) * (self.base_noise + self.uv_noise * ap_s)
        cp = _up(self.cproj(cond), self.hop)                      # (B, cc, Tn) smooth content
        x = torch.cat([h.unsqueeze(1), noise.unsqueeze(1), cp], dim=1)  # (B, 2+cc, Tn)
        x = self.inproj(x)                                        # (B, ch, Tn)
        if s is not None:
            fc = torch.cat([cond, s.unsqueeze(-1).expand(-1, -1, T)], dim=1)
        else:
            fc = cond
        for i, blk in enumerate(self.blocks):
            ss = _up(self.films[i](fc), self.hop)                 # (B, 2ch, Tn)
            scale, shift = ss.chunk(2, dim=1)
            x = blk(x, scale, shift)
        return torch.tanh(self.outproj(x))                        # (B, 1, Tn)


if __name__ == "__main__":
    g = NsfHn2()
    cond = torch.randn(2, 770, 40)
    f0 = torch.rand(2, 40) * 200 + 100
    s = torch.randn(2, 192)
    y = g(cond, f0, s)
    print("out", tuple(y.shape), "params", sum(p.numel() for p in g.parameters()))
