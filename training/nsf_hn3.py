"""Sample-rate Neural Source-Filter Harmonic+Noise vocoder v3 (NSF-HN3).

Evolution of nsf_hn2.py. Keeps the tremolo-solving principle (sample-rate signal
path, NO signal ConvTranspose, content/timbre as smooth conditioning only) and
adds the HiFiGAN/NSF fidelity mechanisms the design review found missing:

  (1) dilation [1,2,4,...,512] + weight_norm + skip aggregation head
      -> RF ~46ms (backward-only, causal, 0 latency), stable optimization,
         multi-scale output. THE free fidelity win.
  (3) frame-rate MRF control-net -> 1x1 additive injection into sample-rate
      blocks. HiFiGAN-class capacity at ~unchanged real-time cost (heavy work
      runs at 86fps, injected additively into the cheap sample-rate stack).
  (4) n_harm=340, learnable broadband aperiodicity WITHOUT unvoiced gate
      (breathiness is controllable in voiced too), noise pre-emphasis.
  adaLN-zero fix: channel LayerNorm before FiLM modulation.

Deferred (add only if the ear gate demands): band-wise noise branch (#2 full),
anti-aliased activation (#1 contingency for voiced inter-harmonic aliasing),
float64 phase (M3/Candle port). GELU kept; watch G-voiced-interharmonic gate.

Causal, groups=1 standard conv (XPU-safe). Interface matches NsfHn2/NsfHifiGan:
forward(cond, f0_frame, s) -> (B, 1, T*hop).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from aa import AAActivation

SR = 44100
HOP = 512


def _up(x: torch.Tensor, hop: int) -> torch.Tensor:
    return F.interpolate(x, scale_factor=hop, mode="linear", align_corners=False)


class HarmonicSource(nn.Module):
    def __init__(self, sr: int = SR, hop: int = HOP, n_harm: int = 340, amp: float = 0.1,
                 tilt: float = 0.0) -> None:
        super().__init__()
        self.sr = sr
        self.hop = hop
        self.n = n_harm
        self.amp = amp * (8.0 / n_harm) ** 0.5
        # per-harmonic tilt: w_k = k^-tilt, energy-normalized. tilt=0 -> flat (current).
        # tilt=1 -> 1/k (-6 dB/oct), physics-aligned glottal source. Distribution only;
        # total energy preserved. Non-persistent buffer so old checkpoints still load.
        k = torch.arange(1, n_harm + 1, dtype=torch.float32)
        w = k ** (-tilt)
        w = w / (w.pow(2).mean().sqrt() + 1e-9)
        self.register_buffer("hw", w.view(1, -1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, f0_frame: torch.Tensor) -> torch.Tensor:
        # f0_frame: (B, T) -> h: (B, T*hop)
        # NOTE (M3/Candle port): float32 cumsum * k(=n_harm) accumulates rad-order
        # phase error over minutes of continuous inference. Harmless at <1s training
        # segments. At port, use frac(k*frac(p))==frac(k*p) to keep precision.
        f0 = _up(f0_frame.unsqueeze(1), self.hop).squeeze(1)
        base = torch.cumsum(2.0 * np.pi * f0 / self.sr, dim=-1)
        ks = torch.arange(1, self.n + 1, device=f0.device, dtype=f0.dtype).view(1, -1, 1)
        phase = ks * base.unsqueeze(1)
        sines = torch.sin(phase)
        mask = (f0.unsqueeze(1) * ks < self.sr / 2).to(f0.dtype)
        uv = (f0 > 1.0).to(f0.dtype).unsqueeze(1)
        h = (sines * mask * uv * self.hw.to(f0.dtype)).sum(1) * self.amp
        return h


class CausalConv1d(nn.Module):
    def __init__(self, ci: int, co: int, k: int, dilation: int = 1, wn: bool = True) -> None:
        super().__init__()
        self.pad = (k - 1) * dilation
        c = nn.Conv1d(ci, co, k, dilation=dilation)
        self.conv = weight_norm(c) if wn else c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class ChanLayerNorm(nn.Module):
    """LayerNorm over channels per time-step (streaming/causal-safe)."""
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, ch, 1))
        self.b = nn.Parameter(torch.zeros(1, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + 1e-5) * self.g + self.b


class MRFResBlock(nn.Module):
    """HiFiGAN-style multi-receptive-field residual block, CAUSAL (left-pad only).
    Runs at frame rate; its RF is entirely backward, so it adds ZERO lookahead
    latency (past frames stream naturally). Symmetric padding here would leak
    ~0.7s of future -> breaks E2E<50ms and would make the mel-ceiling result
    depend on non-causal context."""
    def __init__(self, ch: int, k: int, dils: tuple = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, dilation=d)) for d in dils])
        self.convs2 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, dilation=1)) for _ in dils])
        self.pad1 = [(k - 1) * d for d in dils]
        self.pad2 = k - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, p1 in zip(self.convs1, self.convs2, self.pad1):
            h = c1(F.pad(F.leaky_relu(x, 0.1), (p1, 0)))
            h = c2(F.pad(F.leaky_relu(h, 0.1), (self.pad2, 0)))
            x = x + h
        return x


class ControlNet(nn.Module):
    """Thick frame-rate net -> control feature, injected additively (1x1) into
    each sample-rate block. This is where HiFiGAN-class capacity lives."""
    def __init__(self, in_dim: int, ch: int = 128, ks: tuple = (3, 7, 11)) -> None:
        super().__init__()
        self.pre = weight_norm(nn.Conv1d(in_dim, ch, 7))     # causal: left-pad in forward
        self.res = nn.ModuleList([MRFResBlock(ch, k) for k in ks])
        self.ks = ks

    def forward(self, cond_cat: torch.Tensor) -> torch.Tensor:
        x = self.pre(F.pad(cond_cat, (6, 0)))
        acc = None
        for r in self.res:
            acc = r(x) if acc is None else acc + r(x)
        return acc / len(self.res)


class NsfBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, aa_act: bool = False, use_norm: bool = True) -> None:
        super().__init__()
        self.norm = ChanLayerNorm(ch) if use_norm else nn.Identity()
        self.conv = CausalConv1d(ch, ch, 3, dilation=dilation)
        self.act = AAActivation(nn.GELU()) if aa_act else nn.GELU()

    def forward(self, x: torch.Tensor, ctrl: torch.Tensor,
                scale: torch.Tensor, shift: torch.Tensor) -> tuple:
        h = self.conv(self.norm(x))
        h = h * (1.0 + scale) + shift + ctrl   # additive control-net injection
        a = self.act(h)
        return x + a, a                        # (residual, skip)


class NsfHn3(nn.Module):
    def __init__(self, cond_dim: int = 770, timbre_dim: int = 192, ch: int = 64,
                 n_blocks: int = 10, ctrl_ch: int = 128, sr: int = SR, hop: int = HOP,
                 n_harm: int = 340, base_noise: float = 0.05, uv_noise: float = 0.5,
                 tilt: float = 0.0, aa_act: bool = False, use_norm: bool = True) -> None:
        super().__init__()
        self.hop = hop
        self.base_noise = base_noise
        self.uv_noise = uv_noise
        self.aa_act = aa_act
        self.out_act = AAActivation(nn.GELU()) if aa_act else nn.GELU()
        self.harm = HarmonicSource(sr, hop, n_harm, tilt=tilt)
        self.ap_head = nn.Sequential(
            weight_norm(nn.Conv1d(cond_dim, 64, 1)), nn.GELU(),
            weight_norm(nn.Conv1d(64, 1, 1)))
        self.cc = 48
        self.cproj = weight_norm(nn.Conv1d(cond_dim, self.cc, 1))
        self.inproj = CausalConv1d(3 + self.cc, ch, 7)
        self.control = ControlNet(cond_dim + timbre_dim, ctrl_ch)
        self.ctrl_proj = nn.ModuleList([nn.Conv1d(ctrl_ch, ch, 1) for _ in range(n_blocks)])
        dils = [2 ** i for i in range(n_blocks)]        # 1,2,4,...,512
        self.blocks = nn.ModuleList([NsfBlock(ch, dils[i], aa_act, use_norm) for i in range(n_blocks)])
        self.films = nn.ModuleList([nn.Conv1d(cond_dim + timbre_dim, ch * 2, 1)
                                    for _ in range(n_blocks)])
        for f in self.films:
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)
        for p in self.ctrl_proj:                        # zero-init additive injection
            nn.init.zeros_(p.weight)
            nn.init.zeros_(p.bias)
        self.skip_head = CausalConv1d(ch, ch, 3)
        self.outproj = CausalConv1d(ch, 1, 7)

    def forward(self, cond: torch.Tensor, f0_frame: torch.Tensor,
                s: torch.Tensor = None, s_art: torch.Tensor = None) -> torch.Tensor:
        B, _, T = cond.shape
        if f0_frame.dim() == 3:
            f0_frame = f0_frame.squeeze(1)
        h = self.harm(f0_frame)                                   # (B, Tn)
        ap = torch.sigmoid(self.ap_head(cond))                    # (B, 1, T)
        ap_s = _up(ap, self.hop).squeeze(1)                       # (B, Tn)
        noise = torch.randn_like(h) * (self.base_noise + self.uv_noise * ap_s)
        noise_pe = noise - F.pad(noise, (1, 0))[:, :-1]           # pre-emphasis
        cp = _up(self.cproj(cond), self.hop)                      # (B, cc, Tn) smooth content
        x = torch.cat([h.unsqueeze(1), noise.unsqueeze(1),
                       noise_pe.unsqueeze(1), cp], dim=1)         # (B, 3+cc, Tn)
        x = self.inproj(x)                                        # (B, ch, Tn)
        if s is not None:
            fc = torch.cat([cond, s.unsqueeze(-1).expand(-1, -1, T)], dim=1)
        else:
            fc = torch.cat([cond, cond.new_zeros(B, self.films[0].in_channels - cond.shape[1], T)], dim=1)
        ctrl = self.control(fc)                                   # (B, ctrl_ch, T) frame rate
        skips = 0.0
        for i, blk in enumerate(self.blocks):
            ss = _up(self.films[i](fc), self.hop)                 # (B, 2ch, Tn)
            scale, shift = ss.chunk(2, dim=1)
            c_add = _up(self.ctrl_proj[i](ctrl), self.hop)        # (B, ch, Tn) smooth
            x, a = blk(x, c_add, scale, shift)
            skips = skips + a
        y = self.skip_head(skips / len(self.blocks))
        return torch.tanh(self.outproj(self.out_act(y)))


if __name__ == "__main__":
    g = NsfHn3()
    cond = torch.randn(2, 770, 40)
    f0 = torch.rand(2, 40) * 200 + 100
    s = torch.randn(2, 192)
    y = g(cond, f0, s)
    print("out", tuple(y.shape), "params", sum(p.numel() for p in g.parameters()))
    yn = g(cond, f0, None)
    print("no-timbre out", tuple(yn.shape))
