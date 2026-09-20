"""MelGen — Z1 self-reconstruction mel generator G (zeroshot_vc.md §3.4).

Own mel-output generator for the zero-shot VC front stage. Emits a target mel
that is ABI-identical to freebig's input (mel_of / bigvgan.meldataset), so it can
be fed to the frozen freebig vocoder with no conversion.

Design contract (hard-won lessons, do not violate — see §3.4 / m2-clone-adain-grl):
  - backbone = kansei_vocoder.ConvNeXtBlock1d (isotropic, groups=1 = XPU-safe),
    NO ConvTranspose (time-upsample = jirijiri), NO F0-driven harmonic source
    (that is the dead A/S path — this net only paints a mel).
  - single AdaIN per block (F.instance_norm(x)*(1+gamma)+beta). instance_norm
    strips the source channel statistics each layer so G is FORCED to use the
    target code. Additive FiLM fails clone; two-stage AdaIN cancels (IN(IN)=IN).
  - timbre (z_spk) and articulation (s_art) are fused into the SAME AdaIN by
    ADDING their gamma/beta (films + films_art), both zero-init so warm-start is
    clean (starts at pure instance_norm, art contributes nothing at step 0).
  - head bias = data-mean log-mel (zero-init forbidden, §3.4); small weight std.

Ported faithfully from the proven AdaIN in nsf_hn.py:140-151.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from kansei_vocoder import ConvNeXtBlock1d


class MelGen(nn.Module):
    def __init__(self, cond_dim: int = 770, dim: int = 384, n_layers: int = 6,
                 n_mels: int = 128, timbre_dim: int = 192, art_dim: int = 48,
                 causal: bool = False, mel_mean=None, f0_fourier: int = 0) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.dim = dim
        self.n_layers = n_layers
        self.n_mels = n_mels
        self.timbre_dim = timbre_dim
        self.art_dim = art_dim
        self.causal = causal
        # f0 conditioning strengthening: expand the single logf0 channel (index
        # cond_dim-2) into Fourier features so G can faithfully track / shift pitch.
        self.f0_fourier = f0_fourier
        if f0_fourier > 0:
            self.register_buffer("f0_freqs", 2.0 ** torch.linspace(0.0, 8.0, f0_fourier))
        in_ch = cond_dim + (2 * f0_fourier if f0_fourier > 0 else 0)
        self.in_proj = nn.Conv1d(in_ch, dim, 1)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock1d(dim, causal=causal) for _ in range(n_layers)])
        # single AdaIN per block: timbre gamma/beta + articulation gamma/beta,
        # ADDED (not stacked). zero-init -> clean warm-start (§3.4).
        self.films = nn.ModuleList()
        self.films_art = nn.ModuleList()
        for _ in range(n_layers):
            film = nn.Linear(timbre_dim, 2 * dim)
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
            self.films.append(film)
            fa = nn.Linear(art_dim, 2 * dim)
            nn.init.zeros_(fa.weight)
            nn.init.zeros_(fa.bias)
            self.films_art.append(fa)
        self.out_norm = nn.LayerNorm(dim)
        self.head = nn.Conv1d(dim, n_mels, 1)
        nn.init.normal_(self.head.weight, 0.0, 1e-2)
        if mel_mean is not None:
            self.head.bias.data.copy_(torch.as_tensor(mel_mean, dtype=torch.float32))
        else:
            nn.init.zeros_(self.head.bias)

    def forward(self, cond: torch.Tensor, s: torch.Tensor = None,
                s_art: torch.Tensor = None) -> torch.Tensor:
        # cond: [B, cond_dim, T] = [content(768) | logf0 | energy]
        if self.f0_fourier > 0:
            lf = cond[:, self.cond_dim - 2:self.cond_dim - 1, :] * 7.0   # undo build_cond /7 -> raw log-Hz
            ph = self.f0_freqs.view(1, -1, 1) * lf                       # [B, K, T]
            cond = torch.cat([cond, torch.sin(ph), torch.cos(ph)], dim=1)
        x = self.in_proj(cond)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            has_t = self.timbre_dim and s is not None
            has_a = self.art_dim and s_art is not None
            if has_t or has_a:
                gamma = x.new_zeros(x.shape[0], x.shape[1])
                beta = x.new_zeros(x.shape[0], x.shape[1])
                if has_t:
                    gt, bt = self.films[i](s).chunk(2, dim=-1)
                    gamma, beta = gamma + gt, beta + bt
                if has_a:
                    ga, ba = self.films_art[i](s_art).chunk(2, dim=-1)
                    gamma, beta = gamma + ga, beta + ba
                x = F.instance_norm(x) * (1.0 + gamma).unsqueeze(-1) + beta.unsqueeze(-1)
        x = self.out_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.head(x)  # [B, n_mels, T]


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = MelGen(cond_dim=770, dim=384, n_layers=6).to(dev)
    print("params", round(sum(p.numel() for p in g.parameters()) / 1e6, 2), "M")
    cond = torch.randn(2, 770, 40, device=dev)
    s = torch.randn(2, 192, device=dev)
    s_art = torch.randn(2, 48, device=dev)
    m = g(cond, s, s_art)
    print("cond", tuple(cond.shape), "-> mel", tuple(m.shape))
    m.sum().backward()
    print("backward OK")
