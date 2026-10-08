"""候補B(本命・NAM A2思想): 全層48kHz・上げ層なしの因果decoder。

構成:
  control net(100fps・因果・広い): [z(32), log(f0/200), vuv] → 1x1 で層ごとの条件 [L*C]
  → 因果線形補間(フレーム t の 480 サンプルは c[t-1]→c[t] を直線で結ぶ・先読み0)
  励起(48kHz): 明示f0の倍音和(k·f0<20kHz・有声のみ) + ガウス雑音 → 1x1 で C ch へ
  A2スタック(48kHz): 23層・kernel 6(中盤2層15)・dilation (1,3,7,17,41,101,239)×2 + (1,13,1,3,7,17,41,101,239)
     h = dilated causal conv(x) + cond_l ; a = LeakyReLU(h, 0.01) ; skip += a ; x = x + layer1x1(a)
  head: 因果conv(k16) → tanh
ゲート・FiLM・上げ層なし。grouped conv なし。f0 は decoder 入力(明示f0励起=ピッチ権威の構造的経路)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_codec import ConvStream, causal_conv, LATENT_DIM, HOP_LENGTH, SAMPLE_RATE

A2_DILS = (1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239,
           1, 13, 1, 3, 7, 17, 41, 101, 239)
A2_KS = tuple(15 if i in (9, 10) else 6 for i in range(23))
F_MAX_HARM = 20000.0


class ControlNet(nn.Module):
    def __init__(self, cin: int, width: int, out: int, layers: int = 6) -> None:
        super().__init__()
        self.inp = nn.Conv1d(cin, width, 3)
        self.dils = tuple((1, 2, 4)[i % 3] for i in range(layers))
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=d) for d in self.dils])
        self.mix = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in self.dils])
        self.out = nn.Conv1d(width, out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = causal_conv(x, self.inp)
        for c, m in zip(self.convs, self.mix):
            h = h + m(F.leaky_relu(causal_conv(h, c), 0.01))
        return self.out(F.leaky_relu(h, 0.01))


def interp_causal(c: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
    """c [B,D,F] → [B,D,F*H]。フレームtは prev(=c[t-1]) から c[t] へ直線。先頭の prev は c[0]。"""
    if prev is None:
        prev = c[..., :1]
    p = torch.cat((prev, c[..., :-1]), dim=-1)
    w = (torch.arange(1, HOP_LENGTH + 1, device=c.device, dtype=c.dtype) / HOP_LENGTH)
    y = p[..., :, None] + (c - p)[..., :, None] * w
    return y.reshape(c.shape[0], c.shape[1], -1)


def f0_per_sample(f0: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
    """f0 [B,F](Hz・0=無声) → [B,F*H]。有声同士は直線補間、境界は現フレーム値を保持。"""
    if prev is None:
        prev = f0[:, :1]
    p = torch.cat((prev, f0[:, :-1]), dim=-1)
    both = (p > 0) & (f0 > 0)
    p = torch.where(both, p, f0)
    w = torch.arange(1, HOP_LENGTH + 1, device=f0.device, dtype=f0.dtype) / HOP_LENGTH
    return (p[..., None] + (f0 - p)[..., None] * w).reshape(f0.shape[0], -1)


def harmonic_excitation(fs: torch.Tensor, phase0: torch.Tensor, kmax: int = 96):
    """fs [B,N] Hz → (e [B,1,N], phase_end [B])。位相は float64 で積算(長時間ドリフト対策)。"""
    ph = phase0[:, None] + torch.cumsum(fs.double() / SAMPLE_RATE, dim=-1)
    ph_end = torch.remainder(ph[:, -1], 1.0)
    ph = torch.remainder(ph, 1.0).float() * (2 * math.pi)
    k = torch.arange(1, kmax + 1, device=fs.device, dtype=torch.float32)
    mask = ((k[None, :, None] * fs[:, None, :]) < F_MAX_HARM) & (fs[:, None, :] > 0)
    nh = mask.sum(1, keepdim=True).clamp(min=1).float()
    e = (torch.sin(k[None, :, None] * ph[:, None, :]) * mask).sum(1, keepdim=True) / nh.sqrt()
    return e, ph_end


class DecoderA2(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, channels: int = 16,
                 ctrl_width: int = 256, noise_std: float = 0.3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.C = channels
        self.L = len(A2_DILS)
        self.hop_length = HOP_LENGTH
        self.noise_std = noise_std
        self.ctrl = ControlNet(latent_dim + 2, ctrl_width, self.L * channels)
        self.inp = nn.Conv1d(2, channels, 1)
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, k, dilation=d)
                                    for k, d in zip(A2_KS, A2_DILS)])
        self.l1x1 = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in A2_DILS])
        self.head = nn.Conv1d(channels, 1, 16)

    def cond_frames(self, z: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        vuv = (f0 > 0).float()
        lf0 = torch.log(f0.clamp(min=50.0) / 200.0) * vuv
        return self.ctrl(torch.cat([z, lf0[:, None], vuv[:, None]], 1))

    def stack(self, e: torch.Tensor, cond: torch.Tensor, convs_apply) -> torch.Tensor:
        x = self.inp(e)
        skip = torch.zeros_like(x)
        C = self.C
        for i in range(self.L):
            h = convs_apply(i, x) + cond[:, i * C:(i + 1) * C]
            a = F.leaky_relu(h, 0.01)
            skip = skip + a
            x = x + self.l1x1[i](a)
        return skip

    def forward(self, z: torch.Tensor, f0: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        B, _, Fr = z.shape
        cond = interp_causal(self.cond_frames(z, f0))
        with torch.no_grad():
            fs = f0_per_sample(f0)
            eh, _ = harmonic_excitation(fs, torch.zeros(B, device=z.device, dtype=torch.float64))
            if noise is None:
                noise = torch.randn(B, 1, Fr * HOP_LENGTH, device=z.device)
        e = torch.cat([eh, self.noise_std * noise], 1)
        skip = self.stack(e, cond, lambda i, x: causal_conv(x, self.convs[i]))
        return torch.tanh(causal_conv(skip, self.head))

    def stream(self) -> "DecoderA2Stream":
        return DecoderA2Stream(self)


class DecoderA2Stream:
    def __init__(self, dec: DecoderA2) -> None:
        self.dec = dec
        c = dec.ctrl
        self.c_inp = ConvStream(c.inp)
        self.c_convs = [ConvStream(m) for m in c.convs]
        self.convs = [ConvStream(m) for m in dec.convs]
        self.head = ConvStream(dec.head)
        self.prev_cond = None
        self.prev_f0 = None
        self.phase = None

    def reset(self) -> None:
        for s in [self.c_inp, *self.c_convs, *self.convs, self.head]:
            s.reset()
        self.prev_cond = self.prev_f0 = self.phase = None

    def decode_step(self, z: torch.Tensor, f0: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        d = self.dec
        if z.ndim == 2:
            z = z.unsqueeze(-1)
        B = z.shape[0]
        vuv = (f0 > 0).float()
        lf0 = torch.log(f0.clamp(min=50.0) / 200.0) * vuv
        h = self.c_inp(torch.cat([z, lf0[:, None], vuv[:, None]], 1))
        for cs, m in zip(self.c_convs, d.ctrl.mix):
            h = h + m(F.leaky_relu(cs(h), 0.01))
        cf = d.ctrl.out(F.leaky_relu(h, 0.01))
        cond = interp_causal(cf, self.prev_cond)
        self.prev_cond = cf
        fs = f0_per_sample(f0, self.prev_f0)
        self.prev_f0 = f0
        if self.phase is None:
            self.phase = torch.zeros(B, device=z.device, dtype=torch.float64)
        eh, self.phase = harmonic_excitation(fs, self.phase)
        if noise is None:
            noise = torch.randn(B, 1, HOP_LENGTH, device=z.device)
        e = torch.cat([eh, d.noise_std * noise], 1)
        skip = d.stack(e, cond, lambda i, x: self.convs[i](x))
        return torch.tanh(self.head(skip))

    def decode_chunk(self, z: torch.Tensor, f0: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        out = []
        for i in range(z.shape[-1]):
            n = None if noise is None else noise[..., i * HOP_LENGTH:(i + 1) * HOP_LENGTH]
            out.append(self.decode_step(z[..., i:i + 1], f0[:, i:i + 1], n))
        return torch.cat(out, dim=-1)
