from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

LRELU = 0.1


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(k: int, d: int = 1) -> int:
    return int((k * d - d) / 2)


class SineGen(nn.Module):
    def __init__(self, sr: int, harmonic_num: int = 8, sine_amp: float = 0.1,
                 noise_std: float = 0.003, voiced_threshold: float = 10.0) -> None:
        super().__init__()
        self.sr = sr
        self.dim = harmonic_num + 1
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.uv_noise = 0.0
        self.vt = voiced_threshold

    @torch.no_grad()
    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        fn = f0 * torch.arange(1, self.dim + 1, device=f0.device, dtype=f0.dtype)
        rad = (fn / self.sr) % 1.0
        phase = torch.cumsum(rad, dim=1) * 2.0 * np.pi
        sine = torch.sin(phase) * self.sine_amp
        uv = (f0 > self.vt).float()
        noise_amp = self.noise_std + (1.0 - uv) * self.uv_noise
        noise = torch.randn_like(sine) * noise_amp
        return sine * uv + noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sr: int, harmonic_num: int = 8) -> None:
        super().__init__()
        self.sine_gen = SineGen(sr, harmonic_num)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0_upsampled: torch.Tensor) -> torch.Tensor:
        sine = self.sine_gen(f0_upsampled)
        return self.l_tanh(self.l_linear(sine))


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilations: tuple = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=d, padding=get_padding(k, d)))
            for d in dilations])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=1, padding=get_padding(k, 1)))
            for _ in dilations])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
            x = xt + x
        return x

    def remove_wn(self) -> None:
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class NsfHifiGan(nn.Module):
    def __init__(self, cond_dim: int, sr: int = 44100, hop: int = 512,
                 upsample_rates: tuple = (8, 4, 2, 2, 2, 2),
                 upsample_kernels: tuple = (16, 8, 4, 4, 4, 4),
                 up_init_ch: int = 512, harmonic_num: int = 8,
                 resblock_ks: tuple = (3, 7, 11),
                 resblock_ds: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 timbre_dim: int = 0, art_dim: int = 0) -> None:
        super().__init__()
        self.hop = hop
        self.num_ups = len(upsample_rates)
        self.num_kernels = len(resblock_ks)
        self.timbre_dim = timbre_dim
        self.art_dim = art_dim
        self.m_source = SourceModuleHnNSF(sr, harmonic_num)
        self.conv_pre = weight_norm(nn.Conv1d(cond_dim, up_init_ch, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.films = nn.ModuleList()
        self.films_art = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernels)):
            ch_in = up_init_ch // (2 ** i)
            ch_out = up_init_ch // (2 ** (i + 1))
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                ch_in, ch_out, k, u, padding=(k - u) // 2)))
            stride = int(np.prod(upsample_rates[i + 1:]))
            if stride > 1:
                self.noise_convs.append(nn.Conv1d(
                    1, ch_out, 2 * stride, stride, padding=stride // 2))
            else:
                self.noise_convs.append(nn.Conv1d(1, ch_out, 1))
            if timbre_dim:
                film = nn.Linear(timbre_dim, 2 * ch_out)
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)
                self.films.append(film)
            if art_dim:
                fa = nn.Linear(art_dim, 2 * ch_out)
                nn.init.zeros_(fa.weight)
                nn.init.zeros_(fa.bias)
                self.films_art.append(fa)
            for rk, rd in zip(resblock_ks, resblock_ds):
                self.resblocks.append(ResBlock(ch_out, rk, rd))
        self.conv_post = weight_norm(nn.Conv1d(ch_out, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, cond: torch.Tensor, f0_frame: torch.Tensor,
                s: torch.Tensor = None, s_art: torch.Tensor = None) -> torch.Tensor:
        f0_up = F.interpolate(f0_frame.unsqueeze(1), scale_factor=self.hop,
                              mode="linear", align_corners=False).transpose(1, 2)
        source = self.m_source(f0_up).transpose(1, 2)
        x = self.conv_pre(cond)
        for i in range(self.num_ups):
            x = F.leaky_relu(x, LRELU)
            x = self.ups[i](x)
            x = x + self.noise_convs[i](source)
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
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU)
        x = torch.tanh(self.conv_post(x))
        return x

    def remove_wn(self) -> None:
        for u in self.ups:
            remove_weight_norm(u)
        for r in self.resblocks:
            r.remove_wn()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


if __name__ == "__main__":
    g = NsfHifiGan(cond_dim=130)
    cond = torch.randn(2, 130, 40)
    f0 = torch.rand(2, 40) * 200 + 80
    y = g(cond, f0)
    print("params", round(sum(p.numel() for p in g.parameters()) / 1e6, 2), "M")
    print("cond", tuple(cond.shape), "-> wav", tuple(y.shape), "expected T", 40 * 512)
