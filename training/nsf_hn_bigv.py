from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.utils.checkpoint import checkpoint

from nsf_hn import SourceModuleHnNSF, get_padding, init_weights
from aa import AAActivation, SnakeBeta


def snake_aa(ch: int) -> AAActivation:
    return AAActivation(SnakeBeta(ch))


class AMPBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilations: tuple = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=d, padding=get_padding(k, d)))
            for d in dilations])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(ch, ch, k, 1, dilation=1, padding=get_padding(k, 1)))
            for _ in dilations])
        self.acts1 = nn.ModuleList([snake_aa(ch) for _ in dilations])
        self.acts2 = nn.ModuleList([snake_aa(ch) for _ in dilations])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            xt = c2(a2(c1(a1(x))))
            x = xt + x
        return x

    def remove_wn(self) -> None:
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class NsfBigVGAN(nn.Module):
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
        self.use_ckpt = True
        self.m_source = SourceModuleHnNSF(sr, harmonic_num)
        self.conv_pre = weight_norm(nn.Conv1d(cond_dim, up_init_ch, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.films = nn.ModuleList()
        self.films_art = nn.ModuleList()
        self.acts_pre = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernels)):
            ch_in = up_init_ch // (2 ** i)
            ch_out = up_init_ch // (2 ** (i + 1))
            self.acts_pre.append(snake_aa(ch_in))
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
                self.resblocks.append(AMPBlock(ch_out, rk, rd))
        self.act_post = snake_aa(ch_out)
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
            x = self.acts_pre[i](x)
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
                rb = self.resblocks[i * self.num_kernels + j]
                if self.use_ckpt and self.training and x.requires_grad:
                    r = checkpoint(rb, x, use_reentrant=False)
                else:
                    r = rb(x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = self.act_post(x)
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
    from nsf_hn import NsfHifiGan
    g0 = NsfHifiGan(cond_dim=770, timbre_dim=192)
    g = NsfBigVGAN(cond_dim=770, timbre_dim=192)
    miss, unexp = g.load_state_dict(g0.state_dict(), strict=False)
    conv_miss = [k for k in miss if "filt" not in k and "log_alpha" not in k and "log_beta" not in k]
    print("warm-start: conv/other missing (should be []):", conv_miss)
    print("unexpected (should be []):", list(unexp))
    n_snake = sum(1 for k in miss if "log_alpha" in k)
    print(f"snake groups: {n_snake}")
    cond = torch.randn(1, 770, 40); f0 = torch.rand(1, 40) * 200 + 120; s = torch.randn(1, 192)
    y = g(cond, f0, s)
    print("params", round(sum(p.numel() for p in g.parameters()) / 1e6, 2), "M | out", tuple(y.shape), "exp", 40 * 512)
