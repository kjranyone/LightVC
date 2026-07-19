"""Frame-rate envelope predictor for NSF-LTV (current/vocoder.md §3.4).

Causal TCN, groups=1 (XPU-safe), frame-rate only — its nonlinearities never
touch the audio signal (no aliasing concern). Heads: H_v/H_n log-mag envelopes
(nb bins, linear grid), pitch-sync modulation depth d, subframe noise gains
a^(1..4), soft voicing v. Head init per P-A: NO zero-init — bias = data-mean
log envelope, weight std 0.02.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from nsf_hn3 import CausalConv1d, ChanLayerNorm


class TcnBlock(nn.Module):
    def __init__(self, ch: int, dilation: int) -> None:
        super().__init__()
        self.norm = ChanLayerNorm(ch)
        self.conv = CausalConv1d(ch, ch, 3, dilation=dilation)
        self.act = nn.GELU()
        self.proj = weight_norm(nn.Conv1d(ch, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.act(self.conv(self.norm(x))))
        return x + h


class LtvFrameNet(nn.Module):
    def __init__(self, cond_dim: int = 130, ch: int = 384, nb: int = 1025,
                 n_blocks: int = 10,
                 hv_bias: torch.Tensor | None = None,
                 hn_bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.nb = nb
        self.pre = CausalConv1d(cond_dim, ch, 7)
        dils = [1, 2, 4, 8, 16] * (n_blocks // 5)
        self.blocks = nn.ModuleList([TcnBlock(ch, d) for d in dils])
        self.head = nn.Conv1d(ch, 2 * nb + 1 + 4 + 1, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            if hv_bias is not None:
                self.head.bias[:nb] = hv_bias
            if hn_bias is not None:
                self.head.bias[nb:2 * nb] = hn_bias

    def forward(self, cond: torch.Tensor) -> dict:
        x = self.pre(cond)
        for blk in self.blocks:
            x = blk(x)
        o = self.head(x)
        nb = self.nb
        return {
            "h_v": o[:, :nb].transpose(1, 2),
            "h_n": o[:, nb:2 * nb].transpose(1, 2),
            "d": torch.sigmoid(o[:, 2 * nb]),
            "a": F.softplus(o[:, 2 * nb + 1:2 * nb + 5]).transpose(1, 2),
            "v": torch.sigmoid(o[:, 2 * nb + 5]),
        }


def smooth_reg(h: torch.Tensor) -> torch.Tensor:
    return (h[..., 1:] - h[..., :-1]).pow(2).mean()


if __name__ == "__main__":
    net = LtvFrameNet()
    c = torch.randn(2, 130, 40)
    o = net(c)
    print({k: tuple(v.shape) for k, v in o.items()},
          "params", sum(p.numel() for p in net.parameters()) / 1e6, "M")
