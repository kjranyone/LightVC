"""因果アンチエイリアス活性(左寄せFIR・先読み0)。BigVGANのAA活性の原理を因果に作り直したもの。

up2(零挿入→kaiser-sinc低域FIR×2)→SnakeBeta→down2(同FIR→2:1間引き)。FIRは左寄せなので
先読み0だが群遅延を持つ: 2倍レートで (K-1)/2 ずつ、up+downで基底レート (K-1)/2 サンプル。
per-channel FIRは [B*C,1,T] へ畳んで groups=1 の conv1d で計算(grouped conv禁止規約に適合)。
パラメータは SnakeBeta と同一(alpha/beta)=c32 重みから warm-start 可能。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_codec import SnakeBeta

AA_TAPS = 12


def kaiser_sinc(cutoff: float, half_width: float, taps: int) -> torch.Tensor:
    delta_f = 4 * half_width
    amp = 2.285 * (taps // 2 - 1) * math.pi * delta_f + 7.95
    if amp > 50.0:
        beta = 0.1102 * (amp - 8.7)
    elif amp >= 21.0:
        beta = 0.5842 * (amp - 21) ** 0.4 + 0.07886 * (amp - 21.0)
    else:
        beta = 0.0
    win = torch.kaiser_window(taps, beta=beta, periodic=False, dtype=torch.float64)
    t = torch.arange(taps, dtype=torch.float64) - (taps - 1) / 2
    h = 2 * cutoff * torch.sinc(2 * cutoff * t) * win
    return (h / h.sum()).float()


def _fir(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    B, C, T = x.shape
    y = F.conv1d(F.pad(x.reshape(B * C, 1, T), (h.shape[-1] - 1, 0)), h.view(1, 1, -1))
    return y.view(B, C, -1)


class CausalAASnake(SnakeBeta):
    def __init__(self, channels: int, taps: int = AA_TAPS) -> None:
        super().__init__(channels)
        self.register_buffer("h", kaiser_sinc(0.25, 0.3, taps), persistent=False)

    @property
    def delay(self) -> int:
        return (self.h.shape[-1] - 1) // 2

    def up(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.shape[0], x.shape[1], 2 * x.shape[-1], device=x.device, dtype=x.dtype)
        z[..., ::2] = x
        return 2.0 * _fir(z, self.h)

    def down(self, x: torch.Tensor) -> torch.Tensor:
        return _fir(x, self.h)[..., 1::2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(super().forward(self.up(x)))


class FIRStream:
    """左寄せFIRのストリーミング(直近 K-1 サンプルを保持)。"""

    def __init__(self, h: torch.Tensor) -> None:
        self.h = h
        self.state: torch.Tensor | None = None

    def reset(self) -> None:
        self.state = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = self.h.shape[-1]
        if self.state is None:
            self.state = x.new_zeros(x.shape[0], x.shape[1], k - 1)
        j = torch.cat((self.state, x), dim=-1)
        self.state = j[..., -(k - 1):]
        B, C, T = j.shape
        return F.conv1d(j.reshape(B * C, 1, T), self.h.view(1, 1, -1)).view(B, C, -1)


class AASnakeStream:
    def __init__(self, act: CausalAASnake) -> None:
        self.act = act
        self.fu = FIRStream(act.h)
        self.fd = FIRStream(act.h)
        self.phase = 0

    def reset(self) -> None:
        self.fu.reset()
        self.fd.reset()
        self.phase = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(x.shape[0], x.shape[1], 2 * x.shape[-1], device=x.device, dtype=x.dtype)
        z[..., ::2] = x
        u = 2.0 * self.fu(z)
        s = SnakeBeta.forward(self.act, u)
        return self.fd(s)[..., 1::2]


class DelayStream:
    """整数サンプルの純遅延(残差skipの整列用)。"""

    def __init__(self, d: int) -> None:
        self.d = d
        self.state: torch.Tensor | None = None

    def reset(self) -> None:
        self.state = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.d == 0:
            return x
        if self.state is None:
            self.state = x.new_zeros(x.shape[0], x.shape[1], self.d)
        j = torch.cat((self.state, x), dim=-1)
        self.state = j[..., -self.d:]
        return j[..., :x.shape[-1]]


def delay(x: torch.Tensor, d: int) -> torch.Tensor:
    return F.pad(x, (d, 0))[..., :x.shape[-1]] if d else x
