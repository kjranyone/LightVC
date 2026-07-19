from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half = kernel_size // 2
    delta_f = 2 * half_width
    A = 2.285 * (half - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half, half) + 0.5
    else:
        time = torch.arange(kernel_size) - half
    if cutoff == 0:
        filt = torch.zeros_like(time)
    else:
        filt = 2 * cutoff * torch.sinc(2 * cutoff * time)
        filt = filt * window
        filt = filt / filt.sum()
    return filt.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float = 0.5, half_width: float = 0.6,
                 stride: int = 1, kernel_size: int = 12) -> None:
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.pad_left = kernel_size // 2 - int((kernel_size % 2 == 0))
        self.pad_right = kernel_size // 2
        filt = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filt", filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filt.expand(c, -1, -1), stride=self.stride, groups=c)


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.pad = self.kernel_size // ratio - 1
        self.stride = ratio
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filt = kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, self.kernel_size)
        self.register_buffer("filt", filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filt.expand(c, -1, -1), stride=self.stride, groups=c)
        x = x[..., self.pad_left:-self.pad_right]
        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio,
            stride=ratio, kernel_size=self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Snake(nn.Module):
    def __init__(self, ch: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(ch) * alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.view(1, -1, 1)
        return x + (1.0 / (a + 1e-9)) * torch.sin(a * x) ** 2


class SnakeBeta(nn.Module):
    def __init__(self, ch: int, beta_init: float = 0.0) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(ch))
        self.log_beta = nn.Parameter(torch.full((ch,), beta_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.log_alpha.exp().view(1, -1, 1)
        b = self.log_beta.exp().view(1, -1, 1)
        return x + (1.0 / (b + 1e-9)) * torch.sin(a * x) ** 2


class AAActivation(nn.Module):
    def __init__(self, act: nn.Module, up_ratio: int = 2, down_ratio: int = 2) -> None:
        super().__init__()
        self.up = UpSample1d(up_ratio)
        self.down = DownSample1d(down_ratio)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


class AALeaky(nn.Module):
    def __init__(self, slope: float = 0.1) -> None:
        super().__init__()
        self.up = UpSample1d(2)
        self.down = DownSample1d(2)
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.leaky_relu(self.up(x), self.slope))


if __name__ == "__main__":
    x = torch.randn(2, 32, 100)
    for m in [AALeaky(), AAActivation(Snake(32))]:
        y = m(x)
        print(m.__class__.__name__, tuple(x.shape), "->", tuple(y.shape))
