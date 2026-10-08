"""nvoc の変種: 指定した段の活性化を因果の折り返し防止版にする(2 倍に上げる → LeakyReLU → 下げる・最小位相 FIR)。

nvoc.NVoc と同じ重み構成(state_dict のキーも同じ)。フィルタは固定(学習しない)。既定は段 2・3(16kHz・48kHz)と出力前の活性化。
最小位相 FIR の群遅延は 2 倍レートで ~1〜2 サンプル(段 2・3 で合計 ~1.2ms)。診断用(Rust は未対応)。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import firwin, minimum_phase

import nvoc as N


def _minphase_lowpass(n_lin: int = 31, cutoff: float = 0.45) -> np.ndarray:
    h = firwin(n_lin, cutoff, window=("kaiser", 8.0))
    h2 = np.convolve(h, h)
    m = minimum_phase(h2, method="homomorphic")
    return m / m.sum()


class NVocAA(N.NVoc):
    def __init__(self, *args, aa_stages: tuple[int, ...] = (2, 3), **kw):
        super().__init__(*args, **kw)
        self.aa_stages = set(aa_stages)
        self.cfg["aa_stages"] = sorted(self.aa_stages)
        self.register_buffer("h_aa", torch.from_numpy(_minphase_lowpass()).float())

    def _aa(self, x: torch.Tensor, s: float) -> torch.Tensor:
        C, L = x.shape[1], self.h_aa.numel()
        k = self.h_aa.flip(0).view(1, 1, L).expand(C, 1, L)
        u = torch.zeros(x.shape[0], C, x.shape[-1] * 2, device=x.device, dtype=x.dtype)
        u[..., ::2] = x * 2
        u = F.conv1d(F.pad(u, (L - 1, 0)), k, groups=C)
        u = F.leaky_relu(u, s)
        return F.conv1d(F.pad(u, (L - 1, 0)), k, groups=C)[..., ::2]

    def _act(self, i: int):
        return self._aa if i in self.aa_stages else (lambda z, s: F.leaky_relu(z, s))

    def generate(self, mel: torch.Tensor, exc: torch.Tensor) -> torch.Tensor:
        T = mel.shape[-1]
        x = self.pre(mel)
        n = T
        for i, r in enumerate(N.UPS):
            pre_act = self._act(i - 1) if i > 0 else self._act(-1)
            n *= r
            o = 0 if self.causal else r // 2
            x = self.up[i](pre_act(x, 0.1))[..., o:o + n]
            x = x + self.src[i](exc)
            act = self._act(i)
            outs = []
            for rb in self.res[i]:
                h = x
                for a_, b_ in zip(rb.c1, rb.c2):
                    h = h + b_(act(a_(act(h, 0.1)), 0.1))
                outs.append(h)
            x = sum(outs) / len(outs)
        return self.post(self._act(len(N.UPS) - 1)(x, 0.01)).squeeze(1)


def _selftest() -> None:
    torch.manual_seed(0)
    m = NVocAA().eval()
    h = m.h_aa.numpy()
    w = np.fft.rfft(h, 4096)
    f = np.linspace(0, 1, len(w))
    pb = np.abs(w[f < 0.4]).min()
    sb = np.abs(w[f > 0.6]).max()
    gd = np.sum(np.arange(len(h)) * h ** 2) / np.sum(h ** 2)
    print(f"AA filter taps {len(h)} passband min {pb:.3f} stopband max {sb:.4f} energy-centroid delay {gd:.2f} samples@2x")
    T = 60
    x = torch.randn(1, N.WIN - N.HOP + T * N.HOP) * 0.1
    f0 = torch.zeros(1, T)
    noise = torch.randn(1, T * N.HOP)
    with torch.no_grad():
        y = m(x, f0, noise)
        cut = 30
        x2 = x.clone()
        x2[..., N.WIN - N.HOP + cut * N.HOP:] = torch.randn_like(x2[..., N.WIN - N.HOP + cut * N.HOP:])
        n2 = noise.clone()
        n2[:, cut * N.HOP:] = torch.randn_like(n2[:, cut * N.HOP:])
        d = (m(x2, f0, n2) - y).abs()[0]
        first = int((d > 1e-7).nonzero()[0])
        assert first >= cut * N.HOP, first
        print(f"future invariance: edit at block {cut} → first change {first} (≥ {cut * N.HOP}) OK")


if __name__ == "__main__":
    _selftest()
