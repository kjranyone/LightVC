"""推論だけ(D618): 励起(train_rvoc.excitation = 帯域制限パルス列 × 0.1 と雑音 × 0.01)の第 1 倍音の重みを上げる。
有声で f0 < ft のサンプルだけ第 1 倍音の振幅を 10^(gain_db/20) 倍にし、単位 RMS に戻す(nvoc.harmonic_source と同じ位相・帯域制限)。gain_db = 0 で元と一致。
EXH1 = {"on", "gain_db", "ft"} を描画のたびに設定し、install() で train_rvoc.excitation を差し替える。
"""
from __future__ import annotations

import math

import torch

import nvoc as N

EXH1: dict = {"on": False, "gain_db": 0.0, "ft": 400.0}


def harmonic_source_h1(f0: torch.Tensor, gain_db: float, ft: float) -> torch.Tensor:
    f0 = f0.float()
    prev = torch.cat([f0[:, :1], f0[:, :-1]], -1)
    fa = torch.where(prev > 0, prev, f0)
    fb = torch.where(f0 > 0, f0, prev)
    B, T = f0.shape
    w = (torch.arange(N.HOP, device=f0.device, dtype=torch.float64) + 1) / N.HOP
    f = (fa.double()[..., None] + (fb - fa).double()[..., None] * w).reshape(B, T * N.HOP)
    v = N.frame_interp((f0 > 0).float())
    ph = torch.remainder(torch.cumsum(f / N.SR, -1), 1.0)
    f32 = f.float()
    g1 = torch.where((f32 > 0) & (f32 < ft), torch.full_like(f32, 10 ** (gain_db / 20)), torch.ones_like(f32))
    out = torch.zeros_like(f32)
    nrm = torch.zeros_like(f32)
    kmax = int(min(N.K_MAX, math.ceil(N.H_MAX / max(float(f32[f32 > 0].min()) if (f32 > 0).any() else N.H_MAX, 1.0))))
    for k0 in range(1, kmax + 1, 16):
        k = torch.arange(k0, min(k0 + 16, kmax + 1), device=f0.device, dtype=torch.float64)
        a = ((N.H_MAX - k.float()[:, None, None] * f32[None]) / N.H_ROLL).clamp(0, 1)
        if k0 == 1:
            a = torch.cat([a[:1] * g1[None], a[1:]], 0)
        s = torch.sin((torch.remainder(ph[None] * k[:, None, None], 1.0) * (2 * math.pi)).float())
        out = out + (a * s).sum(0)
        nrm = nrm + (a * a).sum(0)
    return v * out / torch.sqrt(nrm / 2).clamp(min=1e-3)


def excitation(f0: torch.Tensor, n: int, gen: torch.Generator | None = None) -> torch.Tensor:
    if not EXH1["on"]:
        h = N.harmonic_source(f0)[:, :n]
    else:
        h = harmonic_source_h1(f0, EXH1["gain_db"], EXH1["ft"])[:, :n]
    nz = torch.randn(h.shape, generator=gen, device="cpu").to(h.device) if gen is not None else torch.randn_like(h)
    return torch.stack([h * 0.1, nz * 0.01], 1)


def install() -> None:
    import train_rvoc as TR
    TR.excitation = excitation
