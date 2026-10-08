"""Artic-A2 の DSP 部品の torch 版(学習用・GPU)。numpy 版 artic_dsp と数値一致を単体検査で担保する(`python artic_torch.py`)。

  lar_to_k / k_to_a       LAR → 反射係数 → 予測係数(step-up 再帰)
  env_db                  予測係数 → 全極包絡 [dB](任意の周波数格子・微分可能)
  warp_lar                包絡の周波数伸縮(声道長の変換・入力摂動用・勾配なし)
  levinson                自己相関 → 反射係数
"""
from __future__ import annotations

import math

import numpy as np
import torch

SR = 48000
MU = 0.97


def lar_to_k(lar: torch.Tensor) -> torch.Tensor:
    return -torch.tanh(lar / 2.0)


def k_to_lar(k: torch.Tensor) -> torch.Tensor:
    k = k.clamp(-0.99999, 0.99999)
    return torch.log((1.0 - k) / (1.0 + k))


def k_to_a(k: torch.Tensor) -> torch.Tensor:
    """k [...,p] → a [...,p](a0=1 を除く)。"""
    p = k.shape[-1]
    a = k.new_zeros(k.shape)
    for i in range(p):
        ki = k[..., i:i + 1]
        if i > 0:
            prev = a[..., :i]
            a = torch.cat([prev + ki * prev.flip(-1), ki, a[..., i + 1:]], -1)
        else:
            a = torch.cat([ki, a[..., 1:]], -1)
    return a


def env_db(a: torch.Tensor, freqs: torch.Tensor, deemph: bool = False) -> torch.Tensor:
    """a [...,p] → 20log10|1/A(e^{jω})| [...,F]。deemph=True でプリエンファシスを外した包絡。"""
    p = a.shape[-1]
    w = 2 * math.pi * freqs / SR
    i = torch.arange(p + 1, device=a.device, dtype=a.dtype)
    C = torch.cos(i[:, None] * w[None, :])
    S = torch.sin(i[:, None] * w[None, :])
    A = torch.cat([torch.ones_like(a[..., :1]), a], -1)
    re, im = A @ C, -(A @ S)
    db = -10.0 * torch.log10((re * re + im * im).clamp(min=1e-12))
    if deemph:
        pre = (1 - MU * torch.cos(w)) ** 2 + (MU * torch.sin(w)) ** 2
        db = db - 10.0 * torch.log10(pre)
    return db


def levinson(r: torch.Tensor, order: int) -> torch.Tensor:
    """r [...,order+1] → k [...,order]。無音(誤差 ≤1e-12)は k=0。"""
    shp = r.shape[:-1]
    r = r.reshape(-1, r.shape[-1])
    n = r.shape[0]
    a = r.new_zeros(n, order + 1)
    a[:, 0] = 1.0
    ks = r.new_zeros(n, order)
    err = r[:, 0].clone()
    live = err > 1e-12
    for i in range(1, order + 1):
        acc = r[:, i] + (a[:, 1:i] * r[:, 1:i].flip(-1)).sum(1) if i > 1 else r[:, i].clone()
        k = torch.where(live, -acc / torch.where(live, err, torch.ones_like(err)), torch.zeros_like(err))
        k = k.clamp(-0.99999, 0.99999)
        prev = a[:, 1:i].clone()
        a[:, 1:i] = prev + k[:, None] * prev.flip(-1)
        a[:, i] = k
        ks[:, i - 1] = k
        err = err * (1.0 - k * k)
        live = live & (err > 1e-12)
    return ks.reshape(*shp, order)


@torch.no_grad()
def warp_lar(lar: torch.Tensor, alpha: torch.Tensor, nfft: int = 2048, lag_bw: float = 60.0, wnc: float = 1e-4) -> torch.Tensor:
    """lar [...,p]・alpha(lar[...,0] と同形か放送可能)→ 伸縮後の LAR。artic_dsp.warp_lar と同じ手順(f の包絡に f/α の値を置く)。"""
    p = lar.shape[-1]
    dt = torch.float64
    a = k_to_a(lar_to_k(lar.to(dt)))
    A = torch.fft.rfft(torch.cat([torch.ones_like(a[..., :1]), a], -1), n=nfft)
    f = torch.fft.rfftfreq(nfft, 1 / SR, device=lar.device, dtype=dt)
    pre = (1 - MU * torch.cos(2 * math.pi * f / SR)) ** 2 + (MU * torch.sin(2 * math.pi * f / SR)) ** 2
    logPd = -torch.log((A.real ** 2 + A.imag ** 2).clamp(min=1e-12)) - torch.log(pre.clamp(min=1e-12))
    al = alpha.to(dt)
    while al.dim() < lar.dim():
        al = al.unsqueeze(-1)
    src = (f / al).clamp(0, f[-1])
    idx = src / (f[1] - f[0])
    i0 = idx.floor().long().clamp(0, len(f) - 2)
    fr = idx - i0
    lg0 = torch.gather(logPd, -1, i0.expand(logPd.shape))
    lg1 = torch.gather(logPd, -1, (i0 + 1).expand(logPd.shape))
    Pw = torch.exp(lg0 * (1 - fr) + lg1 * fr) * pre
    r = torch.fft.irfft(Pw, n=nfft)[..., :p + 1]
    i = torch.arange(p + 1, device=lar.device, dtype=dt)
    r = r * torch.exp(-0.5 * (2 * math.pi * lag_bw * i / SR) ** 2)
    r[..., 0] = r[..., 0] * (1.0 + wnc)
    return k_to_lar(levinson(r, p)).to(lar.dtype)


def _selftest() -> None:
    import artic_dsp as D
    rng = np.random.default_rng(0)
    x = rng.standard_normal(24000)
    x = np.convolve(x, [1, 0.9, 0.5], mode="same")
    lar = D.lar_frames(x, 24)
    t = torch.from_numpy(lar)
    a_np = D.k_to_a(D.lar_to_k(lar))
    a_t = k_to_a(lar_to_k(t)).numpy()
    print("k_to_a max diff", float(np.abs(a_np - a_t).max()))
    for al in (0.85, 1.0, 1.14):
        w_np = D.warp_lar(lar, al)
        w_t = warp_lar(t, torch.full((lar.shape[0],), al, dtype=torch.float64)).numpy()
        print(f"warp α={al} max |ΔLAR| numpy vs torch", float(np.abs(w_np - w_t).max()))
    fr = torch.tensor([100.0, 1000.0, 5000.0], dtype=torch.float64)
    e = env_db(k_to_a(lar_to_k(t[:2])), fr)
    A = np.concatenate([[1.0], a_np[0]])
    w = 2 * np.pi * fr.numpy() / SR
    ref = -20 * np.log10(np.abs(np.exp(-1j * np.outer(w, np.arange(25))) @ A))
    print("env_db max diff", float(np.abs(e[0].numpy() - ref).max()))


if __name__ == "__main__":
    _selftest()
