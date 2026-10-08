"""PhysVC-ZS の物理変換(current/physvc_zs.md §2)。学習用(torch・包絡領域・微分可能)と推論用(numpy・LPC 再推定)を同じ式で持つ。

物理パラメータ差 dphi = φ_T − φ_S(静的・6 次元): [v, w1, w2, w3, tau, rho]
  包絡 E(f)[dB・プリエンファシスを外した真の包絡] に対し
    1. 構音範囲: E1 = m + exp(rho)·(E − m)         m = 元話者の平均包絡(登録時の定数)
    2. 声道長:   E2(f) = E1(f / α(f)),  log α(f) = v + 区分線形(log f; F1/F2/F3 域の中心に w1..w3 − mean(w)・範囲外は端の値)(平均 0 で v を一様成分に一意化)
    3. 傾斜:     E3(f) = E2(f) + tau·log2(f / 1000)
推論では E3 の線形周波数格子のパワー → プリエンファシスを戻す → 自己相関 → Levinson → LAR(フレームごと・因果・決定的)。

    uv run python physvc.py   # 単体検査
"""
from __future__ import annotations

import math

import numpy as np
import torch

import artic_dsp as D
import artic_torch as AT

GRID = np.geomspace(80.0, 16000.0, 128)
GRID_W = np.geomspace(40.0, 23000.0, 224)
KNOT_HZ = np.array([math.sqrt(250 * 1000), math.sqrt(800 * 2800), math.sqrt(2200 * 4500)])
V, W1, W2, W3, TAU, RHO = range(6)
N_PHI = 6


def log_alpha(f: np.ndarray | torch.Tensor, dphi):
    """f [F] → log α(f) [...,F](dphi [...,6])。"""
    if isinstance(f, torch.Tensor):
        lk = torch.log(torch.as_tensor(KNOT_HZ, dtype=f.dtype, device=f.device))
        lf = torch.log(f.clamp(min=1.0))
        w = dphi[..., W1:W3 + 1]
        w = w - w.mean(-1, keepdim=True)
        t = ((lf - lk[0]) / (lk[1] - lk[0])).clamp(0, 1)
        u = ((lf - lk[1]) / (lk[2] - lk[1])).clamp(0, 1)
        seg = torch.where(lf[None] <= lk[1], w[..., :1] + (w[..., 1:2] - w[..., :1]) * t,
                          w[..., 1:2] + (w[..., 2:3] - w[..., 1:2]) * u) if w.dim() == 2 else None
        return dphi[..., V:V + 1] + seg
    lf = np.log(np.maximum(f, 1.0))
    lk = np.log(KNOT_HZ)
    w = dphi[W1:W3 + 1] - np.mean(dphi[W1:W3 + 1])
    return dphi[V] + np.interp(lf, lk, w)


BAND = (GRID >= 150.0) & (GRID <= 12000.0)
BAND_W = (GRID_W >= 150.0) & (GRID_W <= 12000.0)


def band_of(n: int) -> np.ndarray:
    return BAND if n == len(GRID) else BAND_W


def shape(E):
    """包絡の形(フレームごとに帯域 150–12000Hz 内の平均 dB を引く)。LPC の再推定は利得を持たない(残差が運ぶ)ので、学習と推論は形で揃える。"""
    b = band_of(E.shape[-1])
    b = torch.as_tensor(b, device=E.device) if isinstance(E, torch.Tensor) else b
    return E - E[..., b].mean(-1, keepdims=True)


def env_grid(lar: torch.Tensor, grid: np.ndarray = GRID) -> torch.Tensor:
    """lar [...,24] → 真の包絡 [dB] on grid [...,F]。"""
    g = torch.as_tensor(grid, dtype=lar.dtype, device=lar.device)
    return AT.env_db(AT.k_to_a(AT.lar_to_k(lar)), g, deemph=True)


def transform_env(E: torch.Tensor, m: torch.Tensor, dphi: torch.Tensor) -> torch.Tensor:
    """E [B,T,F]・m [B,F]・dphi [B,6] → 変換後の包絡 [B,T,F](E の格子 GRID か GRID_W 上・微分可能)。"""
    grid = GRID if E.shape[-1] == len(GRID) else GRID_W
    g = torch.as_tensor(grid, dtype=E.dtype, device=E.device)
    lg = torch.log(g)
    E1 = m[:, None] + torch.exp(dphi[:, RHO])[:, None, None] * (E - m[:, None])
    src = lg[None] - log_alpha(g, dphi)
    step = lg[1] - lg[0]
    idx = ((src - lg[0]) / step).clamp(0, len(grid) - 1 - 1e-6)
    i0 = idx.floor().long()
    fr = idx - i0
    i0 = i0[:, None].expand(E1.shape)
    fr = fr[:, None].expand(E1.shape)
    E2 = torch.gather(E1, -1, i0) * (1 - fr) + torch.gather(E1, -1, (i0 + 1).clamp(max=len(grid) - 1)) * fr
    return E2 + dphi[:, TAU, None, None] * torch.log2(g / 1000.0)[None, None]


def phys_lar(lar: np.ndarray, m_grid: np.ndarray, dphi: np.ndarray, nfft: int = 2048,
             lag_bw: float = 60.0, wnc: float = 1e-4) -> np.ndarray:
    """推論用: lar [K,24]・m_grid [F](GRID 上の元話者平均包絡 dB)・dphi [6] → 変換後の LAR [K,24](フレームごと・決定的)。"""
    K, p = lar.shape
    a = D.k_to_a(D.lar_to_k(lar))
    A = np.fft.rfft(np.concatenate([np.ones((K, 1)), a], 1), n=nfft, axis=1)
    f = np.fft.rfftfreq(nfft, 1 / D.SR)
    pre = np.abs(1 - D.MU * np.exp(-2j * np.pi * f / D.SR)) ** 2
    Edb = -10 * np.log10(np.maximum(np.abs(A) ** 2, 1e-12)) - 10 * np.log10(np.maximum(pre, 1e-12))
    grid = GRID if len(m_grid) == len(GRID) else GRID_W
    m = np.interp(np.log(np.maximum(f, 1.0)), np.log(grid), m_grid)
    E1 = m + np.exp(dphi[RHO]) * (Edb - m)
    src = np.clip(f / np.exp(log_alpha(f, dphi)), 0, f[-1])
    idx = src / (f[1] - f[0])
    i0 = np.clip(np.floor(idx).astype(int), 0, len(f) - 2)
    fr = idx - i0
    E2 = E1[:, i0] * (1 - fr) + E1[:, i0 + 1] * fr
    E3 = E2 + dphi[TAU] * np.log2(np.maximum(f, 1.0) / 1000.0)
    Pw = 10 ** (E3 / 10) * pre
    r = np.fft.irfft(Pw, n=nfft, axis=1)[:, :p + 1]
    i = np.arange(p + 1)
    r = r * np.exp(-0.5 * (2 * np.pi * lag_bw * i / D.SR) ** 2)
    r[:, 0] *= 1.0 + wnc
    _, ks = D.levinson(r, p)
    return D.k_to_lar(ks)


def _selftest() -> None:
    rng = np.random.default_rng(0)
    x = np.convolve(rng.standard_normal(48000), [1, 1.6, 0.9, 0.3], mode="same")
    x[:24000] = np.convolve(rng.standard_normal(24000), [1, -0.5, 0.7], mode="same")
    lar = D.lar_frames(x, 24)
    t = torch.from_numpy(lar)
    E = env_grid(t)
    m = E.mean(0)
    for name, dphi in (("identity", np.zeros(6)), ("v=ln1.139", np.array([np.log(1.139), 0, 0, 0, 0, 0])),
                       ("nonuniform+tilt+rho", np.array([0.1, 0.05, -0.03, 0.02, -2.0, 0.2]))):
        ref = shape(transform_env(E[None], m[None], torch.from_numpy(dphi)[None])[0])
        lar2 = phys_lar(lar, m.numpy(), dphi)
        got = shape(env_grid(torch.from_numpy(lar2)))
        err = (got - ref)[:, BAND].numpy()
        print(f"{name:22s} refit-vs-envelope 形の LSD median {np.median(np.sqrt((err ** 2).mean(1))):.3f} dB  "
              f"max|ΔLAR| {np.abs(lar2 - lar).max():.4f}")
    w_np = D.warp_lar(lar, 1.139)
    l2 = phys_lar(lar, m.numpy(), np.array([np.log(1.139), 0, 0, 0, 0, 0]))
    print("uniform α vs artic_dsp.warp_lar max|ΔLAR|", float(np.abs(w_np - l2).max()))
    d = torch.zeros(1, 6, requires_grad=True)
    loss = (transform_env(E[None].float(), m[None].float(), d.float() + torch.tensor([[0.1, 0, 0, 0, 1, 0.1]])) - E[None].float()).pow(2).mean()
    loss.backward()
    print("grad finite", bool(torch.isfinite(d.grad).all()), "grad", d.grad.numpy().round(3))


if __name__ == "__main__":
    _selftest()
