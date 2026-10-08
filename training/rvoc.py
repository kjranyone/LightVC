"""出力部(レンダラ)rvoc: 条件(DCT 包絡 c0..c24・f0・有声・4 帯の周期性・200fps・0 = 無音)+ 励起(f0 のパルス列・雑音)→ 48kHz。因果。current/renderer.md。

nvoc との違い: 引き上げを転置畳み込みから「直線補間 + 因果畳み込み」に替える(転置畳み込みは一定の条件でもフレーム周期の構造を作り
変調の線 +37dB = ガビガビの構造の原因だった)。条件は変換器が出す界面(DCT24 包絡 c0..c24・log f0・有声・4 帯の周期性 = 倍音の縞の深さ)。レベルは c0 だけが運ぶ(重複なし)。
段: 200fps → ×5 → ×4 → ×4 → ×3 = 48kHz。各段 = [LeakyReLU → 直線補間 ×r → 因果 conv(k = 2r+1)→ + 励起(strided 因果 conv)→ ResBlock 平均]。
時刻規約は nvoc と同じ: フレーム t は (t+1)H に終わる窓で決まり、出力ブロック t はフレーム ≤ t だけで決まる(補間はフレーム t−1 → t)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

import nvoc as N

UPS = (5, 4, 4, 3)
N_DCT = 25
N_AP = 4
D_COND = N_DCT + 2 + N_AP
D_COND_MEL = 128 + 2 + N_AP


class CConv(nn.Module):
    def __init__(self, ci: int, co: int, k: int, d: int = 1, stride: int = 1, lpad: int | None = None):
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(ci, co, k, dilation=d, stride=stride))
        self.lpad = (k - 1) * d if lpad is None else lpad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.lpad, 0)))


def interp_up(x: torch.Tensor, r: int) -> torch.Tensor:
    """[B,C,T] → [B,C,T·r]: ブロック t = x[t−1] → x[t] の直線補間(x[−1] = 0 = ストリームの初期状態・Rust と同じ)= 因果。"""
    B, C, T = x.shape
    prev = torch.cat([torch.zeros_like(x[..., :1]), x[..., :-1]], -1)
    w = (torch.arange(r, device=x.device, dtype=x.dtype) + 1) / r
    return (prev[..., None] + (x - prev)[..., None] * w).reshape(B, C, T * r)


class RB(nn.Module):
    def __init__(self, ch: int, k: int, dils: tuple):
        super().__init__()
        self.c1 = nn.ModuleList(CConv(ch, ch, k, d) for d in dils)
        self.c2 = nn.ModuleList(CConv(ch, ch, k, 1) for _ in dils)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for a, b in zip(self.c1, self.c2):
            x = x + b(F.leaky_relu(a(F.leaky_relu(x, 0.1)), 0.1))
        return x


class RVoc(nn.Module):
    def __init__(self, ch: int = 256, kernels: tuple = (3, 7, 11), dils: tuple = (1, 3, 5), d_cond: int = D_COND):
        super().__init__()
        self.cfg = {"ch": ch, "kernels": list(kernels), "dils": list(dils), "ups": list(UPS), "d_cond": d_cond}
        self.pre = CConv(d_cond, ch, 3)
        self.up = nn.ModuleList()
        self.src = nn.ModuleList()
        self.res = nn.ModuleList()
        tot = 1
        for i, r in enumerate(UPS):
            ci, co = ch >> i, ch >> (i + 1)
            self.up.append(CConv(ci, co, 2 * r + 1))
            tot *= r
            s = N.HOP // tot
            self.src.append(CConv(2, co, 2 * s, stride=s, lpad=s) if s > 1 else CConv(2, co, 3))
            self.res.append(nn.ModuleList(RB(co, k, dils) for k in kernels))
        self.post = CConv(ch >> len(UPS), 1, 7)

    def forward(self, cond: torch.Tensor, exc: torch.Tensor) -> torch.Tensor:
        """cond [B, D_COND, T]・exc [B, 2, T·HOP](パルス列・雑音)→ y [B, T·HOP]。"""
        x = self.pre(cond)
        for i, r in enumerate(UPS):
            x = self.up[i](interp_up(F.leaky_relu(x, 0.1), r))
            x = x + self.src[i](exc)
            x = sum(rb(x) for rb in self.res[i]) / len(self.res[i])
        return self.post(F.leaky_relu(x, 0.01)).squeeze(1)


def macs_per_second(m: RVoc) -> float:
    ch, ks, ds = m.cfg["ch"], m.cfg["kernels"], m.cfg["dils"]
    rate = N.SR / N.HOP
    tot = rate * 3 * m.cfg.get("d_cond", D_COND) * ch
    for i, r in enumerate(UPS):
        ci, co = ch >> i, ch >> (i + 1)
        rate *= r
        tot += rate * (2 * r + 1) * ci * co
        tot += rate * 2 * co * 2 * max(1, N.HOP // int(round(rate / (N.SR / N.HOP))))
        tot += rate * sum(2 * len(ds) * k * co * co for k in ks)
    tot += N.SR * 7 * (ch >> len(UPS))
    return tot


def _selftest() -> None:
    import numpy as np
    torch.manual_seed(0)
    m = RVoc().eval()
    print("params (M)", round(sum(p.numel() for p in m.parameters()) / 1e6, 3), "| GMAC/s", round(macs_per_second(m) / 1e9, 2))
    T = 400
    with torch.no_grad():
        cond = torch.randn(1, D_COND, T)
        exc = torch.randn(1, 2, T * N.HOP)
        y = m(cond, exc)
        for cut in (137, 288):
            c2, e2 = cond.clone(), exc.clone()
            c2[..., cut:] = torch.randn_like(c2[..., cut:])
            e2[..., cut * N.HOP:] = torch.randn_like(e2[..., cut * N.HOP:])
            d = (m(c2, e2) - y).abs()[0]
            first = int((d > 0).nonzero()[0])
            assert first >= cut * N.HOP, (cut, first)
            print(f"未来不変性: フレーム {cut}(サンプル {cut * N.HOP})の編集 → 最初の変化 {first}")
        const = torch.zeros(1, D_COND, T) + torch.randn(1, D_COND, 1)
        z = torch.zeros(1, 2, T * N.HOP)
        yc = m(const, z)[0, 50 * N.HOP:]
        ac = float(yc.std() / (yc.abs().mean() + 1e-12))
        print(f"構造: 一定の条件・励起 0 → 出力の変動/平均 = {ac:.2e}(補間 + 畳み込みは一定の入力から一定しか作れない = フレーム周期の成分は重みによらず 0)")
        assert ac < 1e-4, ac
        nv = N.NVoc().eval()
        torch.manual_seed(1)
        for u in nv.up:
            nn.init.normal_(u.parametrizations.weight.original1, 0.0, 0.05)
        mel_c = torch.zeros(1, N.N_MEL, T) + torch.randn(1, N.N_MEL, 1)
        yn = nv.generate(mel_c, torch.zeros(1, 2, T * N.HOP))[0, 50 * N.HOP:]
        acn = float(yn.std() / (yn.abs().mean() + 1e-12))
        per = yn.reshape(-1, N.HOP)
        print(f"対照 nvoc(転置畳み込み・同じ条件): 変動/平均 = {acn:.2e}・フレームごとの波形の一致 {float((per - per[:1]).abs().max()):.1e}(= フレーム周期で繰り返す成分)")
    print("rvoc selftest OK")


if __name__ == "__main__":
    _selftest()
