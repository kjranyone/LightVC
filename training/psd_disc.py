"""f0 で条件づけた細かい構造の判別器(diag_ppgvoc61・D600)。波形を指令の f0 の位相で 1 周期 P 点に引き伸ばし(4 kHz の低域通過の後)、4 周期の窓の FFT で
倍音の番号の軸(1/4 倍音きざみ・h ≤ 12)の対数振幅を得て、帯の包絡(2 倍音の移動平均)を引いた細かい構造 z だけを判別器に見せる = 高さが変わっても同じ形・声道の形は見えない。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import nvoc as N

P = 72
WP = 4
HB = 48


def lpf_taps(fc: float = 4000.0, n: int = 127) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float64) - (n - 1) / 2
    h = 2 * fc / N.SR * torch.sinc(2 * fc / N.SR * t) * torch.hann_window(n, periodic=False, dtype=torch.float64)
    return (h / h.sum()).float()


def per_sample_f0(f0: torch.Tensor) -> tuple:
    f0 = f0.float()
    prev = torch.cat([f0[:, :1], f0[:, :-1]], -1)
    fa = torch.where(prev > 0, prev, f0)
    fb = torch.where(f0 > 0, f0, prev)
    B, T = f0.shape
    w = (torch.arange(N.HOP, device=f0.device, dtype=torch.float64) + 1) / N.HOP
    f = (fa.double()[..., None] + (fb - fa).double()[..., None] * w).reshape(B, T * N.HOP)
    return f, N.frame_interp((f0 > 0).float())


def psgram(x: torch.Tensor, f0: torch.Tensor, s_lo: int, n_per: int | None = 48, u: torch.Tensor | None = None, taps: torch.Tensor | None = None) -> tuple:
    """x [B, L](f0 のフレーム 0 の頭 = 標本 0)・f0 [B, T]・s_lo 以降の標本だけ使う。n_per = None は s_lo 以降の全周期(検査用)。
    返り値 z [B, HB, F]・有効 [B, F]。"""
    B, L = x.shape
    taps = lpf_taps().to(x.device) if taps is None else taps
    xl = F.conv1d(x[:, None], taps.view(1, 1, -1), padding=taps.numel() // 2)[:, 0]
    f, v = per_sample_f0(f0)
    n = min(L, f.shape[1])
    f, v, xl = f[:, :n], v[:, :n], xl[:, :n]
    ph = torch.cumsum(f / N.SR, -1)
    p_lo = ph[:, s_lo]
    p_hi = ph[:, n - 2]
    if n_per is None:
        npr = int(max(1.0, float((p_hi - p_lo).min()) - 1))
        p0 = p_lo
    else:
        npr = n_per
        span = (p_hi - p_lo - npr - 1).clamp(min=0)
        p0 = p_lo + (torch.rand(B, device=x.device, dtype=torch.float64) if u is None else u.double()) * span
    M = npr * P
    pm = p0[:, None] + torch.arange(M, device=x.device, dtype=torch.float64)[None] / P
    s = torch.searchsorted(ph.contiguous(), pm.contiguous()).clamp(1, n - 1)
    a_ = ph.gather(1, s - 1)
    b_ = ph.gather(1, s)
    fr = ((pm - a_) / (b_ - a_).clamp(min=1e-12)).clamp(0, 1).float()
    xw = xl.gather(1, s - 1) * (1 - fr) + xl.gather(1, s) * fr
    sf_ = (s - 1).double() + fr.double()
    step = torch.cat([sf_[:, 1:] - sf_[:, :-1], torch.zeros(B, 1, device=x.device, dtype=torch.float64)], 1)
    ok = (v.gather(1, s) >= 0.99) & (v.gather(1, s - 1) >= 0.99) & (step < N.SR / (50.0 * P)) & (pm <= p_hi[:, None])
    fw = xw.unfold(1, WP * P, P)
    okf = ok.unfold(1, WP * P, P).all(-1)
    win = torch.hann_window(WP * P, periodic=True, device=x.device)
    X = torch.fft.rfft(fw * win, dim=-1)[..., 1:HB + 1]
    pw = X.real ** 2 + X.imag ** 2
    lm = 0.5 * torch.log(pw + 1e-6 * pw.mean(-1, keepdim=True) + 1e-12)
    env = F.avg_pool1d(F.pad(lm.reshape(-1, 1, HB), (4, 4), mode="replicate"), 9, 1).reshape(lm.shape)
    return (lm - env).transpose(1, 2), okf


class PSD(torch.nn.Module):
    def __init__(self, ch: int = 32):
        super().__init__()
        self.net = torch.nn.ModuleList([
            torch.nn.Conv2d(1, ch, (5, 3), padding=(2, 1)),
            torch.nn.Conv2d(ch, ch, (5, 3), stride=(2, 1), padding=(2, 1)),
            torch.nn.Conv2d(ch, ch, (5, 3), stride=(2, 1), padding=(2, 1)),
            torch.nn.Conv2d(ch, ch, (3, 3), padding=(1, 1)),
        ])
        self.out = torch.nn.Conv2d(ch, 1, (3, 3), padding=(1, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z[:, None]
        for c in self.net:
            h = F.leaky_relu(c(h), 0.1)
        return self.out(h)[:, 0].mean(1)


def masked_ls(o: torch.Tensor, ok: torch.Tensor, tgt: float) -> torch.Tensor:
    m = ok.float()
    return (((o.float() - tgt) ** 2) * m).sum() / (m.sum() + 1e-6)


def oe_bins() -> tuple:
    return [4 * h - 1 for h in (1, 3, 5, 7)], [4 * h - 1 for h in (2, 4, 6, 8)], [4 * h + 1 for h in range(0, 8)]


def z_stats(z: torch.Tensor, ok: torch.Tensor) -> dict:
    od, ev, va = oe_bins()
    m = ok.float()
    nv = float(m.sum())
    if nv < 1:
        return {"oe_z": float("nan"), "valley": float("nan"), "nf": 0}
    zo = (z[:, od].mean(1) * m).sum() / nv
    ze = (z[:, ev].mean(1) * m).sum() / nv
    zv = (z[:, va].mean(1) * m).sum() / nv
    return {"oe_z": float(zo - ze), "valley": float(zv), "nf": int(nv)}

