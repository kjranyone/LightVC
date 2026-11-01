"""フレーム同期の高域の包絡の型(ガビ = フレーム周期 200 Hz の包絡変調・D434〜D437)。
2–4 / 4–8 / 8–12 kHz の帯の瞬時パワーを HOP ごとに折りたたみ、全フレームのパワーの和で重みづけた位相の型(平均 1)の分散。
本物の音声はフレームの格子に同期しないので型は平ら(分散 ≈ 有限のフレーム数の雑音)。格子の位置がずれても型が回るだけで次数ごとのパワーは不変。
"""
from __future__ import annotations

import torch

BANDS = ((2000.0, 4000.0), (4000.0, 8000.0), (8000.0, 12000.0))


def fsync(y: torch.Tensor, hop: int = 240, sr: int = 48000, kmax: int = 3) -> torch.Tensor:
    """y [B, L] → [B](3 帯の平均)。kmax > 0: 型の DFT の 1..kmax 次(200 / 400 / 600 Hz)のパワーだけ(高次は有限のフレームの雑音が主)。"""
    n = y.shape[-1] // hop
    y = y[:, : n * hop].float()
    Y = torch.fft.rfft(y)
    f = torch.fft.rfftfreq(n * hop, 1.0 / sr).to(y.device)
    out = []
    for lo, hi in BANDS:
        yb = torch.fft.irfft(Y * ((f >= lo) & (f < hi)).to(Y.dtype), n * hop)
        P = (yb * yb).reshape(y.shape[0], n, hop)
        prof = P.sum(1) / (P.sum((1, 2))[:, None] / hop + 1e-12)
        if kmax > 0:
            out.append((torch.fft.rfft(prof).abs()[:, 1:kmax + 1] ** 2).sum(-1) / hop ** 2)
        else:
            out.append(((prof - 1.0) ** 2).mean(-1))
    return torch.stack(out, -1).mean(-1)


def fline(y: torch.Tensor, sr: int = 48000, harm: tuple = (200.0, 400.0, 600.0), on: float = 3.0, nb: float = 40.0) -> torch.Tensor:
    """y [B, L] → [B] dB(amline の微分可能な版): 帯の瞬時パワー(平均を除く)のスペクトルで、フレーム周期の倍音 ±on Hz の和と ±nb Hz の近傍の平均の比。本物 ≈ 0。"""
    L = y.shape[-1]
    y = y.float()
    Y = torch.fft.rfft(y)
    f = torch.fft.rfftfreq(L, 1.0 / sr).to(y.device)
    out = []
    for lo, hi in BANDS:
        yb = torch.fft.irfft(Y * ((f >= lo) & (f < hi)).to(Y.dtype), L)
        P = yb * yb
        S = torch.fft.rfft(P - P.mean(-1, keepdim=True)).abs() ** 2
        pk, bg = 0.0, 0.0
        for h in harm:
            m_on = (f >= h - on) & (f <= h + on)
            m_nb = (f >= h - nb) & (f <= h + nb) & ~m_on
            pk = pk + S[:, m_on].sum(-1)
            bg = bg + S[:, m_nb].mean(-1) * m_on.sum()
        out.append(10 * torch.log10(pk / (bg + 1e-20) + 1e-12))
    return torch.stack(out, -1).mean(-1)
