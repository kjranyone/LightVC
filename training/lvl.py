"""F2: 因果なレベル(c0)推定器(current/converter.md §3c)。入力 = 因果 log-mel(C1 と同じ MelFront・(mel − log 1e-5)/10)・出力 = 出力部の条件の c0 の行((c0 − C0_SIL)/10・CheapTrick の分析・[.25,.5,.25] 平滑の値)。
フレーム t は (t+1)·240 までの音声だけを見る(先読み 0)。製品の条件は変換器が出すので、CheapTrick(中心窓・先読みあり)は使えない = F2 が代わりに c0 を作る。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Blk(nn.Module):
    def __init__(self, ch: int, d: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, dilation=d)
        self.c2 = nn.Conv1d(ch, ch, 1)
        self.lp = 2 * d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(F.leaky_relu(self.c1(F.pad(F.leaky_relu(x, 0.1), (self.lp, 0))), 0.1))


class Lvl(nn.Module):
    def __init__(self, ch: int = 64, dils: tuple = (1, 2, 4, 8, 16)):
        super().__init__()
        self.cfg = {"ch": ch, "dils": list(dils)}
        self.inp = nn.Conv1d(128, ch, 1)
        self.blks = nn.ModuleList(Blk(ch, d) for d in dils)
        self.out = nn.Conv1d(ch, 1, 1)
        self.rf = 1 + sum(2 * d for d in dils)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel [B, 128, T] → c0n [B, T]((c0 − C0_SIL)/10 の推定)。lev(mel の平均 × √128)に足す残差として出す。"""
        lev = 11.3137085 * mel.mean(1)
        x = self.inp(mel)
        for b in self.blks:
            x = b(x)
        return lev - 2.7 + self.out(F.leaky_relu(x, 0.1))[:, 0]
