"""paper_snv §5 M28(rev42): 推論の E = 登録の統計で白くした演奏(P8 の train_snv_p3.PlayW.white・128 次元)・有声・明示の演技 5(音量 c0 と周期性 4 帯を登録の平均と標準偏差で標準化)→ [内容の軌跡, 明示の演技]。
操作者の参照を読む注意と構造の条件は持たない。声ごとの量は登録の統計(CPC の文脈の平均と 2 次のモーメント・c0 と周期性の平均と標準偏差)だけ。CPC と PCA は凍結。
明示の演技を入力に足す理由: CPC の入力は c0 の差分だけで、音量の絶対値(当てる先の 1 次元)を持たない(検査 I)。
呼び方は train_snv_p11.EncS と同じ: structure(rx, rmask) → 登録の統計・forward(x, s) → (平均, 対数の分散)。x = P11.xin(y, v) の 30 次元。
    cd training && uv run --no-sync python snv_encw.py   # 単体の検査(因果性・統計が登録だけで決まる)
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import train_snv_p1 as T
import train_snv_p3 as P3

ROOT = Path(__file__).resolve().parent.parent


class EncW(nn.Module):
    def __init__(self, cpc: Path, pca: Path, ch: int = 256) -> None:
        super().__init__()
        self.pl = P3.PlayW(cpc, pca)
        self.pl.proj = nn.Identity()
        self.pl.requires_grad_(False)
        self.inp = nn.Conv1d(self.pl.P.shape[1] * len(self.pl.lags) + 1 + 5, ch, 1)
        self.blocks = nn.ModuleList([T.GBlock(ch, d) for d in (1, 2, 4, 8, 16, 32) * 2])
        self.head = nn.Conv1d(ch, 2 * T.D_A, 1)

    def train(self, mode: bool = True) -> "EncW":
        super().train(mode)
        self.pl.eval()
        return self

    @torch.no_grad()
    def structure(self, rx: torch.Tensor, rmask: torch.Tensor) -> dict:
        ry, rv = rx[:, :T.D_Y], rx[:, T.D_Y]
        mu, W = self.pl.mean_of(ry, rv, rmask)
        w = rmask.float()
        n = w.sum(-1).clamp(min=1.0)
        m0 = (ry[:, 0] * w).sum(-1) / n
        s0 = ((((ry[:, 0] - m0[:, None]) ** 2) * w).sum(-1) / n).sqrt() + 1e-3
        wv = rv * w
        nv = wv.sum(-1).clamp(min=1.0)[:, None]
        mp = (ry[:, 25:29] * wv[:, None]).sum(-1) / nv
        sp = ((((ry[:, 25:29] - mp[..., None]) ** 2) * wv[:, None]).sum(-1) / nv).sqrt() + 1e-3
        return {"mu": mu, "W": W, "m0": m0, "s0": s0, "mp": mp, "sp": sp}

    def forward(self, x: torch.Tensor, s: dict) -> tuple[torch.Tensor, torch.Tensor]:
        v = x[:, T.D_Y]
        with torch.no_grad():
            w = self.pl.white(x[:, :T.D_Y], v, (s["mu"], s["W"]))
            act = torch.cat([((x[:, 0] - s["m0"][:, None]) / s["s0"][:, None])[:, None], (x[:, 25:29] - s["mp"][..., None]) / s["sp"][..., None] * v[:, None]], 1)
        h = self.inp(torch.cat([w, v[:, None], act], 1))
        for blk in self.blocks:
            h = blk(h)
        mu, lv = self.head(h).chunk(2, 1)
        return mu, lv.clamp(-10.0, 4.0)


def unit_tests(cpc: Path, pca: Path) -> None:
    torch.manual_seed(0)
    E = EncW(cpc, pca).eval()
    x = torch.cat([torch.randn(2, T.D_Y, 400), torch.ones(2, 1, 400)], 1)
    rx, rm = torch.cat([torch.randn(2, T.D_Y, 600), torch.ones(2, 1, 600)], 1), torch.ones(2, 600, dtype=torch.bool)
    s = E.structure(rx, rm)
    a = E(x, s)[0]
    x2 = x.clone()
    x2[:, :, 300:] = torch.randn_like(x2[:, :, 300:])
    x2[:, T.D_Y, 300:] = 1.0
    assert torch.allclose(a[:, :, :300], E(x2, s)[0][:, :, :300], atol=1e-5), "EncW が未来に依る"
    assert a.shape == (2, T.D_A, 400)
    assert not any(p.requires_grad for p in E.pl.parameters()) and sum(p.numel() for p in E.parameters() if p.requires_grad) > 0
    print("EncW tests ok(因果性・凍結した CPC)", sum(p.numel() for p in E.parameters() if p.requires_grad), "params", flush=True)


if __name__ == "__main__":
    unit_tests(ROOT / "results/snv_p2b_cpc/last.pt", ROOT / "results/snv_p8b/pca_c_P2b.npy")
