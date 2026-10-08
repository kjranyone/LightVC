"""D1: 潜在AR生成器(AR-CFM) — フレーム毎補間CFM + 履歴条件(current/d1_latentar.md rev2)。

構成: enc(履歴z_{<t}+cond → 文脈特徴・因果conv) + head(z_interp+文脈 → 速度場・MLP)。
訓練はT方向完全並列(履歴=GTのteacher-forcing)。推論はフレーム毎:
  feat_t = enc(窓[16履歴+1現在]) を1回 + headをEuler K=8回。
z0定義: 白色 N(0,I)(s7のρ流儀のρ=0縮退)。no_history=並列CFM対照腕。

    CUDA_VISIBLE_DEVICES= uv run python d1_model.py   # RTFプローブ(CPU 1thread・最悪値)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

COND_DIM = 770          # content768 + lf0 + en (主腕=mel80なし)
COND_DIM_M80 = 850      # 比較腕(mel80あり)
Z_DIM = 32
HIST = 16               # 履歴受容(フレーム=160ms)
DILS = (1, 2, 4, 8, 16, 32)


class FLayer(nn.Module):
    def __init__(self, ch: int, dil: int):
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, 3, dilation=dil)
        self.pw = nn.Conv1d(ch, ch, 1)
        self.dil = dil

    def forward(self, h):
        return h + self.pw(F.leaky_relu(self.dw(F.pad(h, (2 * self.dil, 0))), 0.01))


class D1AR(nn.Module):
    def __init__(self, dim: int = 256, layers: int = 12, cond_dim: int = COND_DIM,
                 no_history: bool = False):
        super().__init__()
        self.no_history = no_history
        self.dim = dim
        self.enc_in = nn.Conv1d(Z_DIM + cond_dim, dim, 3)
        self.blocks = nn.ModuleList(
            [FLayer(dim, DILS[i % len(DILS)]) for i in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(Z_DIM + dim + 1, dim), nn.LeakyReLU(0.01),
            nn.Linear(dim, Z_DIM))
        self.rf = 3 + 2 * sum(DILS[i % len(DILS)] for i in range(layers))

    def feat(self, z_hist: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """z_hist [B,32,T](位置t=z_{t-1}の右シフト済み), cond [B,C,T] → [B,dim,T]。"""
        if self.no_history:
            z_hist = torch.zeros_like(z_hist)
        h = self.enc_in(F.pad(torch.cat([z_hist, cond], 1), (2, 0)))
        for b in self.blocks:
            h = b(h)
        return h

    def vel(self, z_interp: torch.Tensor, ft: torch.Tensor,
            t: torch.Tensor) -> torch.Tensor:
        """z_interp [B,32], ft [B,dim], t [B] → 速度 [B,32]。"""
        x = torch.cat([z_interp, ft, t[:, None]], 1)
        return self.head(x)

    def forward(self, z_hist, cond, z_interp, t):
        """訓練用一括: z_interp [B,32,T] → 速度 [B,32,T](全位置並列)。"""
        ft = self.feat(z_hist, cond)
        B, _, T = z_interp.shape
        x = torch.cat([z_interp, ft, t[:, None, None].expand(-1, 1, T)], 1)
        return self.head(x.transpose(1, 2)).transpose(1, 2)


def shift_right(z: torch.Tensor) -> torch.Tensor:
    """z [B,32,T] → 位置tがz_{t-1}を参照(先頭ゼロ)。
    bug6修正: [:( :, 1:]は恒等関数だった(先頭ゼロを捨てて元を返す)。"""
    return F.pad(z, (1, 0))[:, :, :-1]


def _selfcheck():
    z = torch.randn(2, Z_DIM, 10)
    sr = shift_right(z)
    assert sr.shape == z.shape
    assert float(sr[:, :, 0].abs().max()) == 0.0, "先頭がゼロでない"
    assert torch.equal(sr[:, :, 1:], z[:, :, :-1]), "位置tがz_{t-1}でない"
    assert not torch.equal(sr, z), "恒等関数(bug6回帰)"
    print("shift_right unit check: ok")


def sample_frame_ar(net: D1AR, cond: torch.Tensor, K: int = 8, seed: int = 0,
                    z0_rho: float = 0.0):
    """逐次ARサンプリング(推論・未来不変)。cond [1,C,T] → z [1,32,T]。
    訓練のfeat入力は位置jが(z_{j-1}, cond_j) — バッファ末尾=直近サンプルz_{i-1}
    (bug6族のoff-by-one修正: 現在スロットにゼロを置かない)。
    z0_rho>0でz0をAR(1)時間相関(G1-FAILフォールバック・訓練と同一構造)。"""
    B, C, T = cond.shape
    dev = cond.device
    RF = net.rf
    torch.manual_seed(seed)
    e = torch.randn(B, Z_DIM, T, device=dev)
    z0 = e.clone()
    for j in range(1, T):
        z0[:, :, j] = (z0_rho * z0[:, :, j - 1]
                       + math.sqrt(1 - z0_rho ** 2) * e[:, :, j])
    out = torch.zeros(B, Z_DIM, T, device=dev)
    zbuf = torch.zeros(B, Z_DIM, RF + 1, device=dev)   # 末尾=z_{i-1}
    cbuf = torch.zeros(B, C, RF + 1, device=dev)       # 末尾=cond_i
    with torch.no_grad():
        for i in range(T):
            cbuf = torch.cat([cbuf[:, :, 1:], cond[:, :, i:i + 1]], -1)
            ft = net.feat(zbuf, cbuf)[:, :, -1]
            z = z0[:, :, i]
            for k in range(K):
                tk = torch.full((B,), k / K, device=dev)
                z = z + net.vel(z, ft, tk) / K
            out[:, :, i] = z
            zbuf = torch.cat([zbuf[:, :, 1:], z[:, :, None]], -1)
    return out


if __name__ == "__main__":
    import time
    _selfcheck()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    for dim in (256, 192, 128):
        net = D1AR(dim=dim).eval()
        n_par = sum(p.numel() for p in net.parameters()) / 1e6
        cond = torch.randn(1, COND_DIM, 100)
        with torch.no_grad():
            _ = sample_frame_ar(net, cond, K=8)
            runs = []
            for _ in range(5):
                t0 = time.perf_counter()
                _ = sample_frame_ar(net, cond[:, :, :100], K=8)
                runs.append((time.perf_counter() - t0) / 100 * 1000)
        print(f"D1AR dim{dim}: {n_par:.2f}M  frame-AR K=8  "
              f"mean {sum(runs)/5:.2f} / worst {max(runs):.2f} ms/frame (CPU 1thread)")
