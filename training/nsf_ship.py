"""出荷サイズ信号路NSF decoder v3: 低レート信号路+最終段のみ48k。

s3(全段48k=14.29ms/frame=予算4倍)の教訓を設計に組み込む:
- キャリアと信号スタックは 6000fps(hop8・Nyquist 3kHz=h4@750f0まで)で処理
- 48kは最後の ×8 アップサンプル段のみ(c32最終段相当の軽さ)
- z(cond)は100fpsで畳み込み+テント補間で6000fpsへ=マスク帯域≲50Hz
  (周波数変換チート不可・s2bの教訓)
- pitch権威は構造的: キャリア→LTI+pointwise→出力

0-GPU RTFプローブ必須(学習前に判定):
    CUDA_VISIBLE_DEVICES= uv run python nsf_ship.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from causal_codec import HOP_LENGTH, ResUnit, SnakeBeta, causal_conv

N_HARM = 4
SIG_RATE_HOP = 8          # 6000 fps
SIG_CH = (48, 64, 64)
UP_CH = 32


def excitation_lr(f0: torch.Tensor, n_lr: int, n_harm: int = N_HARM,
                  hop: int = SIG_RATE_HOP) -> torch.Tensor:
    """f0 [B,T]@100fps -> キャリア [B, 2+n_harm, n_lr] @6000fps。"""
    B, T = f0.shape
    vuv = (f0 > 0).float()
    t = (torch.arange(n_lr, device=f0.device, dtype=torch.float64) + 1.0) \
        / hop - 2.0
    i = t.floor().clamp(0, T - 2)
    fr = (t - i).clamp(0.0, 1.0).float()
    j = i.long()
    f0u = f0[:, j] * (1 - fr) + f0[:, j + 1] * fr
    vu = vuv[:, j] * (1 - fr) + vuv[:, j + 1] * fr
    f0u = f0u * vu
    phase = torch.remainder(
        2 * math.pi * torch.cumsum(f0u.double(), -1) / (48000.0 / hop),
        2 * math.pi).float()
    ks = torch.arange(1, n_harm + 1, device=f0.device,
                      dtype=torch.float32).view(-1, 1)
    h = torch.sin(ks * phase.unsqueeze(1)) * vu.unsqueeze(1)
    noise = torch.randn(B, 1, n_lr, device=f0.device)
    return torch.cat([vu.unsqueeze(1), h, noise], 1)


class NSFShip(nn.Module):
    def __init__(self, latent_dim: int = 32, cond_ch: int = 96,
                 n_harm: int = N_HARM):
        super().__init__()
        self.n_harm = n_harm
        self.cin = nn.Conv1d(latent_dim, cond_ch, 7)
        self.cres = nn.ModuleList([ResUnit(cond_ch, d) for d in (1, 3, 9)])
        sig_in = 2 + n_harm
        self.proj = nn.ModuleList()
        self.mask = nn.ModuleList()
        self.res = nn.ModuleList()
        for sc, d in zip(SIG_CH, (1, 3, 9)):
            self.proj.append(nn.Conv1d(sig_in, sc, 7))
            self.mask.append(nn.Conv1d(cond_ch, sc, 1))
            self.res.append(ResUnit(sc, d))
            sig_in = sc
        self.up = nn.ConvTranspose1d(SIG_CH[-1], UP_CH, 2 * SIG_RATE_HOP,
                                     stride=SIG_RATE_HOP)
        self.ures = nn.ModuleList([ResUnit(UP_CH, d) for d in (1, 3)])
        self.act = SnakeBeta(UP_CH)
        self.out = nn.Conv1d(UP_CH, 1, 7)

    def cond_env(self, z: torch.Tensor, n_lr: int) -> torch.Tensor:
        c = causal_conv(z, self.cin)
        for res in self.cres:
            c = res(c)
        B, C, T = c.shape
        t = (torch.arange(n_lr, device=z.device, dtype=torch.float64) + 1.0) \
            / SIG_RATE_HOP - 2.0
        i = t.floor().clamp(0, T - 2)
        fr = (t - i).clamp(0.0, 1.0).float()
        j = i.long()
        return c[:, :, j] * (1 - fr) + c[:, :, j + 1] * fr

    def forward(self, z: torch.Tensor, f0: torch.Tensor,
                gen=None) -> torch.Tensor:
        n_lr = z.shape[-1] * (HOP_LENGTH // SIG_RATE_HOP)
        c = self.cond_env(z, n_lr)
        s = excitation_lr(f0, n_lr, self.n_harm)
        for proj, mask, res in zip(self.proj, self.mask, self.res):
            w = conv1x1(c, mask)
            s = res(causal_conv(s, proj) * (1.0 + 0.9 * torch.tanh(w)))
        length = s.shape[-1] * SIG_RATE_HOP
        x = self.up(s)[..., :length]
        for res in self.ures:
            x = res(x)
        return torch.tanh(causal_conv(self.act(x), self.out))


def conv1x1(c: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    w = conv.weight[:, :, 0]
    return torch.einsum("oc,bct->bot", w, c) + conv.bias[:, None]


if __name__ == "__main__":
    import time

    torch.set_num_threads(1)
    d = NSFShip(32).eval()
    z = torch.randn(1, 32, 100)
    f0 = torch.where(torch.rand(1, 100) > 0.3, torch.rand(1, 100) * 300 + 150,
                     torch.zeros(1, 100))
    with torch.no_grad():
        y = d(z, f0)
        t0 = time.perf_counter()
        for _ in range(3):
            y = d(z, f0)
        dt = (time.perf_counter() - t0) / 3
    print(f"NSFShip: out {tuple(y.shape)}  {dt*1000:.1f} ms/100frames "
          f"= {dt*10:.2f} ms/frame (budget 3.5ms)  "
          f"params {sum(p.numel() for p in d.parameters())/1e6:.2f}M")
