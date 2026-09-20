"""信号路NSF decoder: 波形は励起キャリアのみから生成し、zは乗算変調のみ。

従来型の失敗（f0h/f0h2）: pitchを含む特徴への事後f0チャネルは加算無視・
ゲート遮断のどちらかで中和された。本設計はzから波形への加算経路を構造的に
持たない——信号 s は各段で s_proj ⊙ (1+0.9·tanh(mask(z))) を通り、
マスクは励起の位相を参照しないため、学習時の容易解は「マスク=スペクトル包絡」。
推論時にf0を掃引すれば包絡を保ったままpitchのみが構造的に追従する。

キャリア: [vuv, sin(kφ)×4, 白色noise]。encoderはS1-3凍結(z空間不変)。

    CUDA_VISIBLE_DEVICES=0 uv run python train_s2_nsf.py --tag s2_nsf
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from causal_codec import (CausalDecoder, HOP_LENGTH, ResUnit, SnakeBeta,
                          causal_conv, causal_downsample)

N_HARM = 4
STRIDES = (8, 5, 4, 3)
COND_CH = (128, 96, 64, 48)
SIG_CH = (48, 64, 96, 128)


def excitation(f0: torch.Tensor, n: int, n_harm: int = N_HARM,
               gen=None) -> torch.Tensor:
    """f0 [B,T] Hz(0=無声) -> キャリア [B, 2+n_harm, n]。因果2フレーム補間。"""
    B, T = f0.shape
    vuv = (f0 > 0).float()
    t = (torch.arange(n, device=f0.device, dtype=torch.float64) + 1.0) \
        / HOP_LENGTH - 2.0
    i = t.floor().clamp(0, T - 2)
    fr = (t - i).clamp(0.0, 1.0).float()
    j = i.long()
    f0u = f0[:, j] * (1 - fr) + f0[:, j + 1] * fr
    vu = vuv[:, j] * (1 - fr) + vuv[:, j + 1] * fr
    f0u = f0u * vu
    phase = torch.remainder(
        2 * math.pi * torch.cumsum(f0u.double(), -1) / 48000.0,
        2 * math.pi).float()
    ks = torch.arange(1, n_harm + 1, device=f0.device,
                      dtype=torch.float32).view(-1, 1)
    h = torch.sin(ks * phase.unsqueeze(1)) * vu.unsqueeze(1)
    noise = torch.randn(B, 1, n, generator=gen, device=f0.device)
    return torch.cat([vu.unsqueeze(1), h, noise], 1)


class NSFDecoder(nn.Module):
    """v2: condは100fpsでのみ畳み込み、48kへのアップサンプルは固定テント補間、

    マスク射影は1x1のみ——マスク帯域が≲50Hzに制限され、周波数変換チート
    (v1で発見: 位相含有マスク×キャリアの積でz側pitchを合成)が原理的に不可能。
    スペクトル着色はマスク後のLTI信号スタックが担う。
    """

    def __init__(self, latent_dim: int = 32, n_harm: int = N_HARM,
                 cond_ch: int = 128, sig_ch=None):
        super().__init__()
        sig_ch = sig_ch or SIG_CH
        self.n_harm = n_harm
        self.cond_ch = cond_ch
        self.cin = nn.Conv1d(latent_dim, cond_ch, 7)
        self.cres = nn.ModuleList([ResUnit(cond_ch, d) for d in (1, 3, 9, 9)])
        sig_in = 2 + n_harm
        self.proj = nn.ModuleList()
        self.mask = nn.ModuleList()
        self.res1 = nn.ModuleList()
        self.res2 = nn.ModuleList()
        for sc, d in zip(sig_ch, (1, 3, 9, 27)):
            self.proj.append(nn.Conv1d(sig_in, sc, 7))
            self.mask.append(nn.Conv1d(cond_ch, sc, 1))
            self.res1.append(ResUnit(sc, d))
            self.res2.append(ResUnit(sc, d))
            sig_in = sc
        self.act = SnakeBeta(sig_ch[-1])
        self.out = nn.Conv1d(sig_ch[-1], 1, 7)

    def cond_env(self, z: torch.Tensor, n: int) -> torch.Tensor:
        c = causal_conv(z, self.cin)
        for res in self.cres:
            c = res(c)
        B, C, T = c.shape
        t = (torch.arange(n, device=z.device, dtype=torch.float64) + 1.0) \
            / HOP_LENGTH - 2.0
        i = t.floor().clamp(0, T - 2)
        fr = (t - i).clamp(0.0, 1.0).float()
        j = i.long()
        return c[:, :, j] * (1 - fr) + c[:, :, j + 1] * fr

    def forward(self, z: torch.Tensor, f0: torch.Tensor,
                gen=None) -> torch.Tensor:
        n = z.shape[-1] * HOP_LENGTH
        c = self.cond_env(z, n)
        s = excitation(f0, n, self.n_harm, gen)
        for proj, mask, r1, r2 in zip(self.proj, self.mask,
                                      self.res1, self.res2):
            m = 1.0 + 0.9 * torch.tanh(self._pw(c, mask))
            s = r1(causal_conv(s, proj) * m)
            s = r2(s)
        return torch.tanh(causal_conv(self.act(s), self.out))

    @staticmethod
    def _pw(c: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        w = conv.weight[:, :, 0]
        return torch.einsum("oc,bct->bot", w, c) + conv.bias[:, None]


class NSFCodec(nn.Module):
    """S1-3凍結encoder + 学習NSFDecoder。"""

    def __init__(self, dev: str):
        super().__init__()
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        from causal_codec import CausalCodec
        enc_ck = torch.load(root / "results/s1_3_c32/s1_3_c32_last.pt",
                            map_location=dev)
        codec = CausalCodec(latent_dim=32,
                            channels=(32, 64, 128, 256, 512)).to(dev)
        codec.load_state_dict(enc_ck.get("ema") or enc_ck["net"])
        self.encoder = codec.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.decoder = NSFDecoder(32).to(dev)
        self._codec = [codec]

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(wav)

    def decode(self, z: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, f0)

    @staticmethod
    def load_trained(dev: str, ckpt: Path) -> "NSFCodec":
        m = NSFCodec(dev)
        m.decoder.load_state_dict(
            torch.load(ckpt, map_location=dev)["dec"])
        m.decoder.eval()
        return m
