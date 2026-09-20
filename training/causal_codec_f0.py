"""Y-S1 decoderへのNSF式f0励起ヘッド。trunkはS1-3重みで凍結、ヘッドのみ学習。

励起 = vuvゲートされた調和正弦 sin(k·phi)（phi は条件f0の因果積分位相）。
最終波形は (trunk特徴, 励起) から畳み込みで合成するため、出力ピッチは
構造的に条件f0に従う（学習で獲得する権威ではない）。
f0入力は100fps（latent格子と同一）でHz・0=無声。推論時にf0を掃引すれば
ピッチが追従し、trunk特徴（包絡・質感）は不変のまま。

    CUDA_VISIBLE_DEVICES=0 uv run python train_s1_f0head.py --tag s1_f0h
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from causal_codec import (CausalDecoder, HOP_LENGTH, ResUnit, SnakeBeta,
                          causal_conv)

N_HARM = 4
HIDDEN = 96


class DecoderF0Head(nn.Module):
    def __init__(self, feat_ch: int = 32, hidden: int = HIDDEN,
                 n_harm: int = N_HARM):
        super().__init__()
        self.n_harm = n_harm
        self.block1 = nn.Conv1d(feat_ch + 1 + n_harm, hidden, 7)
        self.act1 = SnakeBeta(hidden)
        self.block2 = ResUnit(hidden, 3)
        self.block3 = ResUnit(hidden, 9)
        self.act2 = SnakeBeta(hidden)
        self.out = nn.Conv1d(hidden, 1, 7)

    def excitation(self, f0: torch.Tensor, n: int) -> torch.Tensor:
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
        ks = torch.arange(1, self.n_harm + 1, device=f0.device,
                          dtype=torch.float32).view(-1, 1)
        h = torch.sin(ks * phase.unsqueeze(1)) * vu.unsqueeze(1)
        return torch.cat([vu.unsqueeze(1), h], 1)

    def forward(self, feat: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        exc = self.excitation(f0, feat.shape[-1])
        x = torch.cat([feat, exc], 1)
        x = causal_conv(x, self.block1)
        x = self.act1(x)
        x = self.block2(x)
        x = self.block3(x)
        return torch.tanh(causal_conv(self.act2(x), self.out))


class DecoderF0FiLM(nn.Module):
    """キャリア型NSFヘッド: e = exc*(1+tanh(gamma(feat))) + tanh(beta(feat)) -> convs。

    featはeへ乗算変調としてのみ入り波形への加算経路を持たないため、
    周期性は構造的に励起(=条件f0)由来になる。無声部はbeta経路が担う。
    """

    def __init__(self, feat_ch: int = 32, hidden: int = HIDDEN,
                 n_harm: int = N_HARM):
        super().__init__()
        self.n_harm = n_harm
        self.film = nn.Conv1d(feat_ch, 2 * (1 + n_harm), 7)
        self.block1 = nn.Conv1d(1 + n_harm, hidden, 7)
        self.act1 = SnakeBeta(hidden)
        self.block2 = ResUnit(hidden, 3)
        self.block3 = ResUnit(hidden, 9)
        self.act2 = SnakeBeta(hidden)
        self.out = nn.Conv1d(hidden, 1, 7)

    def excitation(self, f0: torch.Tensor, n: int) -> torch.Tensor:
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
        ks = torch.arange(1, self.n_harm + 1, device=f0.device,
                          dtype=torch.float32).view(-1, 1)
        h = torch.sin(ks * phase.unsqueeze(1)) * vu.unsqueeze(1)
        return torch.cat([vu.unsqueeze(1), h], 1)

    def forward(self, feat: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        exc = self.excitation(f0, feat.shape[-1])
        g, b = causal_conv(feat, self.film).chunk(2, 1)
        e = exc * (1 + torch.tanh(g)) + torch.tanh(b)
        x = causal_conv(e, self.block1)
        x = self.act1(x)
        x = self.block2(x)
        x = self.block3(x)
        return torch.tanh(causal_conv(self.act2(x), self.out))


class CodecF0(nn.Module):
    """凍結CausalDecoder trunk + 学習ヘッド。encodeは親codecと共通。"""

    def __init__(self, trunk: CausalDecoder, head=None):
        super().__init__()
        self.trunk = trunk
        self.head = head or DecoderF0FiLM()
        for p in self.trunk.parameters():
            p.requires_grad_(False)

    def features(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = causal_conv(z, self.trunk.pre)
            for stage in self.trunk.stages:
                x = stage(x)
            return self.trunk.post_act(x)

    def decode(self, z: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        feat = self.features(z)
        return self.head(feat.detach(), f0)

    @staticmethod
    def from_s13(dev: str) -> "CodecF0":
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        ck = torch.load(root / "results/s1_3_c32/s1_3_c32_last.pt",
                        map_location=dev)
        trunk = CausalDecoder(32, (32, 64, 128, 256, 512), (8, 5, 4, 3)).to(dev)
        trunk.load_state_dict(
            {k[len("decoder."):]: v for k, v in (ck.get("ema") or ck["net"]).items()
             if k.startswith("decoder.")})
        trunk.eval()
        cf = CodecF0(trunk)
        cf.head.to(dev)
        return cf
