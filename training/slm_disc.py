"""凍結した音声 SSL(既定 WavLM-base-plus)の全層の特徴の上の判別器(SLM 判別器・StyleTTS2 と同じ考え方)。目標関数 v1(凍結 WavLM-large の特徴で本物の女声と区別できないこと)を学習の側で近似する。
評価の物差し(WavLM-large・wav2vec2)とは別の SSL を使い、物差しへの合わせ込みを避ける。学習だけで使い推論の経路には入らない。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.functional as AF


class SLMDisc(nn.Module):
    def __init__(self, name: str = "microsoft/wavlm-base-plus", ch: int = 256):
        super().__init__()
        from transformers import WavLMModel
        self.ssl = WavLMModel.from_pretrained(name).eval()
        for p in self.ssl.parameters():
            p.requires_grad_(False)
        L, C = self.ssl.config.num_hidden_layers + 1, self.ssl.config.hidden_size
        wn = nn.utils.parametrizations.weight_norm
        self.head = nn.Sequential(wn(nn.Conv1d(L * C, ch, 1)), nn.LeakyReLU(0.2), wn(nn.Conv1d(ch, ch, 5, padding=2)), nn.LeakyReLU(0.2),
                                  wn(nn.Conv1d(ch, ch, 5, padding=2)), nn.LeakyReLU(0.2), wn(nn.Conv1d(ch, 1, 3, padding=1)))

    def train(self, mode: bool = True):
        super().train(mode)
        self.ssl.eval()
        return self

    def feats(self, y48: torch.Tensor) -> torch.Tensor:
        y = AF.resample(y48.float(), 48000, 16000)
        y = (y - y.mean(-1, keepdim=True)) / (y.std(-1, keepdim=True) + 1e-7)
        hs = self.ssl(y, output_hidden_states=True).hidden_states
        return torch.stack(hs, 1).permute(0, 1, 3, 2).flatten(1, 2)

    def forward(self, y48: torch.Tensor) -> torch.Tensor:
        return self.head(self.feats(y48))

    def head_params(self) -> list:
        return list(self.head.parameters())
