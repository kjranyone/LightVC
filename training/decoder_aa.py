"""候補A(対照・BigVGAN系): c32 decoder の 48kHz 段と post_act に因果AA活性を入れた版。

潜在ABI・パラメータ名は c32 と同一(AAのFIRは非永続buffer)=c32 EMA から warm-start 可能。
因果FIRの群遅延で残差枝が遅れる分、skip も同じ整数サンプルだけ遅らせて整列する。
追加遅延(アルゴリズム遅延・先読みではない)は measure_delay() で実測し台帳に計上する。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from causal_codec import (CausalDecoder, ConvStream, UpStream, ResUnit, causal_conv,
                          CHANNELS, STRIDES, LATENT_DIM)
from causal_aa import CausalAASnake, AASnakeStream, DelayStream, delay


def act_delay(taps: int = 12) -> int:
    a = CausalAASnake(1, taps)
    with torch.no_grad():
        a.alpha.fill_(-20.0)
        a.beta.fill_(20.0)
        x = torch.zeros(1, 1, 256)
        x[..., 64] = 1.0
        y = a(x)[0, 0]
    return int(torch.argmax(y.abs()).item()) - 64


class ResUnitAA(ResUnit):
    def __init__(self, channels: int, dilation: int, d: int, taps: int = 12) -> None:
        super().__init__(channels, dilation)
        self.act1 = CausalAASnake(channels, taps)
        self.act2 = CausalAASnake(channels // 2, taps)
        self.skip_delay = 2 * d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = causal_conv(self.act1(x), self.conv1)
        y = self.conv2(self.act2(y))
        return delay(x, self.skip_delay) + y


class DecoderAA(CausalDecoder):
    def __init__(self, latent_dim: int = LATENT_DIM, channels: tuple = CHANNELS,
                 strides: tuple = STRIDES, aa_stages: tuple = (3,), taps: int = 12) -> None:
        super().__init__(latent_dim, channels, strides)
        self.taps = taps
        self.d = act_delay(taps)
        self.aa_stages = aa_stages
        for i in aa_stages:
            st = self.stages[i]
            ch = st.up.out_channels
            st.res = nn.ModuleList([ResUnitAA(ch, blk.dilation, self.d, taps) for blk in st.res])
        self.post_act = CausalAASnake(channels[0], taps)

    def added_delay_samples(self) -> int:
        n = 0
        rate = 1
        for i, st in enumerate(self.stages):
            rate *= st.stride
            if i in self.aa_stages:
                n += len(st.res) * 2 * self.d * (self.hop_length // rate)
        return n + self.d

    def stream(self) -> "DecoderAAStream":
        return DecoderAAStream(self)


class ResUnitAAStream:
    def __init__(self, blk: ResUnitAA) -> None:
        self.blk = blk
        self.a1 = AASnakeStream(blk.act1)
        self.c1 = ConvStream(blk.conv1)
        self.a2 = AASnakeStream(blk.act2)
        self.sk = DelayStream(blk.skip_delay)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(self.a1(x))
        y = self.blk.conv2(self.a2(y))
        return self.sk(x) + y


class DecoderAAStream:
    def __init__(self, dec: DecoderAA) -> None:
        from causal_codec import ResUnitStream
        self.dec = dec
        self.pre = ConvStream(dec.pre)
        self.stages = []
        for st in dec.stages:
            res = [ResUnitAAStream(b) if isinstance(b, ResUnitAA) else ResUnitStream(b)
                   for b in st.res]
            self.stages.append((UpStream(st.up), res))
        self.post_act = AASnakeStream(dec.post_act)
        self.post = ConvStream(dec.post)

    def decode_step(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z.unsqueeze(-1)
        x = self.pre(z)
        for up, res in self.stages:
            x = up(x)
            for r in res:
                x = r(x)
        return torch.tanh(self.post(self.post_act(x)))

    def decode_chunk(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.decode_step(z[..., i:i + 1]) for i in range(z.shape[-1])], dim=-1)
