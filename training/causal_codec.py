from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


SAMPLE_RATE = 48_000
HOP_LENGTH = 480
LATENT_DIM = 32
STRIDES = (8, 5, 4, 3)
CHANNELS = (32, 64, 128, 256, 512)
DILATIONS = (1, 3, 9)


def causal_conv(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    left = (conv.kernel_size[0] - 1) * conv.dilation[0]
    return conv(F.pad(x, (left, 0)))


def causal_downsample(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    left = conv.kernel_size[0] - conv.stride[0]
    return conv(F.pad(x, (left, 0)))


class SnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.exp().view(1, -1, 1)
        beta = self.beta.exp().view(1, -1, 1)
        return x + torch.sin(alpha * x).square() / (beta + 1e-9)


class ResUnit(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.dilation = dilation
        self.act1 = SnakeBeta(channels)
        self.conv1 = nn.Conv1d(channels, hidden, 7, dilation=dilation)
        self.act2 = SnakeBeta(hidden)
        self.conv2 = nn.Conv1d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = causal_conv(self.act1(x), self.conv1)
        y = self.conv2(self.act2(y))
        return x + y


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.res = nn.ModuleList([ResUnit(in_channels, d) for d in DILATIONS])
        self.down = nn.Conv1d(in_channels, out_channels, 2 * stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.res:
            x = block(x)
        return causal_downsample(x, self.down)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        channels: tuple[int, ...] = CHANNELS,
        strides: tuple[int, ...] = STRIDES,
    ) -> None:
        super().__init__()
        if len(channels) != len(strides) + 1:
            raise ValueError("channels must have len(strides) + 1 entries")
        self.hop_length = int(torch.tensor(strides).prod().item())
        self.pre = nn.Conv1d(1, channels[0], 7)
        self.stages = nn.ModuleList([
            EncoderStage(cin, cout, stride)
            for cin, cout, stride in zip(channels[:-1], channels[1:], strides)
        ])
        self.out = nn.Conv1d(channels[-1], latent_dim, 3)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim != 3 or wav.shape[1] != 1:
            raise ValueError(f"expected [B,1,T], got {tuple(wav.shape)}")
        if wav.shape[-1] % self.hop_length:
            raise ValueError(f"length must be divisible by {self.hop_length}")
        x = causal_conv(wav, self.pre)
        for stage in self.stages:
            x = stage(x)
        return causal_conv(x, self.out)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.up = nn.ConvTranspose1d(in_channels, out_channels, 2 * stride, stride=stride)
        self.res = nn.ModuleList([ResUnit(out_channels, d) for d in DILATIONS])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1] * self.stride
        x = self.up(x)[..., :length]
        for block in self.res:
            x = block(x)
        return x


class CausalDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        channels: tuple[int, ...] = CHANNELS,
        strides: tuple[int, ...] = STRIDES,
    ) -> None:
        super().__init__()
        if len(channels) != len(strides) + 1:
            raise ValueError("channels must have len(strides) + 1 entries")
        self.hop_length = int(torch.tensor(strides).prod().item())
        self.latent_dim = latent_dim
        self.pre = nn.Conv1d(latent_dim, channels[-1], 7)
        rev_channels = tuple(reversed(channels))
        rev_strides = tuple(reversed(strides))
        self.stages = nn.ModuleList([
            DecoderStage(cin, cout, stride)
            for cin, cout, stride in zip(rev_channels[:-1], rev_channels[1:], rev_strides)
        ])
        self.post_act = SnakeBeta(channels[0])
        self.post = nn.Conv1d(channels[0], 1, 7)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or z.shape[1] != self.latent_dim:
            raise ValueError(f"expected [B,{self.latent_dim},F], got {tuple(z.shape)}")
        x = causal_conv(z, self.pre)
        for stage in self.stages:
            x = stage(x)
        return torch.tanh(causal_conv(self.post_act(x), self.post))

    def stream(self) -> CausalDecoderStream:
        return CausalDecoderStream(self)


class CausalCodec(nn.Module):
    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        channels: tuple[int, ...] = CHANNELS,
        strides: tuple[int, ...] = STRIDES,
    ) -> None:
        super().__init__()
        self.encoder = CausalEncoder(latent_dim, channels, strides)
        self.decoder = CausalDecoder(latent_dim, channels, strides)
        if self.encoder.hop_length != HOP_LENGTH and strides == STRIDES:
            raise RuntimeError("default strides must produce hop 480")

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.encoder(wav)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(wav)
        return self.decode(z), z


class ConvStream:
    def __init__(self, conv: nn.Conv1d) -> None:
        if conv.stride[0] != 1:
            raise ValueError("ConvStream only supports stride 1")
        self.conv = conv
        self.left = (conv.kernel_size[0] - 1) * conv.dilation[0]
        self.state: torch.Tensor | None = None

    def reset(self) -> None:
        self.state = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.left == 0:
            return self.conv(x)
        if self.state is None:
            self.state = x.new_zeros(x.shape[0], x.shape[1], self.left)
        if self.state.shape[:2] != x.shape[:2]:
            raise ValueError("stream batch/channel shape changed without reset")
        joined = torch.cat((self.state, x), dim=-1)
        y = self.conv(joined)
        self.state = joined[..., -self.left:]
        return y


class ResUnitStream:
    def __init__(self, block: ResUnit) -> None:
        self.block = block
        self.conv1 = ConvStream(block.conv1)

    def reset(self) -> None:
        self.conv1.reset()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.block.act1(x))
        y = self.block.conv2(self.block.act2(y))
        return x + y


class UpStream:
    def __init__(self, up: nn.ConvTranspose1d) -> None:
        self.up = up
        self.stride = up.stride[0]
        if up.kernel_size[0] != 2 * self.stride:
            raise ValueError("streaming upsample requires kernel=2*stride")
        self.tail: torch.Tensor | None = None

    def reset(self) -> None:
        self.tail = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raw = F.conv_transpose1d(x, self.up.weight, None, stride=self.stride)
        if self.tail is None:
            self.tail = raw.new_zeros(raw.shape[0], raw.shape[1], self.stride)
        if self.tail.shape[:2] != raw.shape[:2]:
            raise ValueError("stream batch/channel shape changed without reset")
        raw[..., :self.stride] = raw[..., :self.stride] + self.tail
        emit = raw[..., :x.shape[-1] * self.stride]
        self.tail = raw[..., -self.stride:]
        if self.up.bias is not None:
            emit = emit + self.up.bias.view(1, -1, 1)
        return emit


class DecoderStageStream:
    def __init__(self, stage: DecoderStage) -> None:
        self.up = UpStream(stage.up)
        self.res = [ResUnitStream(block) for block in stage.res]

    def reset(self) -> None:
        self.up.reset()
        for block in self.res:
            block.reset()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        for block in self.res:
            x = block(x)
        return x


class CausalDecoderStream:
    def __init__(self, decoder: CausalDecoder) -> None:
        self.decoder = decoder
        self.pre = ConvStream(decoder.pre)
        self.stages = [DecoderStageStream(stage) for stage in decoder.stages]
        self.post = ConvStream(decoder.post)

    def reset(self) -> None:
        self.pre.reset()
        for stage in self.stages:
            stage.reset()
        self.post.reset()

    def decode_step(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z.unsqueeze(-1)
        if z.ndim != 3 or z.shape[1] != self.decoder.latent_dim or z.shape[-1] != 1:
            raise ValueError(f"expected [B,{self.decoder.latent_dim},1], got {tuple(z.shape)}")
        x = self.pre(z)
        for stage in self.stages:
            x = stage(x)
        y = torch.tanh(self.post(self.decoder.post_act(x)))
        if y.shape[-1] != self.decoder.hop_length:
            raise RuntimeError(f"decode_step emitted {y.shape[-1]} samples")
        return y

    def decode_chunk(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.decode_step(z[..., i:i + 1]) for i in range(z.shape[-1])], dim=-1)


@dataclass(frozen=True)
class ArchitectureStats:
    encoder_parameters: int
    decoder_parameters: int
    total_parameters: int
    decoder_macs_per_second: int


def architecture_stats(codec: CausalCodec) -> ArchitectureStats:
    encoder_parameters = sum(p.numel() for p in codec.encoder.parameters())
    decoder_parameters = sum(p.numel() for p in codec.decoder.parameters())
    rates = [SAMPLE_RATE // codec.decoder.hop_length]
    macs = codec.decoder.pre.in_channels * codec.decoder.pre.out_channels
    macs *= codec.decoder.pre.kernel_size[0] * rates[0]
    for stage in codec.decoder.stages:
        input_rate = rates[-1]
        output_rate = input_rate * stage.stride
        rates.append(output_rate)
        macs += (
            stage.up.in_channels
            * stage.up.out_channels
            * stage.up.kernel_size[0]
            * input_rate
        )
        for block in stage.res:
            macs += (
                block.conv1.in_channels
                * block.conv1.out_channels
                * block.conv1.kernel_size[0]
                * output_rate
            )
            macs += block.conv2.in_channels * block.conv2.out_channels * output_rate
    macs += (
        codec.decoder.post.in_channels
        * codec.decoder.post.out_channels
        * codec.decoder.post.kernel_size[0]
        * SAMPLE_RATE
    )
    return ArchitectureStats(
        encoder_parameters=encoder_parameters,
        decoder_parameters=decoder_parameters,
        total_parameters=encoder_parameters + decoder_parameters,
        decoder_macs_per_second=macs,
    )
