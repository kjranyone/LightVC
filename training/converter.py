"""
LightVC-X Converter Models (PyTorch)

Mirrors the Rust `lightvc_core::converter` so exported safetensors weights
load directly into the Candle implementation.

Three model variants:
  - Converter      : residual-prediction converter (Phase 1 baseline, warm-start)
  - FlowConverter  : mean-flow matching converter (Phase C, the core model)
  - shared modules: Snake1d, CausalConv1d, CausalResBlock, FiLM, SpeakerEncoder,
                    TimbreTokenBank, CrossAttnBlock, BottleneckEncoder, TimeEmbed
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ConverterConfig:
    latent_dim: int = 1024
    hidden_dim: int = 1024
    n_conv_blocks: int = 4
    speaker_embed_dim: int = 256
    n_timbre_tokens: int = 32
    n_attn_heads: int = 8
    enable_timbre: bool = False
    # Flow matching additions
    bottleneck_dim: int = 256
    time_embed_dim: int = 128


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


class Snake1d(nn.Module):
    """Snake activation matching DAC internals."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / (self.alpha + 1e-9)) * torch.sin(self.alpha * x).pow(2)


class CausalConv1d(nn.Module):
    """Causal Conv1d with optional depthwise-separable mode.

    Training uses groups=1 (standard conv) for XPU compatibility.
    Set depthwise=True for lightweight DSConv (groups=in_ch).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int = 1,
        depthwise: bool = False,
    ):
        super().__init__()
        self.depthwise = depthwise
        self.pad = (kernel_size - 1) * dilation
        if depthwise:
            self.depthwise_conv = nn.Conv1d(
                in_ch, in_ch, kernel_size, dilation=dilation, groups=in_ch
            )
            self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        if self.depthwise:
            return self.pointwise(self.depthwise_conv(x))
        return self.conv(x)


class CausalResBlock(nn.Module):
    """Residual block with dilations [1, 3, 9] at hidden_dim.

    Projects latent_dim → hidden_dim → conv blocks → hidden_dim → latent_dim.
    This keeps standard conv (groups=1) XPU-safe while limiting params.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.proj_in = nn.Conv1d(latent_dim, hidden_dim, 1)
        self.snake1 = Snake1d(hidden_dim)
        self.c1 = CausalConv1d(hidden_dim, hidden_dim, 7, dilation=1)
        self.snake2 = Snake1d(hidden_dim)
        self.c2 = CausalConv1d(hidden_dim, hidden_dim, 7, dilation=3)
        self.snake3 = Snake1d(hidden_dim)
        self.c3 = CausalConv1d(hidden_dim, hidden_dim, 7, dilation=9)
        self.proj_out = nn.Conv1d(hidden_dim, latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.proj_in(x)
        h = self.c1(self.snake1(h))
        h = self.c2(self.snake2(h))
        h = self.c3(self.snake3(h))
        h = self.proj_out(h)
        return residual + h


class FilmCond(nn.Module):
    """FiLM: gamma * z + beta from speaker embedding."""

    def __init__(self, embed_dim: int, latent_dim: int):
        super().__init__()
        self.film = nn.Linear(embed_dim, latent_dim * 2)
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        gb = self.film(embed)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma * z + beta


class SpeakerEncoder(nn.Module):
    """Reference latent → global speaker embedding via average pooling."""

    def __init__(self, latent_dim: int, embed_dim: int):
        super().__init__()
        self.p1 = nn.Linear(latent_dim, latent_dim // 2)
        self.p2 = nn.Linear(latent_dim // 2, embed_dim)

    def forward(self, ref_latent: torch.Tensor) -> torch.Tensor:
        pooled = ref_latent.mean(dim=-1)
        h = F.gelu(self.p1(pooled))
        return self.p2(h)


class TimbreTokenBank(nn.Module):
    """Universal Timbre Token Encoder (MeanVC2-style)."""

    def __init__(self, embed_dim: int, n_tokens: int = 32):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.key_prior = nn.Parameter(torch.randn(n_tokens, embed_dim) * 0.02)
        self.val_prior = nn.Parameter(torch.randn(n_tokens, embed_dim) * 0.02)
        self.key_proj = nn.Linear(embed_dim, embed_dim * n_tokens)
        self.val_proj = nn.Linear(embed_dim, embed_dim * n_tokens)

    def forward(self, speaker_embed: torch.Tensor):
        B = speaker_embed.shape[0]
        keys = self.key_proj(speaker_embed).reshape(B, self.n_tokens, self.embed_dim)
        keys = keys + torch.tanh(self.key_prior)
        vals = self.val_proj(speaker_embed).reshape(B, self.n_tokens, self.embed_dim)
        vals = vals + torch.tanh(self.val_prior)
        return keys, vals


class CrossAttnBlock(nn.Module):
    """Cross-attention: z queries timbre tokens."""

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, z: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor
    ) -> torch.Tensor:
        B, D, T = z.shape
        z_t = z.transpose(1, 2)

        q = self.q(z_t).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(keys).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(vals).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        out = self.o(attn)

        z_norm = self.norm(z_t)
        return (z_norm + out).transpose(1, 2)


# ---------------------------------------------------------------------------
# Flow-matching specific modules
# ---------------------------------------------------------------------------


class BottleneckEncoder(nn.Module):
    """Content bottleneck: force speaker info out via channel reduction.

    This is the AutoVC trick (Paradigm 2): a too-narrow content code cannot
    encode speaker identity, so the decoder must take speaker from the
    reference encoder.
    """

    def __init__(self, latent_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = CausalConv1d(latent_dim, bottleneck_dim, 1)
        self.act = Snake1d(bottleneck_dim)
        self.up = CausalConv1d(bottleneck_dim, latent_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim, T] → content_code: [B, latent_dim, T]"""
        c = self.down(z)
        c = self.act(c)
        return self.up(c)


class TimeEmbed(nn.Module):
    """Sinusoidal time embedding for flow-matching timestep t."""

    def __init__(self, embed_dim: int):
        super().__init__()
        half = embed_dim // 2
        self.freqs = nn.Parameter(
            1.0 / (10000 ** (torch.arange(0, half).float() / half)), requires_grad=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] in [0,1] → embed: [B, embed_dim]"""
        args = t[:, None] * self.freqs[None, :] * 2 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Model 1: Residual-prediction Converter (Phase 1 / warm-start)
# ---------------------------------------------------------------------------


class Converter(nn.Module):
    """Residual-prediction converter for warm-start.

    Learns z_src + ref → z_src + Δz. Used in Phase B (bottleneck autoencoder)
    and as initialization for the flow converter.
    """

    def __init__(self, config: ConverterConfig):
        super().__init__()
        self.config = config
        D = config.latent_dim
        E = config.speaker_embed_dim

        self.bottleneck = BottleneckEncoder(D, config.bottleneck_dim)
        self.film = FilmCond(E, D)
        self.speaker_encoder = SpeakerEncoder(D, E)

        self.blocks = nn.ModuleList(
            [CausalResBlock(D, config.hidden_dim) for _ in range(config.n_conv_blocks)]
        )
        self.out_proj = CausalConv1d(D, D, 1)

        if config.enable_timbre:
            self.timbre = TimbreTokenBank(E, config.n_timbre_tokens)
            self.xattn = nn.ModuleList(
                [
                    CrossAttnBlock(D, config.n_attn_heads)
                    for _ in range(config.n_conv_blocks)
                ]
            )
        else:
            self.timbre = None
            self.xattn = None

    def forward(
        self, src_latent: torch.Tensor, ref_latent: torch.Tensor
    ) -> torch.Tensor:
        speaker_embed = self.speaker_encoder(ref_latent)

        content = self.bottleneck(src_latent)
        z = self.film(content, speaker_embed)

        timbre = None
        if self.timbre is not None:
            timbre = self.timbre(speaker_embed)

        for i, block in enumerate(self.blocks):
            z = block(z)
            if timbre is not None and self.xattn is not None:
                keys, vals = timbre
                z = self.xattn[i](z, keys, vals)

        delta = self.out_proj(z)
        return src_latent + delta

    def speaker_embedding(self, ref_latent: torch.Tensor) -> torch.Tensor:
        return self.speaker_encoder(ref_latent)

    def content_code(self, src_latent: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(src_latent)


# ---------------------------------------------------------------------------
# Model 2: Mean-Flow Converter (Phase C, the core)
# ---------------------------------------------------------------------------


class FlowConverter(nn.Module):
    """Mean-flow matching converter.

    Predicts the velocity field v(z_t, t | content, speaker) that transports
    z_0 (source) to z_1 (target speaker). At inference, a single forward pass
    with t=1 gives the mean velocity → one-step conversion.

    Training target:
        z_t = (1-t)*z_0 + t*z_tgt          # linear interpolation
        v_target = z_tgt - z_0              # constant velocity (linear flow)
        loss = MSE(v_pred(z_t, t, c, s), v_target)

    Inference (1-step):
        z_converted = z_0 + v_pred(z_0, t=1, c, s)
    """

    def __init__(self, config: ConverterConfig):
        super().__init__()
        self.config = config
        D = config.latent_dim
        E = config.speaker_embed_dim

        self.bottleneck = BottleneckEncoder(D, config.bottleneck_dim)
        self.speaker_encoder = SpeakerEncoder(D, E)
        self.time_embed = TimeEmbed(config.time_embed_dim)

        # Time + speaker conditioning MLP → FiLM parameters
        self.cond_mlp = nn.Sequential(
            nn.Linear(E + config.time_embed_dim, D),
            nn.GELU(),
            nn.Linear(D, D * 2),
        )

        self.blocks = nn.ModuleList(
            [CausalResBlock(D, config.hidden_dim) for _ in range(config.n_conv_blocks)]
        )
        self.vel_proj = CausalConv1d(D, D, 1)

        # Zero-init final projection so the model starts as identity
        if hasattr(self.vel_proj, "conv"):
            nn.init.zeros_(self.vel_proj.conv.weight)  # type: ignore
            nn.init.zeros_(self.vel_proj.conv.bias)  # type: ignore
        elif hasattr(self.vel_proj, "pointwise"):
            nn.init.zeros_(self.vel_proj.pointwise.weight)  # type: ignore
            nn.init.zeros_(self.vel_proj.pointwise.bias)  # type: ignore

        if config.enable_timbre:
            self.timbre = TimbreTokenBank(E, config.n_timbre_tokens)
            self.xattn = nn.ModuleList(
                [
                    CrossAttnBlock(D, config.n_attn_heads)
                    for _ in range(config.n_conv_blocks)
                ]
            )
        else:
            self.timbre = None
            self.xattn = None

    def _compute_conditioning(
        self, ref_latent: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Returns FiLM parameters (gamma, beta) [B, 2*latent_dim]."""
        speaker_embed = self.speaker_encoder(ref_latent)  # [B, E]
        time_embed = self.time_embed(t)  # [B, time_embed_dim]
        cond = torch.cat([speaker_embed, time_embed], dim=-1)
        return self.cond_mlp(cond)

    def forward_velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        ref_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity field. Used during training.

        Args:
            z_t: [B, latent_dim, T] interpolated latent at time t
            t: [B] timestep in [0, 1]
            ref_latent: [B, latent_dim, T_ref] target speaker reference
        Returns:
            v_pred: [B, latent_dim, T] predicted velocity
        """
        # Content code from z_t (speaker-invariant due to bottleneck)
        content = self.bottleneck(z_t)

        # Conditioning
        cond = self._compute_conditioning(ref_latent, t)
        gamma, beta = cond.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)  # [B, D, 1]
        beta = beta.unsqueeze(-1)
        z = gamma * content + beta

        timbre = None
        if self.timbre is not None:
            speaker_embed = self.speaker_encoder(ref_latent)
            timbre = self.timbre(speaker_embed)

        for i, block in enumerate(self.blocks):
            z = block(z)
            if timbre is not None and self.xattn is not None:
                keys, vals = timbre
                z = self.xattn[i](z, keys, vals)

        return self.vel_proj(z)

    @torch.no_grad()
    def convert(
        self,
        z_src: torch.Tensor,
        ref_latent: torch.Tensor,
    ) -> torch.Tensor:
        """One-step inference (mean-flow, 1-NFE).

        z_converted = z_src + v_pred(z_src, t=1, ref)

        Accepts both batched [B, D, T] and unbatched [D, T] inputs.
        """
        was_unbatched = z_src.ndim == 2
        if was_unbatched:
            z_src = z_src.unsqueeze(0)
            ref_latent = ref_latent.unsqueeze(0)

        B = z_src.shape[0]
        t = torch.ones(B, device=z_src.device)
        v = self.forward_velocity(z_src, t, ref_latent)
        result = z_src + v

        if was_unbatched:
            result = result.squeeze(0)
        return result

    def speaker_embedding(self, ref_latent: torch.Tensor) -> torch.Tensor:
        return self.speaker_encoder(ref_latent)
