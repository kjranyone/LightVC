"""
LightVC Converter Models (PyTorch)

Mirrors the Rust `lightvc_core::converter` so exported safetensors weights
load directly into the Candle implementation.

Three model variants:
  - Converter      : residual-prediction converter (Phase 1 baseline, warm-start)
  - FlowConverter  : flow-matching converter (Phase C, the core model)
  - shared modules: Snake1d, CausalConv1d, CausalResBlock, FiLM, SpeakerEncoder,
                    TimbreTokenBank, CrossAttnBlock, BottleneckEncoder, TimeEmbed
"""

from __future__ import annotations

import enum
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
    # Phase 3: Progressive RVQ-depth factorized FM heads ([07-1]).
    # 0 = single vel_proj (disabled). 3 = coarse(1-3)/mid(4-6)/fine(7-9).
    n_depth_groups: int = 0


# ---------------------------------------------------------------------------
# Phase 4: Prosody / Rhythm factorization ([07-2])
# ---------------------------------------------------------------------------


class ProsodyMode(enum.Enum):
    """Controls how prosody (F0, energy, rhythm) is handled during conversion.

    * ``PreserveSource`` — keep source prosody, convert timbre only
    * ``Blend`` — interpolate prosody between source and target
    * ``ImitateTarget`` — replace prosody with target's (default VC behavior)
    * ``FlattenPrivacy`` — normalize prosody for anti-voice-print
    """

    PreserveSource = 0
    Blend = 1
    ImitateTarget = 2
    FlattenPrivacy = 3


def apply_prosody_mode(
    z_converted: torch.Tensor,
    z_src: torch.Tensor,
    mode: ProsodyMode,
    blend_factor: float = 0.5,
) -> torch.Tensor:
    """Post-process the converted latent to control prosody ([07-2]).

    Prosody is approximated as the per-frame energy envelope (L2 norm across
    the latent dimension). This function preserves, blends, or flattens the
    source's temporal dynamics on the converted output.

    Args:
        z_converted: [B, D, T] latent after voice conversion
        z_src: [B, D, T] original source latent (for envelope reference)
        mode: prosody handling mode
        blend_factor: 0.0 = all source, 1.0 = all target (for Blend mode)
    Returns:
        [B, D, T] prosody-adjusted latent
    """
    if mode == ProsodyMode.ImitateTarget:
        return z_converted

    # Per-frame energy envelope: [B, 1, T]
    src_env = z_src.norm(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1, T]
    conv_env = z_converted.norm(dim=1, keepdim=True).clamp(min=1e-8)

    if mode == ProsodyMode.PreserveSource:
        # Scale converted latent to match source's energy envelope per frame.
        scale = src_env / conv_env
        return z_converted * scale

    if mode == ProsodyMode.Blend:
        # Interpolate energy envelope between source and converted.
        target_env = (1.0 - blend_factor) * src_env + blend_factor * conv_env
        scale = target_env / conv_env
        return z_converted * scale

    if mode == ProsodyMode.FlattenPrivacy:
        # Normalize energy envelope to its mean (remove prosody dynamics).
        mean_env = conv_env.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = mean_env / conv_env
        return z_converted * scale

    return z_converted


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
    """Causal Conv1d (standard conv, groups=1) for XPU compatibility.

    The depthwise-separable variant was removed: CausalResBlock uses
    groups=1 throughout, and the depthwise path was dead code that
    fails on XPU backward (see AGENTS.md Known Issues).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
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
    """Reference latent → global speaker embedding via statistical pooling.

    Uses mean + std pooling (capture both central tendency and variability,
    which carries speaker-diagnostic information). A simple mean-only pool
    collapses to a near-constant output across speakers.
    """

    def __init__(self, latent_dim: int, embed_dim: int):
        super().__init__()
        self.p1 = nn.Linear(latent_dim * 2, latent_dim // 2)
        self.p2 = nn.Linear(latent_dim // 2, embed_dim)

    def forward(self, ref_latent: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([ref_latent.mean(dim=-1), ref_latent.std(dim=-1)], dim=-1)
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
    """Cross-attention: z (latent_dim) queries timbre tokens (embed_dim)."""

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.attn_dim = n_heads * (kv_dim // n_heads)
        self.q = nn.Linear(q_dim, self.attn_dim)
        self.k = nn.Linear(kv_dim, self.attn_dim)
        self.v = nn.Linear(kv_dim, self.attn_dim)
        self.o = nn.Linear(self.attn_dim, q_dim)
        self.norm = nn.LayerNorm(q_dim)

    def forward(
        self, z: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor
    ) -> torch.Tensor:
        B, D, T = z.shape
        z_t = z.transpose(1, 2)

        q = self.q(z_t).reshape(B, T, self.n_heads, self.attn_dim // self.n_heads).transpose(1, 2)
        k = self.k(keys).reshape(B, -1, self.n_heads, self.attn_dim // self.n_heads).transpose(1, 2)
        v = self.v(vals).reshape(B, -1, self.n_heads, self.attn_dim // self.n_heads).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, T, self.attn_dim)
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
                    CrossAttnBlock(D, E, config.n_attn_heads)
                    for _ in range(config.n_conv_blocks)
                ]
            )
        else:
            self.timbre = None
            self.xattn = None

    def forward(
        self, src_latent: torch.Tensor, ref_latent: torch.Tensor
    ) -> torch.Tensor:
        was_unbatched = src_latent.ndim == 2
        if was_unbatched:
            src_latent = src_latent.unsqueeze(0)
            ref_latent = ref_latent.unsqueeze(0)

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
        result = src_latent + delta

        if was_unbatched:
            result = result.squeeze(0)
        return result

    def speaker_embedding(self, ref_latent: torch.Tensor) -> torch.Tensor:
        return self.speaker_encoder(ref_latent)

    def content_code(self, src_latent: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(src_latent)


# ---------------------------------------------------------------------------
# Model 2: Mean-Flow Converter (Phase C, the core)
# ---------------------------------------------------------------------------


class FlowConverter(nn.Module):
    """Flow-matching converter (rectified / linear flow, 1-NFE).

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

        # Velocity projection: single head or factorized heads ([07-1]).
        self.n_depth_groups = config.n_depth_groups
        if config.n_depth_groups > 0:
            # Progressive RVQ-depth factorized FM heads.
            # Group 0 (coarse, ~RVQ 1-3): content/timbre — convert aggressively
            # Group 1 (mid,   ~RVQ 4-6): spectral shape — moderate
            # Group 2 (fine,  ~RVQ 7-9): texture/noise — light or passthrough
            self.vel_heads = nn.ModuleList(
                [CausalConv1d(D, D, 1) for _ in range(config.n_depth_groups)]
            )
            for head in self.vel_heads:
                nn.init.zeros_(head.conv.weight)  # type: ignore
                nn.init.zeros_(head.conv.bias)  # type: ignore
            self.vel_proj = None
        else:
            self.vel_proj = CausalConv1d(D, D, 1)
            # Zero-init final projection so the model starts as identity
            nn.init.zeros_(self.vel_proj.conv.weight)  # type: ignore
            nn.init.zeros_(self.vel_proj.conv.bias)  # type: ignore

        if config.enable_timbre:
            self.timbre = TimbreTokenBank(E, config.n_timbre_tokens)
            self.xattn = nn.ModuleList(
                [
                    CrossAttnBlock(D, E, config.n_attn_heads)
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
        depth_strengths: tuple[float, ...] | None = None,
    ) -> torch.Tensor:
        """Predict velocity field. Used during training.

        Args:
            z_t: [B, latent_dim, T] interpolated latent at time t
            t: [B] timestep in [0, 1]
            ref_latent: [B, latent_dim, T_ref] target speaker reference
            depth_strengths: optional per-group scaling ([07-1]). None = all 1.0.
                e.g. (1.0, 0.0, 0.0) = coarse-only for low-latency mode.
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

        # Velocity prediction: single head or factorized heads ([07-1]).
        if self.vel_proj is not None:
            return self.vel_proj(z)
        # Factorized: sum of per-depth-group velocity contributions.
        if depth_strengths is None:
            depth_strengths = tuple(1.0 for _ in self.vel_heads)
        v_total = None
        for i, head in enumerate(self.vel_heads):
            s = depth_strengths[i] if i < len(depth_strengths) else 0.0
            if s == 0.0:
                continue
            v_i = head(z)
            v_total = v_i * s if v_total is None else v_total + v_i * s
        return v_total if v_total is not None else torch.zeros_like(z)

    # ------------------------------------------------------------------
    # Phase 4: Prosody / Rhythm factorization ([07-2])
    # ------------------------------------------------------------------

    @torch.no_grad()
    def convert(
        self,
        z_src: torch.Tensor,
        ref_latent: torch.Tensor,
        velocity_scale: float = 1.0,
        depth_strengths: tuple[float, ...] | None = None,
        prosody_mode: ProsodyMode = ProsodyMode.ImitateTarget,
        prosody_blend: float = 0.5,
    ) -> torch.Tensor:
        """One-step inference (1-NFE).

        z_converted = z_src + velocity_scale * v_pred(z_src, t=1, ref)

        ``velocity_scale`` (>1 amplifies speaker-translation effect, analogous
        to classifier-free guidance in diffusion models). 1.0 matches the
        training objective exactly. Mirrors the Rust ``FlowConverter::convert``
        API ([03-9]).

        ``depth_strengths`` ([07-1]): per-group scaling for factorized heads.
        None = all groups active at 1.0. Common presets:
          * (1.0, 1.0, 1.0) — full conversion (default)
          * (1.0, 0.0, 0.0) — coarse-only (lowest latency, ~RVQ 1-3)
          * (1.5, 1.5, 0.0) — privacy mode (strong timbre, skip fine detail)

        ``prosody_mode`` ([07-2]): controls F0/energy/rhythm handling.
        ``prosody_blend``: 0.0=all source, 1.0=all target (for Blend mode).

        Accepts both batched [B, D, T] and unbatched [D, T] inputs.
        """
        was_unbatched = z_src.ndim == 2
        if was_unbatched:
            z_src = z_src.unsqueeze(0)
            ref_latent = ref_latent.unsqueeze(0)

        B = z_src.shape[0]
        t = torch.ones(B, device=z_src.device)
        v = self.forward_velocity(z_src, t, ref_latent, depth_strengths=depth_strengths)
        result = z_src + velocity_scale * v

        # Apply prosody factorization ([07-2]).
        result = apply_prosody_mode(result, z_src, prosody_mode, prosody_blend)

        if was_unbatched:
            result = result.squeeze(0)
        return result

    def speaker_embedding(self, ref_latent: torch.Tensor) -> torch.Tensor:
        return self.speaker_encoder(ref_latent)

    # ------------------------------------------------------------------
    # Phase 5: Dual-path converter ([07-3])
    # ------------------------------------------------------------------

    @torch.no_grad()
    def convert_dual_path(
        self,
        z_src: torch.Tensor,
        ref_latent: torch.Tensor,
        coarse_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Dual-path conversion: fast coarse + optional refine ([07-3]).

        Returns ``(coarse_result, refine_result)``:
          * ``coarse_result`` — immediate output using only the coarse head
            (group 0). Suitable for low-latency streaming where the user
            hears this first.
          * ``refine_result`` — detail-corrected output using mid + fine
            heads (groups 1, 2) applied to ``coarse_result``. ``None`` if
            ``coarse_only=True`` or if the model has no factorized heads.

        In a streaming pipeline, ``coarse_result`` is emitted immediately and
        ``refine_result`` replaces it on the next chunk (bounded lookahead).

        Requires ``n_depth_groups >= 2`` in the config.
        """
        if self.n_depth_groups < 2 or self.vel_proj is not None:
            # No factorized heads — single-path fallback.
            full = self.convert(z_src, ref_latent)
            return full, None

        was_unbatched = z_src.ndim == 2
        if was_unbatched:
            z_src = z_src.unsqueeze(0)
            ref_latent = ref_latent.unsqueeze(0)

        B = z_src.shape[0]
        device = z_src.device
        t = torch.ones(B, device=device)

        # Fast path: coarse head only (group 0).
        coarse_strengths = tuple(
            1.0 if i == 0 else 0.0 for i in range(self.n_depth_groups)
        )
        v_coarse = self.forward_velocity(z_src, t, ref_latent, depth_strengths=coarse_strengths)
        coarse_result = z_src + v_coarse

        if coarse_only:
            result = coarse_result.squeeze(0) if was_unbatched else coarse_result
            return result, None

        # Refine path: mid + fine heads (groups 1+) applied to coarse result.
        refine_strengths = tuple(
            1.0 if i > 0 else 0.0 for i in range(self.n_depth_groups)
        )
        v_refine = self.forward_velocity(
            coarse_result, t, ref_latent, depth_strengths=refine_strengths
        )
        refine_result = coarse_result + v_refine

        if was_unbatched:
            coarse_result = coarse_result.squeeze(0)
            refine_result = refine_result.squeeze(0)

        return coarse_result, refine_result


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (GRL) for content/speaker disentanglement
# ---------------------------------------------------------------------------


class GradientReversalFunction(torch.autograd.Function):
    """Multiplies gradients by -lambda on backward.

    Used by the content MI loss ([04-4]): a lightweight speaker classifier
    sits on the content code, but its gradients are reversed so the
    BottleneckEncoder learns to *remove* speaker information — the
    VQMIVC-style mutual information regularization (MODEL_TRAINING C.4 #5).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, lambda_: float
    ) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambda_)


class ContentSpeakerAdversary(nn.Module):
    """Lightweight speaker classifier applied to the content code.

    Adversarially trained via gradient reversal: the bottleneck learns to
    produce content codes from which speaker identity cannot be recovered,
    improving disentanglement and reducing source-voice leakage in
    zero-shot conversion.

    Input: content_code [B, latent_dim, T] → speaker logits [B, n_speakers]
    """

    def __init__(self, latent_dim: int, n_speakers: int, bottleneck_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv1d(latent_dim, bottleneck_dim, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cls = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, n_speakers),
        )

    def forward(self, content_code: torch.Tensor) -> torch.Tensor:
        h = self.proj(content_code)
        h = self.pool(h).squeeze(-1)
        return self.cls(h)


class DisentangledConverter(nn.Module):
    """Wrapper around FlowConverter + ContentSpeakerAdversary.

    The adversary is used standalone in train_flow.py
    (``disentangled.adversary(grad_reverse(content))``). The wrapper
    exists only to keep the adversary's parameters in the same module
    tree for ``.to(device)`` / ``state_dict``. At inference, use the
    inner FlowConverter directly (the adversary is discarded).
    """

    def __init__(self, converter: "FlowConverter", n_speakers: int):
        super().__init__()
        self.converter = converter
        self.adversary = ContentSpeakerAdversary(
            converter.config.latent_dim, n_speakers
        )
