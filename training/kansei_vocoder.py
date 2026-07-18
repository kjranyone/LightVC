"""KanseiVocoder — own frontier neural vocoder for babiko-voice ASMR VC.

Synthesis of every hard-won lesson (this session's copy-synth diagnostic + P1-P5):
  - istft transparent / A/S fails  -> ISTFT head (proven-transparent backend,
    n_fft2048/hop512/win2048 = the exact grid measured at 71dB).
  - time-upsample = jirijiri (M1)  -> isotropic backbone, NO ConvTranspose
    (Vocos-style; jirijiri structurally absent).
  - AFHN phase-regression RMS collapse -> F0-driven harmonic excitation gives a
    deterministic phase PRIOR (HiFTNet); the net predicts only the residual.
  - Vocos complex head mag*(cos p + j sin p), clip(exp) -> phase well-defined,
    no wrap instability (more stable than HiFTNet's sin(phase)).
  - ASMR breath/whisper first-class -> explicit noise branch in the excitation.
  - XPU-safe -> groups=1 convs only (no depthwise ConvNeXt).
  - non-regressive texture -> adversarial (P3), handled in the training harness.

Own architecture, borrowed nothing: excitation = our HarmonicSource; backbone,
source-injection, and head are written here. V-1 is non-causal (center STFT) to
isolate the transparency variable; V-2 makes STFT/backbone causal + streaming.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from ltv_render import HOP, SR, HarmonicSource

NFFT = 2048
WIN = 2048
NB = NFFT // 2 + 1  # 1025


class ConvNeXtBlock1d(nn.Module):
    """Isotropic residual block, groups=1 (XPU-safe), causal-capable."""

    def __init__(self, dim: int, mult: int = 3, k: int = 7, causal: bool = False):
        super().__init__()
        self.k = k
        self.causal = causal
        self.pad = k - 1 if causal else k // 2
        self.dw = nn.Conv1d(dim, dim, k, padding=0, groups=1)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * mult, dim)

    def forward(self, x):  # x: [B, C, T]
        r = x
        if self.causal:
            x = F.pad(x, (self.pad, 0))
        else:
            x = F.pad(x, (self.pad, self.k - 1 - self.pad))
        x = self.dw(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.transpose(1, 2)
        return r + x


class KanseiVocoder(nn.Module):
    def __init__(self, n_mels: int = 128, dim: int = 512, n_layers: int = 8,
                 causal: bool = False, n_harm_max: int = 400):
        super().__init__()
        self.causal = causal
        self.register_buffer("window", torch.hann_window(WIN))
        self.harm = HarmonicSource(causal=causal, n_harm_max=n_harm_max)
        # input: mel + source STFT (real, imag) -> dim
        self.in_proj = nn.Conv1d(n_mels + 2 * NB, dim, 1)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock1d(dim, causal=causal) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2 * NB)

    def _stft(self, wav):
        center = not self.causal
        pad = 0 if center else (WIN - HOP)
        if pad:
            wav = F.pad(wav, (pad, 0))
        S = torch.stft(wav, NFFT, HOP, WIN, self.window, center=center,
                       return_complex=True)
        return S  # [B, NB, T]

    def _istft(self, S):
        center = not self.causal
        y = torch.istft(S, NFFT, HOP, WIN, self.window, center=center)
        return y

    def forward(self, mel, f0):
        # excitation (deterministic phase prior + explicit noise for breath)
        with torch.no_grad():
            e_h, _ = self.harm(f0)                      # [B, T*HOP]
            noise = torch.randn_like(e_h) * 0.1
            exc = e_h + noise
        Es = self._stft(exc)                            # [B, NB, Tf]
        T = min(mel.shape[-1], Es.shape[-1])
        mel, Es = mel[..., :T], Es[..., :T]
        src = torch.cat([Es.real, Es.imag], dim=1)      # [B, 2*NB, T]
        x = self.in_proj(torch.cat([mel, src], dim=1))  # [B, dim, T]
        for b in self.blocks:
            x = b(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = self.head(x.transpose(1, 2)).transpose(1, 2)  # [B, 2*NB, T]
        mag = torch.clip(torch.exp(h[:, :NB]), max=1e2)
        # anchor output phase to the excitation phase (coherent harmonics ->
        # clean inter-harmonic valleys, kills かすれ). net predicts magnitude
        # (the filter envelope) + a bounded phase residual (0 at init).
        unit_src = Es / (Es.abs() + 1e-6)                # complex unit, src phase
        pres = torch.pi * torch.tanh(h[:, NB:])          # bounded residual
        S = mag * unit_src * (torch.cos(pres) + 1j * torch.sin(pres))
        return self._istft(S)                            # [B, T*HOP]


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = KanseiVocoder(causal=False).to(dev)
    n = sum(p.numel() for p in m.parameters())
    print(f"KanseiVocoder {n/1e6:.1f}M params")
    mel = torch.randn(2, 128, 64, device=dev)
    f0 = torch.rand(2, 64, device=dev) * 200 + 100
    y = m(mel, f0)
    print("out", tuple(y.shape), "expected ~", 64 * HOP)
    y.sum().backward()
    print("backward OK")
