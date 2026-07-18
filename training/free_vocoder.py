"""FreeVocoder — F0-free thin neural vocoder (R-proto-A).

Session finding that motivates dropping F0: BigVGAN (no F0 input) is transparent
on this weak-fundamental voice, while kansei (F0-driven harmonic source) hoarses
on UNSEEN utts because F0 is untrackable here. The vocoder does not need F0 —
mel already encodes pitch; F0 control belongs to the upstream VC/prosody stage.
An F0-free vocoder cannot "measure F0 and be wrong", so the unseen-hoarse root
cause disappears (survey §7-3 Vocos / §9-6). Target: match BigVGAN by ear while
being ~10x lighter, then shrink toward CPU-RT.

Architecture = Vocos (MIT) faithful, on our proven ISTFT grid (n_fft2048/hop512/
win2048, 71dB round-trip) and XPU-safe groups=1 ConvNeXt. NO harmonic source, NO
F0: mel -> isotropic backbone -> complex-STFT head (mag*exp(j*phase)) -> ISTFT.
Free phase is what Vocos/BigVGAN use and works at scale (the earlier free-phase
hoarse was a harmonic-source-present artifact, not a pure-Vocos failure).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from ltv_render import HOP
from kansei_vocoder import ConvNeXtBlock1d, NFFT, WIN, NB


class FreeVocoder(nn.Module):
    """F0-free ISTFT-head vocoder with a parameterizable OUTPUT synthesis grid.

    config C (low latency): the OUTPUT synthesis window (`win`/`hop`) is what sets
    streaming reconstruction latency, decoupled from the INPUT mel's analysis
    window (kept rich upstream for resolution). Short causal `win`+`hop` -> ~hop
    latency; the mel still carries high-res conditioning. Defaults reproduce the
    original 2048/512 grid so existing checkpoints load unchanged.
    """

    def __init__(self, n_mels: int = 128, dim: int = 512, n_layers: int = 8,
                 causal: bool = False, nfft: int = NFFT, win: int = WIN,
                 hop: int = HOP):
        super().__init__()
        self.causal = causal
        self.nfft, self.win, self.hop = nfft, win, hop
        self.nb = nfft // 2 + 1
        self.register_buffer("window", torch.hann_window(win))
        self.embed = nn.Conv1d(n_mels, dim, 7, padding=0)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock1d(dim, causal=causal) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2 * self.nb)

    def latency_ms(self, sr: int = 44100) -> float:
        """Algorithmic reconstruction latency of the causal synthesis: the OLA
        tail (win - hop) that must be waited before an output sample is final,
        plus one block (hop). Non-causal adds ~win/2 of lookahead on top."""
        base = (self.win if self.causal else self.win + self.win // 2)
        return 1000.0 * base / sr

    def _istft(self, S):
        # center=True for training stability (Hann is 0 at edges -> center=False
        # fails NOLA at sample 0). The causal lookahead budget lives in the conv
        # stack; streaming deployment reconstructs with a custom causal OLA
        # (cf. ltv_render._ola_fold), so latency_ms reflects that target grid.
        return torch.istft(S, self.nfft, self.hop, self.win, self.window,
                           center=True)

    def forward(self, mel, f0=None):  # f0 ignored (F0-free); kept for harness parity
        if self.causal:
            x = F.pad(mel, (6, 0))
        else:
            x = F.pad(mel, (3, 3))
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = self.head(x.transpose(1, 2)).transpose(1, 2)   # [B, 2*nb, T]
        h = h.float()                                       # fp32 for AMP-safe iSTFT
        mag = torch.clip(torch.exp(h[:, :self.nb]), max=1e2)
        p = h[:, self.nb:]                                  # free phase angle
        S = mag * (torch.cos(p) + 1j * torch.sin(p))
        return self._istft(S)


class FreeVocoderIF(nn.Module):
    """F0-free vocoder with an instantaneous-frequency (cumsum) phase head.

    Diagnosis (10-metric triangulation, 2026-07-14): freeuniv's residual vs gt is
    ~2x bigvgan's in the LOW band (0-300Hz=fundamental) because hop512 > pitch
    period (~200 smp @220Hz) so the free-phase ISTFT head can't hold the
    fundamental's phase continuity across frame boundaries -> the core wobbles =
    "定位感のブレ". Fix: predict a per-frame phase INCREMENT and cumsum over time
    -> phase is temporally continuous by construction (survey §7-6(a) DDSP IF),
    so the fundamental stays coherent. Same size as FreeVocoder (no RT cost).
    Mag path identical; only the phase parameterization changes.
    """

    def __init__(self, n_mels: int = 128, dim: int = 512, n_layers: int = 8,
                 causal: bool = False):
        super().__init__()
        self.causal = causal
        self.register_buffer("window", torch.hann_window(WIN))
        self.embed = nn.Conv1d(n_mels, dim, 7, padding=0)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock1d(dim, causal=causal) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.mag_head = nn.Linear(dim, NB)
        self.dphi_head = nn.Linear(dim, NB)
        nn.init.zeros_(self.dphi_head.weight)
        nn.init.zeros_(self.dphi_head.bias)

    def load_from_free(self, sd):
        """Warm-start backbone/embed/norm + mag half of FreeVocoder's combined
        head; leave dphi_head at zero (learns increments fresh, fast)."""
        own = self.state_dict()
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        # split old head [2*NB, dim] / [2*NB] into mag (first NB)
        if "head.weight" in sd:
            own["mag_head.weight"] = sd["head.weight"][:NB]
            own["mag_head.bias"] = sd["head.bias"][:NB]
        self.load_state_dict(own)

    def _istft(self, S):
        center = not self.causal
        return torch.istft(S, NFFT, HOP, WIN, self.window, center=center)

    def forward(self, mel, f0=None):
        x = F.pad(mel, (6, 0)) if self.causal else F.pad(mel, (3, 3))
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x.transpose(1, 2))            # [B, T, dim]
        mag = torch.clip(torch.exp(self.mag_head(x)), max=1e2)  # [B, T, NB]
        dphi = self.dphi_head(x)                     # per-frame phase increment
        phi = torch.cumsum(dphi, dim=1)              # temporal integration
        S = (mag * (torch.cos(phi) + 1j * torch.sin(phi))).transpose(1, 2)
        return self._istft(S)


class FreeVocoderGCI(nn.Module):
    """F0-free vocoder with a GCI-anchor phase head (cross-frequency coherence).

    Theory (2026-07-15): monaural 定位感 (source compactness) is governed by
    cross-frequency phase coherence — all harmonics sharing a common time origin
    (GCI) makes the harmonic-comb magnitude reconstruct as a sharp periodic pulse
    train = tight image; phase dispersion smears it = wobble. The free-phase head
    has no structural tie binding bins to a common GCI. Here the head predicts,
    per frame, a shared time anchor tau_t (a linear phase -2*pi*k*tau/NFFT applied
    to ALL bins -> harmonics aligned to a common origin) + a small per-bin
    dispersion residual (the natural mixed-phase deviation). tau + comb magnitude
    = compact periodic pulses. Same size class; RT-friendly.
    """

    def __init__(self, n_mels: int = 128, dim: int = 512, n_layers: int = 8,
                 causal: bool = False):
        super().__init__()
        self.causal = causal
        self.register_buffer("window", torch.hann_window(WIN))
        self.register_buffer("kbin", torch.arange(NB).float())
        self.embed = nn.Conv1d(n_mels, dim, 7, padding=0)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock1d(dim, causal=causal) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.mag_head = nn.Linear(dim, NB)
        self.tau_head = nn.Linear(dim, 1)      # per-frame shared time anchor
        self.res_head = nn.Linear(dim, NB)     # per-bin dispersion residual
        nn.init.zeros_(self.tau_head.weight); nn.init.zeros_(self.tau_head.bias)
        nn.init.zeros_(self.res_head.weight); nn.init.zeros_(self.res_head.bias)

    def load_from_free(self, sd):
        own = self.state_dict()
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        if "head.weight" in sd:
            own["mag_head.weight"] = sd["head.weight"][:NB]
            own["mag_head.bias"] = sd["head.bias"][:NB]
        self.load_state_dict(own)

    def _istft(self, S):
        center = not self.causal
        return torch.istft(S, NFFT, HOP, WIN, self.window, center=center)

    def forward(self, mel, f0=None):
        x = F.pad(mel, (6, 0)) if self.causal else F.pad(mel, (3, 3))
        x = self.embed(x)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x.transpose(1, 2))                 # [B, T, dim]
        mag = torch.clip(torch.exp(self.mag_head(x)), max=1e2)
        tau = (WIN / 2) * torch.tanh(self.tau_head(x))   # [B, T, 1] samples
        res = torch.pi * torch.tanh(self.res_head(x))    # bounded dispersion
        phi_lin = -2 * torch.pi * self.kbin[None, None, :] * tau / NFFT
        phi = phi_lin + res                              # [B, T, NB]
        S = (mag * (torch.cos(phi) + 1j * torch.sin(phi))).transpose(1, 2)
        return self._istft(S)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = FreeVocoder(causal=False).to(dev)
    n = sum(p.numel() for p in m.parameters())
    print(f"FreeVocoder {n/1e6:.1f}M params")
    mel = torch.randn(2, 128, 64, device=dev)
    y = m(mel)
    print("out", tuple(y.shape), "expected ~", 64 * HOP)
    y.sum().backward()
    print("backward OK")
