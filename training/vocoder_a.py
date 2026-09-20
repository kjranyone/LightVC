"""Vocoder A (v6 rebuild, 2026-07-22): PROVEN iSTFTNet/HiFiGAN components.

Root cause of v1-v5 failures found: (1) NO weight_norm (HiFiGAN/BigVGAN all use
it -> critical for stable vocoder-GAN training), (2) cumsum phase killed
optimization (v1 free-phase overfit converged; v2-5 cumsum didn't). Fix = drop
the from-scratch bespoke design and use PROVEN pieces:
  mel -> weight_norm convs -> [nearest-upsample(causal, no checkerboard) +
  weight_norm conv + HiFiGAN MRF resblocks(leaky_relu)] xUP -> free-phase tiny
  iSTFT. Free phase = iSTFTNet standard (optimizable); coherence learned by
  capacity + GAN (NOT a hard cumsum constraint). Causal (left-pad).

Escapes the synth-window curse (network supplies freq resolution via upsampling,
tiny iSTFT = low latency). Gate: does the OVERFIT converge (v2-5 cumsum did not)?
"""
from __future__ import annotations
import sys, argparse, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import weight_norm
sys.path.insert(0, str(Path(__file__).parent))
from aa import UpSample1d, AALeaky, SnakeBeta  # AA upsample/act (FIR) + BigVGAN periodic act
from kansei_vocoder import ConvNeXtBlock1d      # proven Vocos-style backbone (freeC uses it)

LRELU = 0.1


def cpad(x, k, d=1):
    return F.pad(x, ((k - 1) * d, 0))            # causal left-pad


class ResBlock(nn.Module):
    """HiFiGAN MRF residual: leaky_relu + weight_norm dilated conv, causal."""
    def __init__(self, ch, k=3, dilations=(1, 3, 5)):
        super().__init__()
        self.k = k
        self.d = dilations
        self.c1 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, dilation=dd)) for dd in dilations])
        self.c2 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, dilation=1)) for _ in dilations])

    def forward(self, x):
        for a, b, dd in zip(self.c1, self.c2, self.d):
            xt = a(cpad(F.leaky_relu(x, LRELU), self.k, dd))
            xt = b(cpad(F.leaky_relu(xt, LRELU), self.k, 1))
            x = x + xt
        return x


class MRF(nn.Module):
    def __init__(self, ch, ks=(3, 7, 11)):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock(ch, k) for k in ks])

    def forward(self, x):
        return sum(b(x) for b in self.blocks) / len(self.blocks)


class FreeVocoderA(nn.Module):
    """v10: TINY-iSTFT (RTF 0.13, ultra-low-latency premise) + CAUSAL ConvTranspose
    upsample (HiFiGAN/iSTFTNet-proven: learned filter, no nearest-imaging, no aa
    over-smooth) + MRF + free phase + weight_norm. Light AND trainable."""
    def __init__(self, n_mels=128, dim=256, n_blocks=6, ups=(2, 2, 2, 2),
                 istft_nfft=64, mel_hop=128, causal=True, **kw):
        super().__init__()
        self.causal = causal
        self.pre = weight_norm(nn.Conv1d(n_mels, dim, 7))
        ch = dim
        self.ratios = list(ups)
        self.up = nn.ModuleList()
        self.mrf = nn.ModuleList()
        for r in ups:
            nch = max(ch // 2, 64)
            self.up.append(weight_norm(nn.ConvTranspose1d(ch, nch, 2 * r, stride=r)))
            self.mrf.append(MRF(nch))
            ch = nch
        self.nb = istft_nfft // 2 + 1
        self.post = weight_norm(nn.Conv1d(ch, 2 * self.nb, 7))
        self.istft_nfft = istft_nfft
        self.istft_win = istft_nfft
        self.istft_hop = mel_hop // int(np.prod(ups))
        self.register_buffer("window", torch.hann_window(istft_nfft))

    def forward(self, mel, f0=None):
        x = self.pre(cpad(mel, 7))
        for up, mrf, r in zip(self.up, self.mrf, self.ratios):
            x = up(F.leaky_relu(x, LRELU))[..., :-r]             # causal ConvTranspose (trim future r)
            x = mrf(x)
        h = self.post(cpad(F.leaky_relu(x, LRELU), 7)).float()
        mag = torch.exp(h[:, :self.nb].clamp(-14.0, 4.0))
        ph = h[:, self.nb:]                                      # free phase
        S = mag * (torch.cos(ph) + 1j * torch.sin(ph))
        return torch.istft(S, self.istft_nfft, self.istft_hop, self.istft_win,
                           self.window, center=True)


class ResBlockW(nn.Module):
    """Causal MRF residual for direct-waveform. act='snake' uses SnakeBeta (BigVGAN
    periodic activation -> HF/harmonic vitality, fixes muffled/lifeless); 'leaky'
    plain causal leaky. aa=True wraps leaky in anti-aliased (AALeaky, non-causal)."""
    def __init__(self, ch, k=3, dilations=(1, 3, 5), aa=False, act="leaky"):
        super().__init__()
        self.k, self.d, self.act = k, dilations, act
        self.c1 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, dilation=dd)) for dd in dilations])
        self.c2 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, dilation=1)) for _ in dilations])
        if act == "snake":
            self.s1 = nn.ModuleList([SnakeBeta(ch) for _ in dilations])
            self.s2 = nn.ModuleList([SnakeBeta(ch) for _ in dilations])
        self.aa = AALeaky() if aa else None

    def _act(self, x, s):
        if self.act == "snake":
            return s(x)
        return self.aa(x) if self.aa is not None else F.leaky_relu(x, LRELU)

    def forward(self, x):
        for i, (a, b, dd) in enumerate(zip(self.c1, self.c2, self.d)):
            s1 = self.s1[i] if self.act == "snake" else None
            s2 = self.s2[i] if self.act == "snake" else None
            xt = a(cpad(self._act(x, s1), self.k, dd))
            xt = b(cpad(self._act(xt, s2), self.k, 1))
            x = x + xt
        return x


class MRFW(nn.Module):
    def __init__(self, ch, ks=(3, 7, 11), aa=False, act="leaky"):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlockW(ch, k, aa=aa, act=act) for k in ks])

    def forward(self, x):
        return sum(b(x) for b in self.blocks) / len(self.blocks)


class WaveVocoderA(nn.Module):
    """Direct-waveform head (NO phase head -> escapes free-phase iSTFT ceiling).
    Causal ConvTranspose upsample x prod(ups)=mel_hop to waveform + tanh. aa=True
    inserts anti-aliased leaky (kills v7 imaging, but AALeaky FIR is non-causal ->
    small lookahead). F0-non-slaved (mel only). RTF/lookahead measured, not assumed."""
    def __init__(self, n_mels=128, dim=256, ups=(4, 4, 2, 2, 2), mel_hop=128,
                 causal=True, aa=False, aa_up=None, lean_last=1, act="leaky", **kw):
        super().__init__()
        assert int(np.prod(ups)) == mel_hop, f"prod(ups)={np.prod(ups)} != mel_hop {mel_hop}"
        aa_up = aa if aa_up is None else aa_up   # aa_up=upsampling-transition AA; aa=resblock AA
        self.causal = causal
        self.act = act
        self.ratios = list(ups)
        self.pre = weight_norm(nn.Conv1d(n_mels, dim, 7))
        ch = dim
        self.up = nn.ModuleList(); self.mrf = nn.ModuleList(); self.act_top = nn.ModuleList()
        self.aa_top = AALeaky() if aa_up else None
        n = len(ups)
        for i, r in enumerate(ups):
            self.act_top.append(SnakeBeta(ch) if act == "snake" else None)
            nch = max(ch // 2, 32)
            self.up.append(weight_norm(nn.ConvTranspose1d(ch, nch, 2 * r, stride=r)))
            # lean: drop MRF (heaviest, at high sample rate) on the last `lean_last` stages
            self.mrf.append(None if i >= n - lean_last else MRFW(nch, aa=aa, act=act))
            ch = nch
        self.post_act = SnakeBeta(ch) if act == "snake" else None
        self.post = weight_norm(nn.Conv1d(ch, 1, 7))

    def _top(self, x, i):
        if self.act == "snake":
            return self.act_top[i](x)
        return self.aa_top(x) if self.aa_top is not None else F.leaky_relu(x, LRELU)

    def forward(self, mel, f0=None):
        x = self.pre(cpad(mel, 7))
        for i, (up, mrf, r) in enumerate(zip(self.up, self.mrf, self.ratios)):
            x = self._top(x, i)
            x = up(x)[..., :-r]
            if mrf is not None:
                x = mrf(x)
        xp = self.post_act(x) if self.post_act is not None else F.leaky_relu(x, LRELU)
        return torch.tanh(self.post(cpad(xp, 7))).squeeze(1)


def ola_istft(S, nfft, hop, window):
    """Manual causal overlap-add iSTFT (center=False). Output sample t depends only
    on frames starting <= t => 0 lookahead regardless of nfft. Escapes the false
    'synthesis window = latency' belief that drove freeC to nfft256 (muddy)."""
    fr = torch.fft.irfft(S, n=nfft, dim=1) * window.view(1, -1, 1)      # [B,nfft,T]
    B, _, T = fr.shape
    L = (T - 1) * hop + nfft
    out = F.fold(fr, (1, L), (1, nfft), stride=(1, hop)).squeeze(2).squeeze(1)
    wn = F.fold((window ** 2).view(1, nfft, 1).expand(B, nfft, T),
                (1, L), (1, nfft), stride=(1, hop)).squeeze(2).squeeze(1)
    return out / (wn + 1e-8)


class CBlockC(nn.Module):
    """Causal dilated-conv residual block (frame-rate)."""
    def __init__(self, ch, k=3, ds=(1, 3, 5)):
        super().__init__()
        self.k, self.d = k, ds
        self.c = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, dilation=dd)) for dd in ds])

    def forward(self, x):
        for c, dd in zip(self.c, self.d):
            x = x + c(cpad(F.leaky_relu(x, LRELU), self.k, dd))
        return x


class CausalISTFTHead(nn.Module):
    """LARGE-nfft causal iSTFT-head. Fine freq resolution (nfft2048=22Hz/bin resolves
    mid+HF harmonics -> no muddiness, no muffling) at 0 lookahead (causal OLA iSTFT)
    and RTF ~0.10 (frame-rate). Free phase (coherent at large nfft, unlike tiny v10).
    The correct escape from the mid/HF tradeoff, found by rejecting 'nfft=latency'."""
    def __init__(self, n_mels=128, dim=384, n_blocks=8, nfft=2048, hop=128, causal=True, **kw):
        super().__init__()
        self.causal = causal
        self.pre = weight_norm(nn.Conv1d(n_mels, dim, 7))
        self.blocks = nn.ModuleList([ConvNeXtBlock1d(dim, causal=causal) for _ in range(n_blocks)])
        self.nb = nfft // 2 + 1
        self.post = weight_norm(nn.Conv1d(dim, 2 * self.nb, 7))
        self.nfft, self.hop = nfft, hop
        self.register_buffer("window", torch.hann_window(nfft))

    def forward(self, mel, f0=None):
        x = self.pre(cpad(mel, 7))
        for b in self.blocks:
            x = b(x)
        h = self.post(cpad(F.leaky_relu(x, LRELU), 7)).float()
        mag = torch.exp(h[:, :self.nb].clamp(-14.0, 4.0))
        ph = h[:, self.nb:].clamp(-1e3, 1e3)                   # bound phase: fp16-overflow inf -> cos(inf)=NaN guard
        S = mag * (torch.cos(ph) + 1j * torch.sin(ph))
        return ola_istft(S, self.nfft, self.hop, self.window)


class CombISTFTHead(CausalISTFTHead):
    """c11: CausalISTFTHead (PLAIN free phase) + INTERNAL-f0 harmonic-comb MAGNITUDE
    prior. Isolates the one idea from HarmonicISTFTHead that was never tested clean:
    c4 shipped the comb together with heterodyne phase (cumsum random-walk) and a
    per-bin random-phase gate, and the c7 diagnosis confirmed BOTH of those as the
    artifact sources (decoherence / 74% noise, crest 180). Here the comb rides the
    c8-c10 clean chain instead: plain free phase, nothing else added.

    f0 / voiced-scale predicted INTERNALLY (recon loss supervises => no external
    octave-error dependency, cf. f0-octave-weak-fundamental). F0 shapes the MAGNITUDE
    PRIOR only, not a time-domain excitation => NOT source-filter/NSF.

    Warm-start safe: pre/blocks/post are bit-identical to CausalISTFTHead, so a c10
    checkpoint loads 1:1 and only `f0head`/`alpha` are new. svoi=0 => bit-identical
    to c10 (lossless fallback). If svoi stays 0 the net is saying the prior is
    unnecessary -- but see the three fixes below, without which that verdict is not
    earned (the c11 run reached svoi=8e-12 by step 8k for the WRONG reasons):

    (1) INIT. weight_norm keeps the ORIGINAL magnitude in weight_g, so writing a
        small weight_v after wrapping only rotates the filter -- it does NOT shrink
        it. c11 therefore launched with svoi~1 on a random f0 (comb fully on,
        garbage) and the optimizer spent 8k steps killing it. Init the raw conv
        BEFORE wrapping so the magnitude is actually small.
    (2) ZERO-MEAN COMB. a*(cos-1) is <=0 everywhere, so switching the prior on can
        only REMOVE energy -> the early gradient always points at svoi=0 no matter
        how good f0 is. Subtracting the per-frame bin mean makes the comb
        redistribute (boost at k*f0, cut between) with no energy bias.
    (3) SVOI WARMUP. d(mag)/d(f0) is proportional to svoi, so svoi=0 also kills the
        f0 gradient: f0 can never improve, so svoi never has a reason to come back.
        Chicken-and-egg. `warm` (set by the trainer, 1->0) blends a forced svoi in
        so f0 receives gradient before the net is allowed to switch the prior off."""
    def __init__(self, n_mels=128, dim=384, n_blocks=8, nfft=2048, hop=128, sr=44100,
                 causal=True, fmin=60.0, fmax=500.0, warm_svoi=0.3, **kw):
        super().__init__(n_mels=n_mels, dim=dim, n_blocks=n_blocks, nfft=nfft,
                         hop=hop, causal=causal, **kw)
        # bin-Nyquist: the comb is sampled at bin centers (sr/nfft apart), so a
        # harmonic period needs >=2 bins or the prior ALIASES into a wrong pattern.
        # nfft512@44.1k => 86.1Hz/bin => f0 must stay above 172Hz (fine for the
        # female corpus). nfft2048 => 43Hz floor, so fmin stays at 60.
        self.fmin, self.fmax = max(fmin, 2.0 * sr / nfft), fmax
        self.warm_svoi, self.warm = warm_svoi, 0.0                 # warm: trainer sets 1->0
        f0c = nn.Conv1d(dim, 2, 7)
        nn.init.normal_(f0c.weight, std=1e-4)                      # small, NOT zero (dead-grad rule)
        nn.init.zeros_(f0c.bias)
        f0c.bias.data[1] = -4.0                                    # sigmoid -> svoi ~0.018 at init
        self.f0head = weight_norm(f0c)                             # wrap AFTER init (fix 1)
        self.alpha = nn.Parameter(torch.tensor(1.0))               # comb sharpness (softplus)
        self.register_buffer("fbin", torch.arange(self.nb).float() * sr / nfft)

    def forward(self, mel, f0=None):
        x = self.pre(cpad(mel, 7))
        for b in self.blocks:
            x = b(x)
        xa = F.leaky_relu(x, LRELU)
        h = self.post(cpad(xa, 7)).float()
        g = self.f0head(cpad(xa, 7)).float()
        f0v = self.fmin + (self.fmax - self.fmin) * torch.sigmoid(g[:, 0:1])        # [B,1,T]
        svoi = torch.sigmoid(g[:, 1:2])                                            # [B,1,T]
        if self.warm > 0.0:
            svoi = (1.0 - self.warm) * svoi + self.warm * self.warm_svoi           # fix 3
        c = torch.cos(2 * math.pi * self.fbin.view(1, -1, 1) / f0v.clamp(min=1.0))
        comb = F.softplus(self.alpha) * (c - c.mean(dim=1, keepdim=True))           # zero-mean (fix 2)
        mag = torch.exp((h[:, :self.nb] + svoi * comb).clamp(-14.0, 4.0))
        ph = h[:, self.nb:].clamp(-1e3, 1e3)
        S = mag * (torch.cos(ph) + 1j * torch.sin(ph))
        return ola_istft(S, self.nfft, self.hop, self.window)


class HeteroISTFTHead(nn.Module):
    """Heterodyned-phase causal iSTFT-head (derived from the coherence requirement).
    Phase = exact per-bin nominal advance (2*pi*k*hop/N, computed, no wrap-alias)
    + cumsum of a small learned deviation (coherent + learnable) -> sharp harmonics
    that survive overlap-add. Per-bin aperiodic gate: harmonic bins get coherent
    phase, noise bins get random phase (breath/ASMR first-class). F0-free (per-bin,
    not per-f0). Causal (cumsum is causal), 0 lookahead (OLA iSTFT), RTF ~0.1."""
    def __init__(self, n_mels=128, dim=384, n_blocks=8, nfft=2048, hop=128,
                 causal=True, beta=0.30, **kw):
        super().__init__()
        self.causal = causal
        self.nfft, self.hop, self.beta = nfft, hop, beta
        self.nb = nfft // 2 + 1
        self.pre = weight_norm(nn.Conv1d(n_mels, dim, 7))
        self.blocks = nn.ModuleList([ConvNeXtBlock1d(dim, causal=causal) for _ in range(n_blocks)])
        self.head = weight_norm(nn.Conv1d(dim, 3 * self.nb, 7))     # log-mag, phase-dev, aperiodic-gate
        self.register_buffer("window", torch.hann_window(nfft))
        self.register_buffer("kbin", torch.arange(self.nb).float())

    def forward(self, mel, f0=None):
        x = self.pre(cpad(mel, 7))
        for b in self.blocks:
            x = b(x)
        h = self.head(cpad(F.leaky_relu(x, LRELU), 7)).float()      # [B, 3nb, T]
        nb, T = self.nb, h.shape[-1]
        mag = torch.exp(h[:, :nb].clamp(-14.0, 4.0))
        dev = self.beta * torch.tanh(h[:, nb:2 * nb])               # small freq deviation [B,nb,T]
        gate = torch.sigmoid(h[:, 2 * nb:])                          # harmonic prob per bin [B,nb,T]
        # exact per-bin nominal phase advance (precision-safe: cycles then remainder)
        midx = torch.arange(T, device=h.device).float()
        cyc = torch.outer(self.kbin, midx) * (self.hop / self.nfft)  # [nb,T] in cycles
        nom = (2 * math.pi) * torch.remainder(cyc, 1.0)              # [nb,T]
        phi = nom.unsqueeze(0) + torch.cumsum(dev, dim=2)           # coherent phase (causal cumsum)
        rnd = (2 * math.pi) * torch.rand_like(mag)                   # random phase for noise
        S = mag * (gate * (torch.cos(phi) + 1j * torch.sin(phi))
                   + (1.0 - gate) * (torch.cos(rnd) + 1j * torch.sin(rnd)))
        return ola_istft(S, self.nfft, self.hop, self.window)


class HarmonicISTFTHead(nn.Module):
    """Heterodyne phase + INTERNAL-f0 differentiable harmonic-comb magnitude prior.
    Attacks the real bottleneck (magnitude sharpening = harmonic placement = needs
    f0). Network predicts f0 INTERNALLY (self-supervised by recon loss -> robust to
    octave-detector errors, unlike external f0). Comb C=exp(alpha*(cos(2*pi*f/f0)-1))
    peaks at k*f0 -> sharp harmonics. F0 shapes the MAGNITUDE PRIOR (not a time-domain
    excitation -> NOT source-filter/Beatrice/NSF). Voiced-scale s->0 disables comb =>
    safe fallback to plain heterodyne (never worse). Causal, 0-LA, RTF ~0.15."""
    def __init__(self, n_mels=128, dim=384, n_blocks=8, nfft=2048, hop=128, sr=44100,
                 causal=True, beta=0.30, fmin=60.0, fmax=500.0, **kw):
        super().__init__()
        self.causal = causal
        self.nfft, self.hop, self.beta = nfft, hop, beta
        self.fmin, self.fmax = fmin, fmax
        self.nb = nfft // 2 + 1
        self.pre = weight_norm(nn.Conv1d(n_mels, dim, 7))
        self.blocks = nn.ModuleList([ConvNeXtBlock1d(dim, causal=causal) for _ in range(n_blocks)])
        self.head = weight_norm(nn.Conv1d(dim, 3 * self.nb + 2, 7))   # logE, dev, gate | f0, voiced-scale
        self.alpha = nn.Parameter(torch.tensor(1.0))                   # comb sharpness (softplus)
        self.register_buffer("window", torch.hann_window(nfft))
        self.register_buffer("kbin", torch.arange(self.nb).float())
        self.register_buffer("fbin", torch.arange(self.nb).float() * sr / nfft)

    def forward(self, mel, f0=None):
        x = self.pre(cpad(mel, 7))
        for b in self.blocks:
            x = b(x)
        h = self.head(cpad(F.leaky_relu(x, LRELU), 7)).float()
        nb, T = self.nb, h.shape[-1]
        logE = h[:, :nb].clamp(-14.0, 4.0)
        dev = self.beta * torch.tanh(h[:, nb:2 * nb])
        gate = torch.sigmoid(h[:, 2 * nb:3 * nb])
        f0v = self.fmin + (self.fmax - self.fmin) * torch.sigmoid(h[:, 3 * nb:3 * nb + 1])   # [B,1,T]
        svoi = torch.sigmoid(h[:, 3 * nb + 1:3 * nb + 2])                                    # [B,1,T]
        a = F.softplus(self.alpha)
        comb = a * (torch.cos(2 * math.pi * self.fbin.view(1, nb, 1) / f0v.clamp(min=1.0)) - 1.0)  # <=0 [B,nb,T]
        mag = torch.exp((logE + svoi * comb).clamp(-14.0, 4.0))
        midx = torch.arange(T, device=h.device).float()
        cyc = torch.outer(self.kbin, midx) * (self.hop / self.nfft)
        nom = (2 * math.pi) * torch.remainder(cyc, 1.0)
        phi = nom.unsqueeze(0) + torch.cumsum(dev, dim=2)
        rnd = (2 * math.pi) * torch.rand_like(mag)
        S = mag * (gate * (torch.cos(phi) + 1j * torch.sin(phi))
                   + (1.0 - gate) * (torch.cos(rnd) + 1j * torch.sin(rnd)))
        return ola_istft(S, self.nfft, self.hop, self.window)


def mrstft(y, yh, ffts=(512, 1024, 2048)):
    loss = 0.0
    for n in ffts:
        w = torch.hann_window(n, device=y.device)
        Y = torch.stft(y, n, n // 4, n, w, return_complex=True).abs()
        Yh = torch.stft(yh, n, n // 4, n, w, return_complex=True).abs()
        m = min(Y.shape[-1], Yh.shape[-1])
        Y, Yh = Y[..., :m], Yh[..., :m]
        loss = loss + (torch.log(Y + 1e-5) - torch.log(Yh + 1e-5)).abs().mean() \
            + ((Y - Yh).norm(dim=1) / (Y.norm(dim=1) + 1e-5)).mean()
    return loss


def _overfit():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-utt", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--seg", type=float, default=1.2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="../results/interpretable_vc/ear_vocoderA")
    args = ap.parse_args()
    from train_m1 import SR, DEV
    from f0leak_probe import load_wav
    from causal_mel import causal_mel
    import soundfile as sf
    held = sorted(Path(x.strip()) for x in open("../results/interpretable_vc/held_multispk.txt") if x.strip())
    seen, wavs = set(), []
    for f in held:
        d = torch.load(f, weights_only=False); spk = Path(d["path"]).parent.name
        if spk in seen:
            continue
        seen.add(spk)
        w = np.asarray(load_wav(d["path"]), np.float32)
        n = int(args.seg * SR)
        if len(w) >= n:
            wavs.append((spk[:8], w[:n]))
        if len(wavs) >= args.n_utt:
            break
    mels = [causal_mel(torch.from_numpy(w).to(DEV), n_fft=2048, hop=128) for _, w in wavs]
    ys = [torch.from_numpy(w).float().to(DEV) for _, w in wavs]
    net = FreeVocoderA(causal=bool(args.causal)).to(DEV)
    print(f"FreeVocoderA(v6 iSTFTNet+weightnorm) {sum(p.numel() for p in net.parameters())/1e6:.2f}M "
          f"| overfit {len(wavs)} utts | free-phase", flush=True)
    opt = torch.optim.AdamW(net.parameters(), args.lr, betas=(0.8, 0.99))
    for step in range(1, args.steps + 1):
        i = step % len(wavs)
        yh = net(mels[i]); y = ys[i]
        m = min(y.shape[-1], yh.shape[-1])
        loss = mrstft(y[:m].unsqueeze(0), yh[..., :m])
        opt.zero_grad(); loss.backward()
        gn = nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        if torch.isfinite(torch.as_tensor(gn)):
            opt.step()
        if step % 500 == 0:
            print(f"step {step} mrstft {loss.item():.3f}", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    net.eval()
    for (spk, w), mel in zip(wavs, mels):
        with torch.no_grad():
            yh = net(mel).squeeze().cpu().numpy()
        m = min(len(w), len(yh))
        g = np.sqrt((w[:m] ** 2).mean() / ((yh[:m] ** 2).mean() + 1e-9))
        sf.write(out / f"{spk}_0gt.wav", np.clip(w[:m], -1, 1), SR, subtype="PCM_16")
        sf.write(out / f"{spk}_1vocoderA.wav", np.clip(yh[:m] * g, -1, 1), SR, subtype="PCM_16")
    print("overfit renders ->", out, flush=True)


if __name__ == "__main__":
    _overfit()
