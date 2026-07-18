"""NSF-LTV v1.3 shared renderer (current/vocoder.md §3.1/§3.2).

Single source of truth for the LTV synthesis physics, shared by E0 oracle
analysis-synthesis (e0_oracle_ltv.py), E1+ training, and the E4 Candle parity
gate. Differentiable w.r.t. envelopes/d/a (excitation is non-learned, P2).

v1.3 (code review 2026-07-12):
  - backend="mm" is fully XPU-safe: min-phase FIR AND the LTV convolution run
    as fixed-matrix GEMMs (F.conv1d with groups=B*T is the known-fatal
    depthwise pattern on XPU backward).
  - causal=True (default): control-rate upsampling (f0/d/a) anchors on
    [t-1, t] = half-frame response delay, zero lookahead; hold_f0 never
    back-fills. causal=False is a reference/debug mode only — E0 gates and
    all recorded evidence run with causal=True; do not "fix" E0 to False.
  - nb_in default 1025 per spec; envelope shape is asserted.
  - linear-phase reference lives on MinPhaseFIR (backend-consistent).

Components:
  - HarmonicSource: optional time-varying maximum voiced frequency gate
    (HNM/DSM lineage; raised-cosine 500 Hz transition, closed-form RMS still
    holds with gated g_k). Gate math is differentiable w.r.t. mvf, but the
    source runs under no_grad — for E1 learnable Fm, lift the gating out of
    no_grad (frame-net head: sigmoid x Nyquist).
  - HarmonicSource: excitation hygiene per §3.1 (closed-form unit-RMS
    sqrt(sum g_k^2 / 2) — per-frame empirical RMS re-introduces frame-grid AM
    at low F0; Nyquist raised-cosine rolloff; continuous F0 hold; frac-phase).
  - pitch_sync_mod / subframe_gain: F2 noise modulation.
  - MinPhaseFIR: log-mag envelope (Nb bins, linear grid) -> causal min-phase
    FIR via real cepstrum (Oppenheim-Schafer). Backends: torch.fft ("fft",
    CPU/CUDA fast path) and matmul-DFT ("mm", XPU-safe, no complex dtype).
  - ltv_ola: NHV-style time-varying filtering, non-overlapping hop segments,
    tails overlap-added into the future (zero lookahead).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 44100
HOP = 512
NFFT = 4096
NBINS = NFFT // 2 + 1

_OLA_CACHE: dict = {}


def _up(x: torch.Tensor, factor: int) -> torch.Tensor:
    return F.interpolate(x, scale_factor=factor, mode="linear", align_corners=False)


def _up_causal(x: torch.Tensor, factor: int) -> torch.Tensor:
    n = x.shape[-1] * factor
    xp = torch.cat([x[..., :1], x], dim=-1)
    u = F.interpolate(xp, scale_factor=factor, mode="linear", align_corners=False)
    return u[..., factor // 2:factor // 2 + n]


def _upsample(x: torch.Tensor, factor: int, causal: bool) -> torch.Tensor:
    return _up_causal(x, factor) if causal else _up(x, factor)


def hold_f0(f0: torch.Tensor, fallback: float = 220.0, causal: bool = True) -> torch.Tensor:
    B, T = f0.shape
    voiced = f0 > 1.0
    ar = torch.arange(T, device=f0.device).expand(B, T)
    pos = torch.where(voiced, ar, torch.full_like(ar, -1))
    ff = pos.cummax(1).values
    held = torch.gather(f0, 1, ff.clamp(min=0))
    fb = torch.full_like(f0, fallback)
    if causal:
        return torch.where(ff >= 0, held, fb)
    first = torch.argmax(voiced.int(), 1, keepdim=True)
    ff2 = torch.where(ff < 0, first.expand(B, T), ff)
    out = torch.gather(f0, 1, ff2.clamp(min=0))
    return torch.where(voiced.any(1, keepdim=True).expand(B, T), out, fb)


class HarmonicSource(nn.Module):
    def __init__(self, sr: int = SR, hop: int = HOP, roll_start: float = 0.9,
                 n_harm_max: int = 400, chunk: int = 32, f0_floor: float = 55.0,
                 causal: bool = True, jitter: float = 0.0, shimmer: float = 0.0,
                 disp: str = "none", disp_c: float = 0.0) -> None:
        super().__init__()
        self.sr = sr
        self.hop = hop
        self.roll_start = roll_start
        self.n_harm_max = n_harm_max
        self.chunk = chunk
        self.f0_floor = f0_floor
        self.causal = causal
        self.jitter = jitter
        self.shimmer = shimmer
        self.disp = disp
        self.disp_c = disp_c

    @torch.no_grad()
    def forward(self, f0_frame: torch.Tensor,
                mvf_frame: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        f0c = hold_f0(f0_frame, causal=self.causal)
        mvf_s = None
        if mvf_frame is not None:
            mvf_s = _upsample(mvf_frame.unsqueeze(1), self.hop, self.causal).squeeze(1)
        f0s = _upsample(f0c.unsqueeze(1), self.hop, self.causal).squeeze(1)
        if self.jitter > 0.0:
            tau = int(0.005 * self.sr)
            t = torch.arange(4 * tau, device=f0s.device, dtype=torch.float32)
            ker = torch.exp(-t / tau)
            ker = (ker / ker.pow(2).sum().sqrt()).view(1, 1, -1)
            eps = torch.randn(f0s.shape[0], 1, f0s.shape[1] + 4 * tau - 1,
                              device=f0s.device)
            w = F.conv1d(eps, ker).squeeze(1)[:, :f0s.shape[1]]
            f0s = f0s * (1.0 + self.jitter * w).clamp(0.5, 2.0)
        p = torch.cumsum(f0s.double() / self.sr, dim=-1)
        pf = torch.frac(p)
        nyq = self.sr * 0.5
        r0 = self.roll_start * nyq
        floor = self.f0_floor if self.causal else max(self.f0_floor, float(f0c.min()))
        n_harm = int(min(self.n_harm_max, math.floor(nyq / floor)))
        if self.disp == "rand":
            phk_all = 2.0 * math.pi * torch.rand(n_harm + 1, device=f0s.device)
        elif self.disp == "hfrand":
            phk_all = 2.0 * math.pi * torch.rand(n_harm + 1, device=f0s.device)
            f0m = float(f0c[f0c > 1.0].median()) if (f0c > 1.0).any() else 200.0
            k_cut = max(1, int(self.disp_c / f0m))
            phk_all[:k_cut + 1] = 0.0
        e = torch.zeros_like(f0s)
        g2 = torch.zeros_like(f0s)
        for k0 in range(1, n_harm + 1, self.chunk):
            ks = torch.arange(k0, min(k0 + self.chunk, n_harm + 1),
                              device=f0s.device, dtype=torch.float64).view(1, -1, 1)
            ph = (2.0 * math.pi) * torch.frac(ks * pf.unsqueeze(1))
            if self.disp in ("rand", "hfrand"):
                ph = ph + phk_all[ks.long().view(-1)].view(1, -1, 1)
            elif self.disp == "quad":
                ph = ph - self.disp_c * (ks.float() ** 2)
            fk = (ks * f0s.double().unsqueeze(1))
            x = ((fk - r0) / (nyq - r0)).clamp(0.0, 1.0)
            g = (0.5 * (1.0 + torch.cos(math.pi * x))).float()
            if mvf_s is not None:
                xm = ((fk - (mvf_s.double().unsqueeze(1) - 250.0)) / 500.0).clamp(0.0, 1.0)
                g = g * (0.5 * (1.0 + torch.cos(math.pi * xm))).float()
            e = e + (torch.sin(ph.float()) * g).sum(1)
            g2 = g2 + g.pow(2).sum(1)
        e = e / (g2 * 0.5).sqrt().clamp(min=1e-4)
        if self.shimmer > 0.0:
            B, T = f0c.shape
            sh = (1.0 + torch.randn_like(f0c) * self.shimmer).clamp(0.2, 3.0)
            e = e * _upsample(sh.unsqueeze(1), self.hop, self.causal).squeeze(1)
        phase = (2.0 * math.pi) * pf.float()
        return e, phase


def pitch_sync_mod(phase: torch.Tensor, d_frame: torch.Tensor, phi0: torch.Tensor,
                   hop: int = HOP, causal: bool = True, p: int = 1) -> torch.Tensor:
    d = _upsample(d_frame.unsqueeze(1), hop, causal).squeeze(1).clamp(0.0, 1.0)
    q = 0.5 * (1.0 + torch.cos(phase - phi0))
    if p != 1:
        q = q ** p
    qmean = math.comb(2 * p, p) / 4.0 ** p
    return ((1.0 - d) + d * q) / ((1.0 - d) + d * qmean + 1e-6)


def subframe_gain(a: torch.Tensor, hop: int = HOP, causal: bool = True) -> torch.Tensor:
    B, T, J = a.shape
    a = a.clamp(min=0.0)
    a = a / a.mean(-1, keepdim=True).clamp(min=1e-4)
    return _upsample(a.reshape(B, 1, T * J), hop // J, causal).squeeze(1)


def clamp_envelope(h_log: torch.Tensor, nats: float = 8.0, tail: float = 4.0) -> torch.Tensor:
    m = h_log.mean(-1, keepdim=True)
    x = h_log - m
    ax = x.abs()
    soft = nats + tail * torch.tanh((ax - nats) / tail)
    return m + torch.sign(x) * torch.where(ax <= nats, ax, soft)


class MinPhaseFIR(nn.Module):
    def __init__(self, nb_in: int = 1025, k: int = 1024, n_fft: int = NFFT,
                 clamp_nats: float = 8.0, lifter_gamma: float = 0.0,
                 backend: str = "fft") -> None:
        super().__init__()
        self.nb_in = nb_in
        self.k = k
        self.n_fft = n_fft
        self.half = n_fft // 2
        self.clamp_nats = clamp_nats
        self.backend = backend
        if nb_in != NBINS:
            src = torch.linspace(0.0, 1.0, nb_in)
            dst = torch.linspace(0.0, 1.0, NBINS)
            idx = torch.searchsorted(src, dst).clamp(1, nb_in - 1)
            lo, hi = idx - 1, idx
            w = (dst - src[lo]) / (src[hi] - src[lo])
            interp = torch.zeros(NBINS, nb_in)
            interp[torch.arange(NBINS), lo] = 1.0 - w
            interp[torch.arange(NBINS), hi] += w
            self.register_buffer("interp", interp, persistent=False)
        else:
            self.interp = None
        if lifter_gamma > 0.0:
            lif = lifter_gamma ** torch.arange(NBINS, dtype=torch.float32)
            self.register_buffer("lif", lif, persistent=False)
        else:
            self.lif = None
        self._mm: dict = {}
        self._mm_lin: dict = {}

    def _build_mm(self, device: torch.device) -> dict:
        key = str(device)
        if key not in self._mm:
            n = torch.arange(NBINS, dtype=torch.float64)
            theta = (2.0 * math.pi / self.n_fft) * n.view(-1, 1) * n.view(1, -1)
            wk = torch.full((NBINS,), 2.0 / self.n_fft, dtype=torch.float64)
            wk[0] = wk[-1] = 1.0 / self.n_fft
            m = torch.arange(self.k, dtype=torch.float64)
            phi = (2.0 * math.pi / self.n_fft) * n.view(-1, 1) * m.view(1, -1)
            self._mm[key] = {
                "cos": torch.cos(theta).float().to(device),
                "sin": torch.sin(theta).float().to(device),
                "wk": wk.float().to(device),
                "cosm": torch.cos(phi).float().to(device),
                "sinm": torch.sin(phi).float().to(device),
            }
        return self._mm[key]

    def _build_mm_lin(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._mm_lin:
            n = torch.arange(NBINS, dtype=torch.float64)
            m = torch.arange(self.k, dtype=torch.float64) - float(self.k // 2)
            phi = (2.0 * math.pi / self.n_fft) * n.view(-1, 1) * m.view(1, -1)
            self._mm_lin[key] = torch.cos(phi).float().to(device)
        return self._mm_lin[key]

    def _fold(self, c: torch.Tensor) -> torch.Tensor:
        return torch.cat([c[..., :1], 2.0 * c[..., 1:self.half],
                          c[..., self.half:self.half + 1]], dim=-1)

    def _prep(self, h_log: torch.Tensor) -> torch.Tensor:
        assert h_log.shape[-1] == self.nb_in, (h_log.shape, self.nb_in)
        if self.interp is not None:
            h_log = h_log @ self.interp.T
        return clamp_envelope(h_log, self.clamp_nats)

    def _fft_path(self, h: torch.Tensor) -> torch.Tensor:
        c = torch.fft.irfft(torch.complex(h, torch.zeros_like(h)), n=self.n_fft)
        ch = self._fold(c[..., :NBINS])
        if self.lif is not None:
            ch = ch * self.lif
        chf = F.pad(ch, (0, self.n_fft - NBINS))
        spec = torch.exp(torch.fft.rfft(chf, n=self.n_fft))
        return torch.fft.irfft(spec, n=self.n_fft)[..., :self.k]

    def _mm_path(self, h: torch.Tensor) -> torch.Tensor:
        mm = self._build_mm(h.device)
        c = (h * mm["wk"]) @ mm["cos"]
        ch = self._fold(c)
        if self.lif is not None:
            ch = ch * self.lif
        re = ch @ mm["cos"]
        im = -(ch @ mm["sin"])
        mag = torch.exp(re)
        sre, sim = mag * torch.cos(im), mag * torch.sin(im)
        return (sre * mm["wk"]) @ mm["cosm"] - (sim * mm["wk"]) @ mm["sinm"]

    def forward(self, h_log: torch.Tensor) -> torch.Tensor:
        h = self._prep(h_log)
        return self._fft_path(h) if self.backend == "fft" else self._mm_path(h)

    def forward_linear(self, h_log: torch.Tensor) -> tuple[torch.Tensor, int]:
        h = self._prep(h_log)
        mag = torch.exp(h)
        half = self.k // 2
        if self.backend == "fft":
            b_full = torch.fft.irfft(torch.complex(mag, torch.zeros_like(mag)),
                                     n=self.n_fft)
            b = torch.cat([b_full[..., self.n_fft - half:],
                           b_full[..., :self.k - half]], dim=-1)
        else:
            mm = self._build_mm(h.device)
            b = (mag * mm["wk"]) @ self._build_mm_lin(h.device)
        return b, half


def _ola_fold(y: torch.Tensor, B: int, T: int, hop: int, K: int) -> torch.Tensor:
    L = hop + K - 1
    N = T * hop
    y = y.reshape(B, T, L).transpose(1, 2)
    out = F.fold(y, output_size=(1, N + K - 1), kernel_size=(1, L), stride=(1, hop))
    return out.reshape(B, N + K - 1)[..., :N]


def _build_ola_mm(hop: int, K: int, device: torch.device) -> dict:
    key = (hop, K, str(device))
    if key not in _OLA_CACHE:
        L = hop + K - 1
        n2 = 1 << (L - 1).bit_length()
        nb2 = n2 // 2 + 1
        k = torch.arange(nb2, dtype=torch.float64)
        th_e = (2.0 * math.pi / n2) * torch.arange(hop, dtype=torch.float64).view(-1, 1) * k
        th_b = (2.0 * math.pi / n2) * torch.arange(K, dtype=torch.float64).view(-1, 1) * k
        th_y = (2.0 * math.pi / n2) * k.view(-1, 1) * torch.arange(L, dtype=torch.float64)
        wk = torch.full((nb2,), 2.0 / n2, dtype=torch.float64)
        wk[0] = 1.0 / n2
        if n2 % 2 == 0:
            wk[-1] = 1.0 / n2
        _OLA_CACHE[key] = {
            "ce": torch.cos(th_e).float().to(device), "se": torch.sin(th_e).float().to(device),
            "cb": torch.cos(th_b).float().to(device), "sb": torch.sin(th_b).float().to(device),
            "cy": torch.cos(th_y).float().to(device), "sy": torch.sin(th_y).float().to(device),
            "wk": wk.float().to(device),
        }
    return _OLA_CACHE[key]


def ltv_ola(e: torch.Tensor, b: torch.Tensor, hop: int = HOP,
            backend: str = "conv") -> torch.Tensor:
    B, N = e.shape
    _, T, K = b.shape
    assert N == T * hop, (N, T, hop)
    if backend == "conv":
        segs = e.reshape(1, B * T, hop)
        w = b.reshape(B * T, 1, K).flip(-1)
        y = F.conv1d(F.pad(segs, (K - 1, K - 1)), w, groups=B * T)
        return _ola_fold(y, B, T, hop, K)
    mm = _build_ola_mm(hop, K, e.device)
    segs = e.reshape(B * T, hop)
    bk = b.reshape(B * T, K)
    sre, sim = segs @ mm["ce"], -(segs @ mm["se"])
    bre, bim = bk @ mm["cb"], -(bk @ mm["sb"])
    yre = sre * bre - sim * bim
    yim = sre * bim + sim * bre
    y = (yre * mm["wk"]) @ mm["cy"] - (yim * mm["wk"]) @ mm["sy"]
    return _ola_fold(y, B, T, hop, K)


class LtvRenderer(nn.Module):
    def __init__(self, sr: int = SR, hop: int = HOP, k_v: int = 1024, k_n: int = 256,
                 nb_in: int = 1025, lifter_gamma: float = 0.0, backend: str = "fft",
                 phase_mode: str = "min", roll_start: float = 0.9,
                 causal: bool = True, mod_p: int = 1, jitter: float = 0.0,
                 shimmer: float = 0.0, disp: str = "none", disp_c: float = 0.0) -> None:
        super().__init__()
        self.hop = hop
        self.k_v = k_v
        self.k_n = k_n
        self.phase_mode = phase_mode
        self.nb_in = nb_in
        self.backend = backend
        self.causal = causal
        self.mod_p = mod_p
        self.ola_backend = "conv" if backend == "fft" else "mm"
        self.harm = HarmonicSource(sr, hop, roll_start=roll_start, causal=causal,
                                   jitter=jitter, shimmer=shimmer, disp=disp,
                                   disp_c=disp_c)
        self.fir_v = MinPhaseFIR(nb_in, k_v, lifter_gamma=lifter_gamma, backend=backend)
        self.fir_n = MinPhaseFIR(nb_in, k_n, lifter_gamma=lifter_gamma, backend=backend)
        self.phi0 = nn.Parameter(torch.tensor(math.pi))

    def forward(self, f0_frame: torch.Tensor, h_v: torch.Tensor, h_n: torch.Tensor,
                d: torch.Tensor | None = None, a: torch.Tensor | None = None,
                noise: torch.Tensor | None = None,
                mvf: torch.Tensor | None = None) -> dict:
        e_h, phase = self.harm(f0_frame, mvf)
        T = h_v.shape[1]
        n = T * self.hop
        e_h, phase = e_h[:, :n], phase[:, :n]
        if noise is None:
            noise = torch.randn_like(e_h)
        else:
            noise = noise[:, :n]
        if d is not None:
            noise = noise * pitch_sync_mod(phase, d[:, :T], self.phi0, self.hop,
                                           self.causal, self.mod_p)
        if a is not None:
            noise = noise * subframe_gain(a[:, :T], self.hop, self.causal)
        if self.phase_mode == "min":
            y_h = ltv_ola(e_h, self.fir_v(h_v), self.hop, self.ola_backend)
            y_n = ltv_ola(noise, self.fir_n(h_n), self.hop, self.ola_backend)
        else:
            b_v, dv = self.fir_v.forward_linear(h_v)
            b_n, dn = self.fir_n.forward_linear(h_n)
            y_h = F.pad(ltv_ola(e_h, b_v, self.hop, self.ola_backend)[:, dv:], (0, dv))
            y_n = F.pad(ltv_ola(noise, b_n, self.hop, self.ola_backend)[:, dn:], (0, dn))
        return {"y": y_h + y_n, "y_h": y_h, "y_n": y_n, "phase": phase}


def _self_test() -> None:
    torch.manual_seed(0)
    fir_fft = MinPhaseFIR(nb_in=1025, k=1024, backend="fft")
    fir_mm = MinPhaseFIR(nb_in=1025, k=1024, backend="mm")
    h = torch.randn(2, 6, 1025) * 1.5
    b1, b2 = fir_fft(h), fir_mm(h)
    rel = ((b1 - b2).abs().max() / b1.abs().max()).item()
    print(f"parity minphase fft-vs-mm: rel {rel:.3e}")
    assert rel < 1e-4
    l1, d1 = fir_fft.forward_linear(h)
    l2, d2 = fir_mm.forward_linear(h)
    rel = ((l1 - l2).abs().max() / l1.abs().max()).item()
    print(f"parity linphase fft-vs-mm: rel {rel:.3e} (delay {d1}=={d2})")
    assert rel < 1e-4 and d1 == d2

    e = torch.randn(2, 24 * HOP)
    bb = torch.randn(2, 24, 1024) * 0.1
    y1 = ltv_ola(e, bb, HOP, "conv")
    y2 = ltv_ola(e, bb, HOP, "mm")
    rel = ((y1 - y2).abs().max() / y1.abs().max()).item()
    print(f"parity ltv_ola conv-vs-mm: rel {rel:.3e}")
    assert rel < 1e-4

    flat = torch.zeros(1, 3, 1025)
    bf = fir_fft(flat)
    print(f"flat env -> delta: b[0]={bf[0, 0, 0]:.4f} tail_max={bf[0, 0, 1:].abs().max():.2e}")

    T = 24
    t0 = T // 2
    base = {
        "f0": torch.full((1, T), 220.0),
        "h_v": (torch.randn(1, T, 1025).cumsum(-1) * 0.01),
        "d": torch.full((1, T), 0.6),
        "a": torch.rand(1, T, 4) + 0.5,
    }
    base["h_n"] = base["h_v"] - 3.0
    noise = torch.randn(1, T * HOP)
    r = LtvRenderer(nb_in=1025, k_v=1024, k_n=256, causal=True)

    def render(o):
        with torch.no_grad():
            return r(o["f0"], o["h_v"], o["h_n"], d=o["d"], a=o["a"], noise=noise)["y"]

    y0 = render(base)
    for key in ["f0", "h_v", "h_n", "d", "a"]:
        o = {k: v.clone() for k, v in base.items()}
        if key == "f0":
            o["f0"][:, t0:] = 137.0
        elif key == "d":
            o["d"][:, t0:] = 0.1
        else:
            o[key][:, t0:] = o[key][:, t0:] + torch.randn_like(o[key][:, t0:]) * 0.5
        y1 = render(o)
        pre = (y0[:, :t0 * HOP] - y1[:, :t0 * HOP]).abs().max().item()
        post = (y0[:, t0 * HOP:] - y1[:, t0 * HOP:]).abs().max().item()
        print(f"causality[{key}]: pre {pre:.3e} (must be 0), post {post:.3e}")
        assert pre == 0.0 and post > 0.0, key

    grads = {k: base[k].clone().requires_grad_(True) for k in ["h_v", "h_n", "d", "a"]}
    y = r(base["f0"], grads["h_v"], grads["h_n"], d=grads["d"], a=grads["a"], noise=noise)["y"]
    y.pow(2).mean().backward()
    for k, v in grads.items():
        g = v.grad.abs().mean().item()
        print(f"grad[{k}]: {g:.3e}")
        assert g > 0.0, k
    g = r.phi0.grad.abs().item()
    print(f"grad[phi0]: {g:.3e}")
    assert g > 0.0

    try:
        MinPhaseFIR(nb_in=1025, k=64)(torch.zeros(1, 2, 2049))
        raise SystemExit("nb_in assert missing")
    except AssertionError:
        print("nb_in shape assert OK")
    print("self-test OK")


if __name__ == "__main__":
    _self_test()
