"""Analysis-by-synthesis residual noise (HNM-style) vs the MVF crossfade.

The shipping path splits the spectrum at MVF: harmonics below, magnitude-shaped
noise above. Everything above MVF therefore sits at the phase-blind floor
(PESQ 2.078 measured), and MVF*0.45 clamped to >=1kHz puts most of the spectrum
there. This asks the other question: render the harmonic bank over the FULL band
and let the noise branch carry only what it actually missed, |STFT(x - h)|.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import Cache, score_one


def render_harm(f0: torch.Tensor, n: int, gci: torch.Tensor, Xg: torch.Tensor,
                kmax: int = 400, mvf: torch.Tensor | None = None,
                apbins: torch.Tensor | None = None, soft: float = 0.15):
    f0u = R.frame_upsample(f0.double(), n).clamp(min=0.0)
    voiced = (f0u > 50).double()
    gi = gci.clamp(0, n - 1)
    t_g = gi.double()
    tt = torch.arange(n, dtype=torch.double)
    j = torch.searchsorted(t_g, tt).clamp(1, len(t_g) - 1)
    t0, t1 = t_g[j - 1], t_g[j]
    fr = ((tt - t0) / (t1 - t0).clamp(min=1.0)).clamp(0, 1)
    Phi = 2 * math.pi * torch.cumsum(f0u, dim=-1) / R.SR
    Phi_g = Phi[gi]
    mvfu = R.frame_upsample(mvf.double(), n) if mvf is not None else None
    apu = R.frame_upsample(apbins, n) if apbins is not None else None
    frc = fr.to(torch.complex64)

    h = torch.zeros(n, dtype=torch.float64)
    for k in range(1, kmax + 1):
        fk = k * f0u
        if (fk < R.SR / 2).sum() == 0:
            break
        Ok = Xg[k - 1] * torch.exp(-1j * k * Phi_g.to(torch.complex64))
        a = Ok.abs()
        if float(a.max()) <= 0:
            continue
        o = Ok / (a + R.EPS)
        oi = o[j - 1] * (1 - frc) + o[j] * frc
        oi = oi / (oi.abs() + R.EPS)
        ai = (a[j - 1] * (1 - fr) + a[j] * fr).double()
        w = torch.ones(n, dtype=torch.float64)
        if mvfu is not None:
            w = torch.sigmoid((mvfu - fk) / (soft * mvfu.clamp(min=1.0)))
        if apu is not None:
            w = w * (1.0 - R._sample_curve(apu, fk.float()).double())
        alive = (fk < R.SR / 2 - R.SR / R.NFFT).double()
        h = h + ai * w * alive * voiced * torch.cos(k * Phi + torch.angle(oi).double())
    return h.float()


def src_for(gci: torch.Tensor, n: int, fm: float, cyc: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed + 1)
    wn = torch.randn(n, generator=g)
    if cyc <= 0:
        return wn
    cn = R.cyclic_noise(gci, n, fm, seed=seed)
    return cyc * cn / cn.std().clamp(min=R.EPS) + (1 - cyc) * wn


def main() -> None:
    c = Cache()
    NF, NH = R.NOISE_NFFT, R.NOISE_NFFT // R.NOISE_HOPDIV
    rows = {}

    for uid, gt in c.items:
        n = gt.shape[-1]
        p = R.analyze(gt)
        f0 = p["f0"]
        fm = float(f0[f0 > 50].median())
        gci, soe = R.zff_gci(gt, fm)
        f0_at = f0[(gci // R.HOP).clamp(0, len(f0) - 1)]
        vm = f0_at > 50
        gci, f0_at = gci[vm], f0_at[vm]
        Xg = R.harmonic_analysis_at(gt, gci, f0_at)

        base, _, _, _ = R.resynthesize(gt)
        rows.setdefault("baseline v16", []).append(score_one(gt, base))

        h_gated = render_harm(f0, n, gci, Xg, mvf=p["mvf"], apbins=p["apbins"])
        rows.setdefault("harm only (gated)", []).append(score_one(gt, h_gated))

        h_full = render_harm(f0, n, gci, Xg)
        rows.setdefault("harm only (full band)", []).append(score_one(gt, h_full))

        h_fap = render_harm(f0, n, gci, Xg, apbins=p["apbins"])
        rows.setdefault("harm only (full, ap)", []).append(score_one(gt, h_fap))

        for tag, hh in (("resid full", h_full), ("resid full+ap", h_fap),
                        ("resid gated", h_gated)):
            nm = R.stft(gt - hh, NF, NH).abs()
            for cyc in (0.0, 0.5):
                src = src_for(gci, n, fm, cyc)
                y = hh + R._shaped_noise(nm, src, n, 0, NF, NH)
                rows.setdefault(f"{tag} cyc{cyc}", []).append(score_one(gt, y))

    print(f"{'variant':28s} {'PESQ':>7s}  {'delta':>7s}")
    b = sum(rows["baseline v16"]) / len(rows["baseline v16"])
    for k, v in rows.items():
        m = sum(v) / len(v)
        print(f"{k:28s} {m:7.4f}  {m-b:+7.4f}")


if __name__ == "__main__":
    main()
