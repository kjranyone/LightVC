"""Is the harmonic envelope lost in the ESTIMATE or in the RECONSTRUCTION?

harmonic_analysis_at() is, exactly, a heterodyne demodulation of harmonic k by
exp(-j k Phi) low-passed with a 2-period Hann -- the same operation, written as
a windowed DFT. So the estimator itself is not obviously wrong. What follows it
is: the complex envelope A_k(t) that filter produces is band-limited to +-f0/2,
and synthesis samples it at the GCIs, which arrive at f0. That is exactly the
Nyquist rate, and it is then reconstructed with LINEAR interpolation. A linear
interpolator at Nyquist is a bad reconstruction filter, and the loss is largest
where the envelope moves fastest.

This measures the whole ladder against the same noise branch:
  per-sample A_k(t)        -- the ceiling of this harmonic model
  GCI-sampled, linear      -- what ships
  GCI-sampled, cubic       -- better filter, same rate
  GCI+midpoint, linear     -- 2x the rate, same filter
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import Cache, score_one

FMAX = 2400.0


def lowpass(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    L = w.shape[-1]
    zr = torch.nn.functional.conv1d(z.real[None, None], w[None, None].flip(-1), padding=L // 2)[0, 0]
    zi = torch.nn.functional.conv1d(z.imag[None, None], w[None, None].flip(-1), padding=L // 2)[0, 0]
    return (zr + 1j * zi)[: z.shape[-1]]


def lerp(A: torch.Tensor, pos: torch.Tensor, n: int) -> torch.Tensor:
    t_g = pos.double()
    tt = torch.arange(n, dtype=torch.double)
    j = torch.searchsorted(t_g, tt).clamp(1, len(t_g) - 1)
    fr = ((tt - t_g[j - 1]) / (t_g[j] - t_g[j - 1]).clamp(min=1.0)).clamp(0, 1)
    return A[j - 1] * (1 - fr) + A[j] * fr


def cubic(A: torch.Tensor, pos: torch.Tensor, n: int) -> torch.Tensor:
    t_g = pos.double()
    tt = torch.arange(n, dtype=torch.double)
    j = torch.searchsorted(t_g, tt).clamp(1, len(t_g) - 1)
    fr = ((tt - t_g[j - 1]) / (t_g[j] - t_g[j - 1]).clamp(min=1.0)).clamp(0, 1)
    P = len(A)
    p0 = A[(j - 2).clamp(0, P - 1)]
    p1 = A[(j - 1).clamp(0, P - 1)]
    p2 = A[j.clamp(0, P - 1)]
    p3 = A[(j + 1).clamp(0, P - 1)]
    t = fr
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def main() -> None:
    c = Cache()
    rows: dict[str, list[float]] = {}

    for uid, gt in c.items:
        n = gt.shape[-1]
        y, h, nz, p = R.resynthesize(gt)
        rows.setdefault("baseline", []).append(score_one(gt, y))
        f0 = p["f0"]
        gci = p["gci"]
        f0u = R.frame_upsample(f0.double(), n).clamp(min=0.0)
        Phi = 2 * math.pi * torch.cumsum(f0u, dim=-1) / R.SR
        voiced = (f0u > 50).double()
        mvfu = R.frame_upsample(p["mvf"].double(), n)
        apu = R.frame_upsample(p["apbins"], n)
        fm = float(f0[f0 > 50].median())
        L = (2 * int(R.SR / fm)) | 1
        w = torch.hann_window(L, dtype=torch.float64)
        w = w / w.sum()

        mid = ((gci[:-1] + gci[1:]) // 2)
        dense = torch.sort(torch.cat([gci, mid]))[0]

        acc = {k: torch.zeros(n, dtype=torch.float64) for k in
               ("persample", "lin", "cub", "dense")}
        for k in range(1, int(FMAX / fm) + 2):
            fk = k * f0u
            if float((fk < FMAX).double().sum()) == 0:
                break
            car = torch.exp(1j * k * Phi)
            A = 2.0 * lowpass(gt.double() * car.conj(), w)
            gate = torch.sigmoid((mvfu - fk) / (R.GATE_SOFT * mvfu.clamp(min=1.0)))
            wgt = gate * (1.0 - R._sample_curve(apu, fk.float()).double())
            wgt = wgt * (fk < R.SR / 2 - R.SR / R.NFFT).double() * voiced
            for tag, env in (("persample", A),
                             ("lin", lerp(A[gci.clamp(0, n - 1)], gci.clamp(0, n - 1), n)),
                             ("cub", cubic(A[gci.clamp(0, n - 1)], gci.clamp(0, n - 1), n)),
                             ("dense", lerp(A[dense.clamp(0, n - 1)], dense.clamp(0, n - 1), n))):
                acc[tag] = acc[tag] + wgt * (env * car).real

        for tag, hh in acc.items():
            rows.setdefault(f"env {tag}", []).append(score_one(gt, hh.float() + nz))
            rows.setdefault(f"env {tag} (h only)", []).append(score_one(gt, hh.float()))

    print(f"\n{'variant':24s} {'PESQ':>7s} {'delta':>7s}")
    b = sum(rows["baseline"]) / len(rows["baseline"])
    for k, v in rows.items():
        m = sum(v) / len(v)
        print(f"{k:24s} {m:7.4f} {m-b:+7.4f}")


if __name__ == "__main__":
    main()
