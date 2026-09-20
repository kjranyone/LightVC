"""Analysis-by-synthesis correction of the per-harmonic amplitudes.

Established by rddsp_vu.py: the entire remaining PESQ gap is the MAGNITUDE of
VOICED frames below 1 kHz (+0.624 to fix; unvoiced is worth +0.041, phase
+0.319). Band totals there are already right to -0.1..-0.3 dB, so the error is
fine structure -- per harmonic, per frame -- not a level bias.

The amplitudes come from a windowed DFT of a 2-period Hann window at k*f0, an
open-loop estimate: it is smoothed over two glottal cycles, it assumes f0 is
exact, and it never checks what the rendered signal actually measures. So close
the loop -- render, compare |H| against |X| at each harmonic bin, scale, repeat.
The corrected amplitudes are still just amplitudes, the same kind of parameter
analyze() already produces, so nothing about the synthesis path or its latency
changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import Cache, score_one
from rddsp_resid import render_harm


def amp_correct(X: torch.Tensor, h: torch.Tensor, f0: torch.Tensor,
                gci: torch.Tensor, Xg: torch.Tensor, lo: float, hi: float,
                clamp: float = 4.0) -> torch.Tensor:
    H = R.stft(h)
    T = min(X.shape[-1], H.shape[-1], f0.shape[-1])
    binhz = R.SR / R.NFFT
    kmax = Xg.shape[0]
    out = Xg.clone()
    fr = (gci // R.HOP).clamp(0, T - 1)
    for i in range(len(gci)):
        t = int(fr[i])
        f = float(f0[t])
        if f <= 50:
            continue
        kk = int(min(kmax, (R.SR / 2 - 1) // f))
        ks = torch.arange(1, kk + 1, dtype=torch.float32)
        b = torch.round(ks * f / binhz).long().clamp(0, R.NB - 1)
        g = (X[b, t].abs() / (H[b, t].abs() + R.EPS)).clamp(1.0 / clamp, clamp)
        band = ((ks * f >= lo) & (ks * f < hi)).float()
        out[:kk, i] = out[:kk, i] * (g * band + (1 - band))
    return out


def main() -> None:
    c = Cache()
    rows: dict[str, list[float]] = {}

    for uid, gt in c.items:
        n = gt.shape[-1]
        y, h, nz, p = R.resynthesize(gt)
        rows.setdefault("baseline", []).append(score_one(gt, y))
        f0 = p["f0"]
        gci, Xg = p["gci"], p["Xg"]
        X = R.stft(gt)

        for lo, hi in ((0.0, 1000.0), (0.0, 22050.0)):
            cur, hh = Xg, h
            for it in (1, 2, 3):
                cur = amp_correct(X, hh, f0, gci, cur, lo, hi)
                hh = render_harm(f0, n, gci, cur, mvf=p["mvf"], apbins=p["apbins"])
                rows.setdefault(f"ampfix {hi:.0f}Hz x{it}", []).append(
                    score_one(gt, hh + nz))

    print(f"\n{'variant':22s} {'PESQ':>7s} {'delta':>7s}")
    b = sum(rows["baseline"]) / len(rows["baseline"])
    for k, v in rows.items():
        m = sum(v) / len(v)
        print(f"{k:22s} {m:7.4f} {m-b:+7.4f}")


if __name__ == "__main__":
    main()
