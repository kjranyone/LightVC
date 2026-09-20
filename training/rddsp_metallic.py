"""The ear said metallic. Measure the inter-harmonic valley.

--logfloor -60 stops penalising any bin more than 60 dB below the TARGET's peak.
In voiced speech the noise floor BETWEEN harmonics sits far below the harmonic
peaks, so a peak-relative floor removes the penalty for flattening it. A voiced
spectrum whose valleys are emptier than the reference's is the textbook cause of
a metallic / robotic timbre, and PESQ rewards matching the peaks far more than
it punishes an over-clean valley.

rddsp_distance.py already tried to measure this and was discarded as broken --
correctly, because it applied the utterance MEDIAN f0 to every frame, so the
peak/valley masks were misaligned wherever f0 moved. Fixed here: the mask is
rebuilt per frame from that frame's own f0, and only voiced frames count.

Reported for gt, the DSP core, the net trained under the original absolute
epsilon, and the net trained under the -60 dB floor. If the floored net's
peak-to-valley ratio is HIGHER than gt's, it has over-cleaned the valleys, the
ear is right, and the +0.30 PESQ was bought with a defect PESQ cannot see.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_gpu import (build, to_dev, prior_wave, feats, istft, CACHE_DIR, safe_score)
from rddsp_hf import Wavehax2D

NF, HP = 2048, 256


def ptv(x: torch.Tensor, f0: torch.Tensor, lo=300.0, hi=5000.0):
    """Harmonic peak-to-valley in dB, per voiced frame, using that frame's f0."""
    w = torch.hann_window(NF)
    S = torch.stft(x, NF, HP, NF, w, center=True, return_complex=True).abs()
    f = torch.fft.rfftfreq(NF, 1 / R.SR)
    band = (f > lo) & (f < hi)
    fb = f[band]
    T = S.shape[-1]
    # f0 is at rddsp's frame rate; resample to this STFT's frames
    idx = (torch.arange(T) * (len(f0) - 1) / max(T - 1, 1)).long().clamp(0, len(f0) - 1)
    f0f = f0[idx]
    out = []
    Sv = S[band]
    for t in range(T):
        v = float(f0f[t])
        if v <= 50:
            continue
        d = torch.remainder(fb / v, 1.0)
        d = torch.minimum(d, 1 - d)
        pk, vl = d < 0.15, d > 0.35
        if pk.sum() < 3 or vl.sum() < 3:
            continue
        p = float((Sv[pk, t] ** 2).mean())
        q = float((Sv[vl, t] ** 2).mean())
        if p <= 0 or q <= 0:
            continue
        out.append(10 * np.log10(p / q))
    return float(np.mean(out)) if out else float("nan")


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, te = build(80, 12)
    items = to_dev(te, dev)
    tags = [t for t in ("convfix", "lf60")
            if (CACHE_DIR / f"{t}_ch48_s0.pt").exists()]
    nets = {}
    for t in tags:
        ck = torch.load(CACHE_DIR / f"{t}_ch48_s0.pt", map_location=dev, weights_only=False)
        n = Wavehax2D(cin=4, ch=ck["args"]["ch"], layers=ck["args"]["layers"]).to(dev)
        n.load_state_dict(ck["net"]); n.eval()
        nets[t] = (n, ck["args"], ck["step"])

    rows = {k: [] for k in ["gt", "core"] + tags}
    ge = torch.Generator(device=dev).manual_seed(999)
    with torch.no_grad():
        for it, raw in zip(items, te):
            f0 = raw["f0"]
            n = min(it["gt"].shape[-1], it["y"].shape[-1])
            gt = it["gt"].cpu()[:n]
            rows["gt"].append(ptv(gt, f0))
            rows["core"].append(ptv(it["y"].cpu()[:n], f0))
            for t, (net, cfg, _) in nets.items():
                pw = prior_wave(it, ge, dev, cfg.get("pnoise", 0.05),
                                cfg.get("prior", "core"))
                f, P = feats(it, pw)
                o = net(f[None])[0]
                S = (torch.complex(o[0], o[1]) if cfg.get("direct") else
                     torch.complex(P.real + o[0], P.imag + o[1]))
                rows[t].append(ptv(istft(S, it["gt"].shape[-1]).cpu()[:n], f0))

    g = np.array(rows["gt"])
    print(f"\n  {len(g)} held-out speakers, voiced frames, 300-5000 Hz, per-frame f0.")
    print("  Harmonic peak-to-valley in dB. HIGHER than gt = valleys emptied =")
    print("  the harmonic structure is cleaner than real speech = metallic.\n")
    print(f"  {'system':<12} {'P/V dB':>9} {'vs gt':>9} {'95% t-CI':>11}")
    from scipy import stats
    for k in rows:
        v = np.array(rows[k])
        d = v - g
        ci = (stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
              if k != "gt" else 0.0)
        step = f" (step {nets[k][2]})" if k in nets else ""
        print(f"  {k:<12} {v.mean():9.2f} {d.mean():+9.2f} {ci:11.2f}{step}")
    print("\n  A positive 'vs gt' is over-cleaning. PESQ rewards matching the peaks")
    print("  and barely punishes an empty valley, so it cannot arbitrate this.")


if __name__ == "__main__":
    main()
