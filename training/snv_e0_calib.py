"""E0 の時刻のずれの物差しの較正: Y を既知の量ずらしたもの(±20 ms → lag ∓4 / ±4)と、WORLD で高さ −12 半音・包絡 0.85 倍に伸縮した別の声(時刻は構成上一致 → lag 0)。
    uv run python snv_e0_calib.py
"""
from __future__ import annotations

import json

import numpy as np
import pyworld as pw
import soxr
import torch

import c1_content as CC
import nvoc as N
import snv_e0_measure as M


def world_voice(x: np.ndarray) -> np.ndarray:
    x16 = soxr.resample(x, N.SR, 16000).astype(np.float64)
    f0, t = pw.harvest(x16, 16000, f0_floor=60.0, f0_ceil=900.0, frame_period=5.0)
    sp = pw.cheaptrick(x16, f0, t, 16000)
    ap = pw.d4c(x16, f0, t, 16000)
    n = sp.shape[1]
    src = np.clip(np.arange(n) / 0.85, 0, n - 1)
    sp2 = np.stack([np.interp(src, np.arange(n), r) for r in sp])
    ap2 = np.stack([np.interp(src, np.arange(n), r) for r in ap])
    y = pw.synthesize(f0 * 0.5, sp2, ap2, 16000, 5.0)
    return soxr.resample(y.astype(np.float32), 16000, N.SR).astype(np.float32)


def main() -> int:
    dev = "cuda"
    st1 = torch.load(M.ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"])
    c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    sh = int(0.02 * N.SR)
    res = {"late20": [], "early20": [], "world": []}
    for p in sorted(M.WAV.glob("i*_Y.wav"))[:16]:
        y = M.load(p)
        T = len(y)
        cands = {"late20": np.concatenate([np.zeros(sh, np.float32), y])[:T], "early20": np.concatenate([y[sh:], np.zeros(sh, np.float32)]),
                 "world": world_voice(y)[:T]}
        qy = M.post(y, c1n, mfc, dev)
        ey = M.energy_db(y, len(qy))
        act = ey > ey.max() - 35
        for k, x in cands.items():
            x = np.pad(x, (0, max(0, T - len(x))))
            qx = M.post(x, c1n, mfc, dev)
            Tq = min(len(qy), len(qx))
            res[k] += [w["lag"] for w in M.local_lags(qy[:Tq], qx[:Tq], act[:Tq])]
    out = {k: {"n": len(v), "median_lag_frames": float(np.median(v)), "frac_within_1_of_expected": float(np.mean(np.abs(np.array(v) - e) <= 1)),
               "frac_lag_ok_0": float(np.mean(np.abs(np.array(v)) <= M.LAG_OK))}
           for (k, v), e in zip(res.items(), (4, -4, 0))}
    (M.OUT / "e0_calib.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
