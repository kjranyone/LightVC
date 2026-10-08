"""高い声の層別の差の帰属(学習なし): 教師の f0(harvest)が高い声で外れていて、倍音の位置がずれて logmel が悪化していないか。
元音声のスペクトルに倍音の櫛(k·f0·(1+δ)・k ≤ 5kHz)を当て、フレームごとに δ ∈ [−4%, +4%](0.25% 刻み)で櫛の対数振幅の和が最大の δ を採る(診断専用の非因果・中心窓)。
補正した f0 で写し合成し直し、低い声・高い声の群ごとに logmel と PESQ_h を比べる。補正量(cent)の分布も出す。
    uv run python f0refine_probe.py --ckpt <ema.pt> --env_smooth 0.25 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_nvoc as E
import nvoc as N
import rvoc as R
import train_rvoc as TR

NFFT = 4096


def refine(x: np.ndarray, f0: np.ndarray, dev: str) -> np.ndarray:
    n = len(f0)
    xt = torch.from_numpy(x).to(dev)
    xp = torch.nn.functional.pad(xt, (NFFT // 2, NFFT // 2))
    win = torch.hann_window(2048, device=dev)
    idx = torch.arange(n, device=dev) * N.HOP + NFFT // 2
    seg = xp[idx[:, None] + torch.arange(-1024, 1024, device=dev)[None]] * win
    S = torch.fft.rfft(seg, NFFT).abs().clamp(min=1e-7).log()
    df = N.SR / NFFT
    deltas = torch.arange(-16, 17, device=dev) * 0.0025
    f = torch.from_numpy(f0).to(dev)
    sc = torch.zeros(n, len(deltas), device=dev)
    for j, d in enumerate(deltas):
        ff = f * (1 + d)
        K = int(5000 // max(float(f[f > 0].min()), 60.0)) if (f > 0).any() else 1
        k = torch.arange(1, K + 1, device=dev)[None].float()
        fr = (ff[:, None] * k) / df
        ok = (fr < NFFT // 2 - 1) & (ff[:, None] * k < 5000)
        lo = fr.floor().long().clamp(0, NFFT // 2 - 1)
        w = fr - lo
        v = S.gather(1, lo) * (1 - w) + S.gather(1, (lo + 1).clamp(max=NFFT // 2)) * w
        sc[:, j] = (v * ok).sum(1) / ok.sum(1).clamp(min=1)
    best = deltas[sc.argmax(1)]
    out = (f * (1 + best)).cpu().numpy()
    out[f0 <= 0] = 0
    return out.astype(np.float32), (1200 * np.log2(1 + best.cpu().numpy()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env_smooth", type=float, default=None, help="省略時は ckpt の env_smooth(無ければ 0)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    g = R.RVoc(ch=st["cfg"]["ch"], kernels=tuple(st["cfg"]["kernels"]), dils=tuple(st["cfg"]["dils"]), d_cond=st["cfg"]["d_cond"]).to(dev)
    g.load_state_dict(st["ema"])
    g.eval()
    fr = TR.Front("pae", TR.ckpt_env_smooth(st, a.env_smooth)).to(dev)
    rows = []
    with torch.no_grad():
        for it in TR.held(dev):
            f0 = it["f0_h"]
            v = f0[f0 > 0]
            fref, cent = refine(it["x"], f0, dev)
            m0 = E.metrics(TR.render(g, fr, it, f0, dev), it["x"], N.DELAY, dev)
            m1 = E.metrics(TR.render(g, fr, it, fref, dev), it["x"], N.DELAY, dev)
            rows.append((float(np.median(v)), m0, m1, float(np.median(np.abs(cent[f0 > 0])))))
    rep = {}
    for name, sel in (("low_lt450", lambda f: f < 450), ("high_ge490", lambda f: f >= 490)):
        r = [x for x in rows if sel(x[0])]
        rep[name] = {"n": len(r), "harvest": {k: round(float(np.mean([x[1][k] for x in r])), 4) for k in ("logmel", "pesq")},
                     "refined": {k: round(float(np.mean([x[2][k] for x in r])), 4) for k in ("logmel", "pesq")},
                     "median_abs_correction_cent": round(float(np.mean([x[3] for x in r])), 1)}
    print(json.dumps(rep))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
