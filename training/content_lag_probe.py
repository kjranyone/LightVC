"""因果な内容符号器 E2 と ContentVec の差は「遅れ」か「質」か(converter.md C1 の設計のための測定・学習なし)。

E2 の出力(ContentVec 768 次元の予測)を ContentVec と同じコードブック(評価話者を含まない VCTK 話者・k-means)に割り当て、
200fps のフレーム t の E2 の単位と、t − s の ContentVec の単位の一致率を s = −4..12(5ms 刻み)で測る。
最大になる s = E2 の実効の遅れ、その一致率 = 遅れを補っても残る質の差。対照: ContentVec を自分自身の s ずらしと比べた一致率(単位の持続の長さ)。
材料 = conv_c0 の R0(男声・音域移動・225 組)と T0(女声 45)。

    uv run python content_lag_probe.py --work <scratchpad>/c0_vctk --out ../results/conv_c0/content_lag.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=200)
    a = ap.parse_args()
    import torch
    import artic_g2_unit as U
    sys.path.insert(0, a.ladder)
    import gen as G
    from train_dec2 import load48
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cv, e2 = U.CV(dev), U.E2(dev)
    J = json.loads((Path(a.ladder) / "jobs.json").read_text())
    evals = set(J["fems"]) | set(J["males"])
    others = sorted(p.name for p in (VC / "wav48").iterdir() if p.name not in evals)
    pool = [cv(G.trim(load48(str(VC / f"wav48/{s}/{s}_{u:03d}.wav")).astype(np.float64))) for s in others for u in range(41, 52) if (VC / f"wav48/{s}/{s}_{u:03d}.wav").exists()]
    C = U.kmeans(np.concatenate(pool), a.k, dev)
    shifts = list(range(-4, 13))
    files = sorted((Path(a.work) / "sig").glob("*__R0.npz")) + sorted((Path(a.work) / "sig").glob("*__T0.npz"))
    acc = {"e2_vs_cv": {s: [] for s in shifts}, "cv_vs_cv": {s: [] for s in shifts}}
    cosr = []
    for f in files:
        x = np.load(f)["x"].astype(np.float64)
        T = int(len(x) / 240)
        t = np.arange(T) * 240 / 48000
        hc = cv(x)
        he = e2(x)
        uc = (hc @ C.T).argmax(1)[np.clip(np.round((t - 0.0125) / 0.02).astype(int), 0, len(hc) - 1)]
        je = np.clip(np.floor(t * 44100 / 256).astype(int), 0, len(he) - 1)
        ue = (he @ C.T).argmax(1)[je]
        hcf = hc[np.clip(np.round((t - 0.0125) / 0.02).astype(int), 0, len(hc) - 1)]
        for s in shifts:
            if s >= 0:
                a_, b_, c_ = ue[s:], uc[:T - s], uc[s:]
            else:
                a_, b_, c_ = ue[:T + s], uc[-s:], uc[:T + s]
            acc["e2_vs_cv"][s].append(float((a_ == b_).mean()))
            acc["cv_vs_cv"][s].append(float((c_ == (uc[:T - s] if s >= 0 else uc[-s:])).mean()))
        best = max(shifts, key=lambda s: np.mean(acc["e2_vs_cv"][s][-1:]))
        cosr.append(float((he[je] * hcf).sum(1).mean()))
    rep = {"n_files": len(files), "k": a.k, "shift_ms": {s: s * 5 for s in shifts},
           "agree_e2_vs_cv_lagged": {s: round(float(np.mean(v)), 4) for s, v in acc["e2_vs_cv"].items()},
           "agree_cv_vs_cv_shift": {s: round(float(np.mean(v)), 4) for s, v in acc["cv_vs_cv"].items()},
           "cos_e2_cv_same_time": round(float(np.mean(cosr)), 4)}
    b = max(shifts, key=lambda s: rep["agree_e2_vs_cv_lagged"][s])
    rep["best_shift_frames"] = b
    rep["best_agree"] = rep["agree_e2_vs_cv_lagged"][b]
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
