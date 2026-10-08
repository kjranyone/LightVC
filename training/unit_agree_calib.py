"""内容表現の性別不変性の物差し: 同じ文(VCTK 共通文 3〜6)の 2 話者を DTW で対応させ、ContentVec k-means の単位の一致率を
男女(MF)・女女(FF)・男男(MM)で並べる。FF/MM が同性の上限、MF との差が性別による不一致。差し替えラダー(artic_g2 と同じ)の S0/T0 を使う。

    uv run python unit_agree_calib.py --ladder <scratchpad>/r4_spk --out ../results/artic_inv/unit_agree.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))
import artic_g2_unit as U


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[50, 200])
    a = ap.parse_args()
    lad = Path(a.ladder)
    sys.path.insert(0, str(lad))
    import gen as G
    from train_dec2 import load48
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    J = json.loads((lad / "jobs.json").read_text())
    cv = U.CV(dev)
    pool = []
    for s in J["fems"] + J["males"]:
        for u in range(41, 52):
            if U.utt(s, u).exists():
                pool.append(cv(G.trim(load48(str(U.utt(s, u))).astype(np.float64))))
    books = {k: U.kmeans(np.concatenate(pool), k, dev) for k in a.k}
    res: dict = {}
    for tag in ("MF", "FF", "MM"):
        jobs = [j for j in J["jobs"] if j[0] == tag]
        agree = {k: [] for k in a.k}
        for _, src, tgt, _ in jobs:
            xs, _ = sf.read(lad / "wav" / f"{tag}__{src}__{tgt}__S0.wav")
            xt, _ = sf.read(lad / "wav" / f"{tag}__{src}__{tgt}__T0.wav")
            xs, xt = xs.astype(np.float64), xt.astype(np.float64)
            Ts, Tt = G.stft(xs).shape[1], G.stft(xt).shape[1]
            pos = G.align_map(xs, xt, Ts, Tt)
            jo = np.clip(np.round(pos).astype(int), 0, Tt - 1)
            fs, ft = cv(xs), cv(xt)
            for k, C in books.items():
                us, ut = U.frame_units(fs, C, Ts), U.frame_units(ft, C, Tt)
                agree[k].append(float((us == ut[jo]).mean()))
        res[tag] = {"n": len(jobs), **{f"k{k}": round(float(np.mean(v)), 3) for k, v in agree.items()}}
        print(tag, res[tag], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
