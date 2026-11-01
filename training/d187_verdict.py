"""D187(学習なし・判定): TTS のクローンの実現(c0 + 動き)を操作者の時間へ写した XCDZ を目標本人の実現 XCD・表だけ TAB0 と独立の中心で並べ、
results/conv_p0/d187_prereg.yaml の fork で判定する。差 = 前の平均順位 − 後(負 = 前が本人に近い)。
    uv run python d187_verdict.py --s <scratchpad> --out ../results/conv_p0/d187.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import d181_verdict as V

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = V.ranks(Path(a.s) / "d187_lda.npz", ROOT / "results/conv_p0/d187_tri.json")
    pairs = [("XCDZ", "XCD"), ("XCDZ", "TAB0"), ("XC0Z", "XC0"), ("XC0Z", "TAB0"), ("XCD", "TAB0"), ("XCDZ", "CONV")]
    rng = np.random.default_rng(0)
    rep: dict = {}
    for m in ("LDA", "ECAPA"):
        conds = sorted(R[m])
        keys = sorted(set.intersection(*(set(R[m][c]) for c in conds)))
        tg = np.array([int(k.split("_t")[1]) for k in keys]); us = sorted(set(tg)); idx = {u: np.flatnonzero(tg == u) for u in us}
        B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
        X = {c: np.array([R[m][c][k] for k in keys], float) for c in conds}
        blk: dict = {"n": len(keys), "targets": len(us)} | {c: {"top1": round(float((x == 1).mean()), 3), "mean_rank": round(float(x.mean()), 2)} for c, x in X.items()}
        for p, o in pairs:
            d = X[p] - X[o]
            q = np.percentile([d[ii].mean() for ii in B], [2.5, 97.5])
            blk[f"{p}-{o}"] = {"rank_diff": round(float(d.mean()), 3), "ci95": [round(float(q[0]), 3), round(float(q[1]), 3)]}
        rep[m] = blk
    L = rep["LDA"]
    if L["XCDZ-XCD"]["ci95"][1] <= 1.0 and L["XCDZ-TAB0"]["ci95"][1] < 0:
        v = "CLONE_OK"
    elif L["XCDZ-TAB0"]["ci95"][1] < 0:
        v = "CLONE_WEAK"
    else:
        v = "CLONE_NONE"
    rep["verdict"] = v
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
