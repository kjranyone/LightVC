"""D182(学習なし・判定): 実女声の擬似パラレル(D165 の 93 組・独立の中心)で目標本人の f0 の実現が c0 + 動きの上に足す同一性。results/conv_p0/d182_prereg.yaml の fork。
利得 = 比べる側の平均順位 − 処置の平均順位(正 = 処置が本人に近い)。目標単位のブートストラップ 5000 回。
    uv run python d182_verdict.py --s <scratchpad> --out ../results/conv_p0/d182.json
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
    R = V.ranks(Path(a.s) / "d182_lda.npz", ROOT / "results/conv_p0/d182_tri.json")
    pairs = [("XCDF", "XCD"), ("XF0", "CONV"), ("XCDFP", "XCDF"), ("XCD", "CONV"), ("ASELF", "XCDFP"), ("TAB0", "CONV")]
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
            g = X[o] - X[p]
            q = np.percentile([g[ii].mean() for ii in B], [2.5, 97.5])
            blk[f"{p}_over_{o}"] = {"rank_gain": round(float(g.mean()), 3), "ci95": [round(float(q[0]), 3), round(float(q[1]), 3)]}
        rep[m] = blk
    main_ = rep["LDA"]["XCDF_over_XCD"]
    rep["verdict"] = "PROSODY" if main_["ci95"][0] > 0 and rep["ECAPA"]["XCDF_over_XCD"]["rank_gain"] > 0 else "NO_PROSODY"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
