"""D585: d4b_lda_verifier の --dump(ファイルごとの全目標への得点)から、同じ組(pNNNN_tTTT)の版どうしの目標の順位の差を目標でまとめたブートストラップで。
    uv run python d585_rank_pairs.py --dumps <dev2 の npz> <dev3 の npz> --pairs OURS5R,REFK5 ... --out ../results/conv_p0/d585_pairs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rk: dict = {}
    for si, f in enumerate(a.dumps):
        d = np.load(f)
        for nm, pi, ti, c, sc in zip(d["name"], d["pi"], d["ti"], d["cond"], d["score"]):
            rk.setdefault(str(c), {})[(si, int(pi), int(ti))] = int((sc > sc[int(ti)]).sum()) + 1
    rng = np.random.default_rng(0)
    rep = {"mean_rank": {c: round(float(np.mean(list(v.values()))), 2) for c, v in sorted(rk.items())}, "paired": {}}
    for pr in a.pairs:
        x, y = pr.split(",")
        keys = sorted(set(rk[x]) & set(rk[y]))
        d = np.array([rk[x][k] - rk[y][k] for k in keys], float)
        tg = np.array([f"{k[0]}_{k[2]}" for k in keys])
        us = sorted(set(tg))
        idx = {u: np.flatnonzero(tg == u) for u in us}
        bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
        rep["paired"][f"{x}-{y}"] = {"n": len(keys), "diff": round(float(d.mean()), 2), "ci95": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["mean_rank"]))
    for k, v in rep["paired"].items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
