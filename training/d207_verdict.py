"""D207 の記述と選択(学習なし): dev + dev2(独立の中心・各 60 組)で R 系(RRS14C・RRT14C)を製品の経路の自然さ合格の構成(G03S10ZH・G05S10ZH)・表・kNN-VC と並べ、results/conv_p0/d207_prereg.yaml の fork を当てる。
差 = 前の平均順位 − 後(負 = 前が本人に近い)・クラスタ = (組, 目標)。CER は両組の平均。
    uv run python d207_verdict.py --s <scratchpad> --out ../results/conv_p0/d207.json
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
    S = Path(a.s)
    sets = {D: V.ranks(S / f"d207{D}_lda.npz", ROOT / f"results/conv_p0/d207{D}_tri.json") for D in ("dev", "dev2")}
    cer = {D: json.loads((ROOT / f"results/conv_p0/d207{D}_cer.json").read_text())[D] for D in ("dev", "dev2")}
    rng = np.random.default_rng(0)
    pairs = [("RRS14C", "G03S10ZH"), ("RRT14C", "G03S10ZH"), ("G03S10ZH", "KNN"), ("G05S10ZH", "KNN"), ("G05S10ZH", "G03S10ZH"), ("RRS14C", "KNN"), ("RRT14C", "KNN"), ("G03S10ZH", "TABR")]
    rep: dict = {}
    for m in ("LDA", "ECAPA"):
        blk: dict = {}
        for D, R in sets.items():
            blk[f"levels_{D}"] = {c: [round(float(np.mean([v == 1 for v in R[m][c].values()])), 3), round(float(np.mean(list(R[m][c].values()))), 2)] for c in sorted(R[m])}
        for p, o in pairs:
            rows = [(D, int(k.split("_t")[1]), R[m][p][k] - R[m][o][k]) for D, R in sets.items() if p in R[m] and o in R[m] for k in sorted(set(R[m][p]) & set(R[m][o]))]
            cl = np.array([f"{r[0]}:{r[1]}" for r in rows]); d = np.array([r[2] for r in rows], float)
            us = sorted(set(cl)); idx = {u: np.flatnonzero(cl == u) for u in us}
            B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
            q = np.percentile([d[ii].mean() for ii in B], [2.5, 97.5])
            blk[f"{p}-{o}"] = {"rank_diff": round(float(d.mean()), 3), "ci95": [round(float(q[0]), 3), round(float(q[1]), 3)], "n": len(d), "targets": len(us)}
        rep[m] = blk
    rep["cer"] = {D: {c: round(float(v["cer_mean"]), 4) for c, v in cer[D].items() if c not in ("per", "examples")} for D in cer}
    cm = {c: float(np.mean([rep["cer"][D][c] for D in cer])) for c in ("RRS14C", "RRT14C", "G03S10ZH", "G05S10ZH", "TABR", "KNN") if all(c in rep["cer"][D] for D in cer)}
    rep["cer_mean_both"] = {c: round(v, 4) for c, v in cm.items()}
    ahead = {c: rep["LDA"][f"{c}-G03S10ZH"]["ci95"][1] < 0 and rep["ECAPA"][f"{c}-G03S10ZH"]["rank_diff"] < 0 for c in ("RRS14C", "RRT14C")}
    rep["ahead"] = ahead
    rep["verdict"] = "R_AHEAD" if any(ahead.values()) else "NOT_AHEAD"
    rep["owner_line_descriptive"] = {c: "MET" if all(rep[m][f"{c}-KNN"]["rank_diff"] <= 0 and rep[m][f"{c}-KNN"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA")) else "NOT_MET" for c in ("RRS14C", "RRT14C", "G03S10ZH", "G05S10ZH")}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
