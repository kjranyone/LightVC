"""D200 の判定(学習なし・帰属): dev + dev2(独立の中心・各 60 組)で RRX(rr8c の一般の分 + rr14c の話者の分)・RRG8C を RRS14C・RR14C・表・kNN-VC と並べ、results/conv_p0/d200_prereg.yaml の fork を当てる。
差 = 前の平均順位 − 後(負 = 前が本人に近い)・クラスタ = (組, 目標)。CER は両組の平均。
    uv run python d200_verdict.py --s <scratchpad> --out ../results/conv_p0/d200.json
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
    sets = {D: V.ranks(S / f"d200{D}_lda.npz", ROOT / f"results/conv_p0/d200{D}_tri.json") for D in ("dev", "dev2")}
    cer = {D: json.loads((ROOT / f"results/conv_p0/d200{D}_cer.json").read_text())[D] for D in ("dev", "dev2")}
    rng = np.random.default_rng(0)
    pairs = [("RRX", "RRS14C"), ("RRG8C", "TABR"), ("RRX", "TABR"), ("RRX", "KNN"), ("RRX", "RR14C"), ("RRG8C", "RRG14C")]
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
    cm = {c: float(np.mean([rep["cer"][D][c] for D in cer])) for c in ("RR14C", "RRS14C", "RRG14C", "RRG8C", "RRX", "TABR") if all(c in rep["cer"][D] for D in cer)}
    rep["cer_mean_both"] = {c: round(v, 4) for c, v in cm.items()}
    L, E = rep["LDA"]["RRX-RRS14C"], rep["ECAPA"]["RRX-RRS14C"]
    rep["verdict"] = "REAL_GENERIC_OK" if cm["RRX"] <= cm["RRS14C"] - 0.015 and L["ci95"][1] <= 1.0 and E["ci95"][1] <= 1.0 else "NOT_OK"
    K = [rep[m]["RRX-KNN"] for m in ("LDA", "ECAPA")]
    rep["owner_line_descriptive"] = "MET" if all(k["rank_diff"] <= 0 and k["ci95"][1] <= 1.0 for k in K) else "NOT_MET"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
