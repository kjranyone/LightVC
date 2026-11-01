"""diag_rr16c の記述の追加(prereg の addendum_2026-10-15_0655UTC・D205 の choice_at_rr16c): 実女声 dev + dev2 で RRT16C(表 + 話者の分 + G_rel 8 人・c0 は話者の分だけ)を
RRS16C・RRT14C・RRS14C・表・kNN-VC と並べ、本線の推論の形を choice_at_rr16c の規則で決める。fork(LINE_MET 等)は rr16c_verdict.py のまま。
    uv run python rr16c_rrt_verdict.py --s <scratchpad> --out ../results/diag_rr16c/rrt_eval_real.json
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
    sets = {D: V.ranks(S / f"rr16t{D}_lda.npz", ROOT / f"results/conv_p0/rr16t{D}_tri.json") for D in ("dev", "dev2")}
    cer = {D: json.loads((ROOT / f"results/conv_p0/rr16t{D}_cer.json").read_text())[D] for D in ("dev", "dev2")}
    rng = np.random.default_rng(0)
    pairs = [("RRT16C", "RRS16C"), ("RRT16C", "KNN"), ("RRT16C", "RRT14C"), ("RRT16C", "TABR"), ("RRS16C", "KNN"), ("RRT14C", "RRS14C")]
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
    cm = {c: float(np.mean([rep["cer"][D][c] for D in cer])) for c in ("RRS14C", "RRT14C", "RRS16C", "RRT16C", "TABR") if all(c in rep["cer"][D] for D in cer)}
    rep["cer_mean_both"] = {c: round(v, 4) for c, v in cm.items()}
    dT = [rep[m]["RRT16C-RRS16C"]["rank_diff"] for m in ("LDA", "ECAPA")]
    rep["choice"] = "RRT16C" if cm["RRT16C"] <= cm["RRS16C"] - 0.015 and all(d <= 0.25 for d in dT) else "RRS16C"
    rep["owner_line_descriptive"] = {c: "MET" if all(rep[m][f"{c}-KNN"]["rank_diff"] <= 0 and rep[m][f"{c}-KNN"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA")) else "NOT_MET" for c in ("RRT16C", "RRS16C")}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
