"""diag_rr16c の記述の追加(prereg の addendum_2026-10-15_0655UTC・addendum_2026-10-15_0943UTC): 実女声 dev + dev2 で RRT16C(+ G_rel)・RRSN16C / RRTN16C(+ 自然さの手当て Z・B・H)を
RRS16C・RRTN14C・G03S10ZH・表・kNN-VC と並べ、d109 v2 の 4 軸(results/conv_p0/d109v2_rr16n{dev,dev2}.json)と合わせて本線の推論の形を拡張した choice の規則で決める。
fork(LINE_MET 等)は rr16c_verdict.py のまま。
    uv run python rr16c_nat_verdict.py --s <scratchpad> --out ../results/diag_rr16c/nat_eval_real.json
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
    sets = {D: V.ranks(S / f"rr16n{D}_lda.npz", ROOT / f"results/conv_p0/rr16n{D}_tri.json") for D in ("dev", "dev2")}
    cer = {D: json.loads((ROOT / f"results/conv_p0/rr16n{D}_cer.json").read_text())[D] for D in ("dev", "dev2")}
    rng = np.random.default_rng(0)
    pairs = [("RRT16C", "RRS16C"), ("RRTN16C", "RRSN16C"), ("RRSN16C", "KNN"), ("RRTN16C", "KNN"), ("RRS16C", "KNN"), ("RRT16C", "KNN"), ("RRTN16C", "RRTN14C"), ("RRSN16C", "RRS16C"),
             ("RRTN16C", "RRT16C"), ("RRTN16C", "G03S10ZH"), ("RRSN16C", "G03S10ZH")]
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
    cm = {c: float(np.mean([rep["cer"][D][c] for D in cer])) for c in ("RRS16C", "RRT16C", "RRSN16C", "RRTN16C", "RRTN14C", "G03S10ZH", "TABR", "KNN") if all(c in rep["cer"][D] for D in cer)}
    rep["cer_mean_both"] = {c: round(v, 4) for c, v in cm.items()}
    ch = lambda s_, t_: t_ if cm[t_] <= cm[s_] - 0.015 and all(rep[m][f"{t_}-{s_}"]["rank_diff"] <= 0.25 for m in ("LDA", "ECAPA")) else s_
    rep["choice_raw"] = ch("RRS16C", "RRT16C")
    ax = {D: json.loads((ROOT / f"results/conv_p0/d109v2_rr16n{D}.json").read_text()) for D in ("dev", "dev2")}
    rep["axes"] = {c: {D: {"axes": ax[D][f"d139:{c}"]["axes"], "fail": [k for k, v in ax[D][f"d139:{c}"]["stats"].items() if not v["pass"]]} for D in ax} for c in ("RRS16C", "RRT16C", "RRSN16C", "RRTN16C", "RRTN14C")}
    nat = [c for c in ("RRSN16C", "RRTN16C") if all(ax[D][f"d139:{c}"]["pass"] for D in ax)]
    rep["natural_pass"] = nat
    rep["choice"] = ch("RRSN16C", "RRTN16C") if len(nat) == 2 else nat[0] if nat else None
    rep["owner_line_descriptive"] = {c: "MET" if all(rep[m][f"{c}-KNN"]["rank_diff"] <= 0 and rep[m][f"{c}-KNN"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA")) else "NOT_MET" for c in ("RRS16C", "RRT16C", "RRSN16C", "RRTN16C")}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
