"""D208 の判定(学習なし): dev + dev2 で自然さの手当てを載せた R 系(RRSN14C・RRTN14C)を手当て前(RRS14C・RRT14C)・G03S10ZH・表・kNN-VC と並べ、d109 の 4 軸(results/conv_p0/d109v2_d208{dev,dev2}.json)と合わせて results/conv_p0/d208_prereg.yaml の fork を当てる。
差 = 前の平均順位 − 後(負 = 前が本人に近い)・クラスタ = (組, 目標)。CER は両組の平均。
    uv run python d208_verdict.py --s <scratchpad> --out ../results/conv_p0/d208.json
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
    sets = {D: V.ranks(S / f"d208{D}_lda.npz", ROOT / f"results/conv_p0/d208{D}_tri.json") for D in ("dev", "dev2")}
    cer = {D: json.loads((ROOT / f"results/conv_p0/d208{D}_cer.json").read_text())[D] for D in ("dev", "dev2")}
    rng = np.random.default_rng(0)
    pairs = [("RRSN14C", "RRS14C"), ("RRTN14C", "RRT14C"), ("RRSN14C", "G03S10ZH"), ("RRTN14C", "G03S10ZH"), ("RRSN14C", "KNN"), ("RRTN14C", "KNN"), ("RRS14C", "G03S10ZH")]
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
    cm = {c: float(np.mean([rep["cer"][D][c] for D in cer])) for c in ("RRS14C", "RRT14C", "RRSN14C", "RRTN14C", "G03S10ZH", "TABR") if all(c in rep["cer"][D] for D in cer)}
    rep["cer_mean_both"] = {c: round(v, 4) for c, v in cm.items()}
    ax = {D: json.loads((ROOT / f"results/conv_p0/d109v2_d208{D}.json").read_text()) for D in ("dev", "dev2")}
    nfail = lambda D, c: sum(1 for v in ax[D][f"d139:{c}"]["stats"].values() if not v["pass"])
    rep["axes"] = {c: {D: {"axes": ax[D][f"d139:{c}"]["axes"], "n_fail_stats": nfail(D, c), "fail": [k for k, v in ax[D][f"d139:{c}"]["stats"].items() if not v["pass"]]} for D in ax} for c in ("RRS14C", "RRT14C", "RRSN14C", "RRTN14C")}
    ok, part = {}, {}
    for x, u in (("RRSN14C", "RRS14C"), ("RRTN14C", "RRT14C")):
        ax_ok = all(ax[D][f"d139:{x}"]["pass"] for D in ax)
        id_ok = all(rep[m][f"{x}-{u}"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA"))
        ok[x] = ax_ok and id_ok and cm[x] - cm[u] <= 0.01
        part[x] = all(nfail(D, x) < nfail(D, u) for D in ax)
    rep["pass"] = ok; rep["fewer_fail"] = part
    rep["verdict"] = "NAT_OK" if any(ok.values()) else "NAT_PART" if any(part.values()) else "NAT_NO"
    rep["owner_line_descriptive"] = {c: "MET" if all(rep[m][f"{c}-KNN"]["rank_diff"] <= 0 and rep[m][f"{c}-KNN"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA")) else "NOT_MET" for c in ("RRSN14C", "RRTN14C")}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
