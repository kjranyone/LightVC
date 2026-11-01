"""diag_rr17c の判定(学習なし・results/diag_rr17c/prereg.yaml): fork はクローンの未知の話者(d193 の verdict_eval.json の diag_rr17c − diag_rr16c)。
記述 = 実女声 dev・dev2・dev3(90 目標・各組 1 つの実行 rr17{D})の順位の差((組, 目標) のクラスタのブートストラップ 5000 回)・CER・d109 v2 の 4 軸(d109v2_rr17{D})。
    uv run python rr17c_verdict.py --s <scratchpad> --out ../results/diag_rr17c/verdict_eval_real.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import d181_verdict as V

ROOT = Path(__file__).resolve().parent.parent
SETS = ("dev", "dev2", "dev3")
PAIRS = [("RRS17C", "RRS16C"), ("RRT17C", "RRT16C"), ("RRSN17C", "RRSN16C"), ("RRTN17C", "RRTN16C"), ("RRTN17C", "RRTN14C"),
         ("RRS17C", "KNN"), ("RRT17C", "KNN"), ("RRSN17C", "KNN"), ("RRTN17C", "KNN"), ("RRS16C", "KNN"), ("RRTN16C", "KNN"), ("RRTN14C", "KNN"),
         ("RRTN17C", "G03S10ZH"), ("RRS17C", "TABR")]
FORMS = ("RRS17C", "RRT17C", "RRSN17C", "RRTN17C", "RRTN16C", "RRTN14C")


def ci(rows: list, rng: np.random.Generator) -> dict:
    cl = np.array([f"{r[0]}:{r[1]}" for r in rows]); d = np.array([r[2] for r in rows], float)
    us = sorted(set(cl)); idx = {u: np.flatnonzero(cl == u) for u in us}
    B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
    q = np.percentile([d[ii].mean() for ii in B], [2.5, 97.5])
    return {"rank_diff": round(float(d.mean()), 3), "ci95": [round(float(q[0]), 3), round(float(q[1]), 3)], "n": len(d), "targets": len(us)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.s)
    cl = json.loads((ROOT / "results/diag_rr17c/verdict_eval.json").read_text())["diag_rr17c_minus_diag_rr16c"]["unseen_all"]
    up = cl["gain_mem_ci95"][0] > 0 and cl["gain_mem"] >= 0.025 and cl["dcorr_mem"] >= 0.020
    rep: dict = {"clone_unseen_minus_rr16c": cl, "verdict": "STEPS_UP" if up else "STEPS_FLAT"}
    runs = {D: V.ranks(S / f"rr17{D}_lda.npz", ROOT / f"results/conv_p0/rr17{D}_tri.json") for D in SETS}
    for m in ("LDA", "ECAPA"):
        blk: dict = {f"levels_{D}": {c: [round(float(np.mean([v == 1 for v in r.values()])), 3), round(float(np.mean(list(r.values()))), 2)] for c, r in sorted(runs[D][m].items())} for D in SETS}
        for p, o in PAIRS:
            rows = {D: [(D, int(k.split("_t")[1]), R[m][p][k] - R[m][o][k]) for k in sorted(set(R[m][p]) & set(R[m][o]))] for D, R in runs.items() if p in R[m] and o in R[m]}
            rng = np.random.default_rng(0)
            blk[f"{p}-{o}"] = {"pooled90": ci(sum(rows.values(), []), rng), "dev_dev2": ci(rows["dev"] + rows["dev2"], rng), "dev3": ci(rows["dev3"], rng)}
        rep[m] = blk
    cer = {D: json.loads((ROOT / f"results/conv_p0/rr17{D}_cer.json").read_text())[D] for D in SETS}
    rep["cer"] = {D: {c: round(float(v["cer_mean"]), 4) for c, v in cer[D].items() if c not in ("per", "examples")} for D in SETS}
    rep["cer_mean_3"] = {c: round(float(np.mean([rep["cer"][D][c] for D in SETS])), 4) for c in rep["cer"]["dev"] if all(c in rep["cer"][D] for D in SETS)}
    ax = {D: json.loads((ROOT / f"results/conv_p0/d109v2_rr17{D}.json").read_text()) for D in SETS}
    rep["axes"] = {c: {D: {"pass": ax[D][f"d139:{c}"]["pass"], "fail": [k for k, v in ax[D][f"d139:{c}"]["stats"].items() if not v["pass"]],
                           "shimmer": ax[D][f"d139:{c}"]["stats"]["shimmer"]["median"], "lmstd": ax[D][f"d139:{c}"]["stats"]["lmstd"]["median"]} for D in SETS} for c in FORMS}
    rep["owner_line_descriptive_pooled90"] = {p: "MET" if all(rep[m][f"{p}-KNN"]["pooled90"]["rank_diff"] <= 0 and rep[m][f"{p}-KNN"]["pooled90"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA")) else "NOT_MET"
                                              for p, o in PAIRS if o == "KNN"}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
