"""diag_rr16c の記述の追加(prereg の addendum_2026-10-15_1211UTC・D211): dev3 で作った同じ出力を dev3 単独と dev + dev2 + dev3 の 90 目標のまとめで並べる(記述のみ・fork と choice は変えない)。
dev / dev2 は各組でその対を両方含む実行(rr16n、無ければ rr16)・dev3 は 1 つの実行(rr16d3)。(組, 目標) のクラスタのブートストラップ 5000 回。
    uv run python rr16c_dev3_verdict.py --s <scratchpad> --out ../results/diag_rr16c/dev3_eval_real.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import d181_verdict as V

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [("RRS16C", "KNN"), ("RRT16C", "KNN"), ("RRSN16C", "KNN"), ("RRTN16C", "KNN"), ("RRTN14C", "KNN"), ("RRS16C", "RRS14C"), ("RRTN16C", "RRTN14C"), ("RRTN16C", "G03S10ZH")]
FORMS = ("RRS16C", "RRT16C", "RRSN16C", "RRTN16C", "RRSN14C", "RRTN14C")


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
    runs = {D: [V.ranks(S / f"{t}{D}_lda.npz", ROOT / f"results/conv_p0/{t}{D}_tri.json") for t in ("rr16n", "rr16")] for D in ("dev", "dev2")}
    runs["dev3"] = [V.ranks(S / "rr16d3_lda.npz", ROOT / "results/conv_p0/rr16d3_tri.json")]
    rep: dict = {}
    for m in ("LDA", "ECAPA"):
        blk: dict = {"levels_dev3": {c: [round(float(np.mean([v == 1 for v in r.values()])), 3), round(float(np.mean(list(r.values()))), 2)] for c, r in sorted(runs["dev3"][0][m].items())}}
        for p, o in PAIRS:
            rows = {}
            for D, rs in runs.items():
                R = next((x[m] for x in rs if p in x[m] and o in x[m]), None)
                if R is None:
                    continue
                rows[D] = [(D, int(k.split("_t")[1]), R[p][k] - R[o][k]) for k in sorted(set(R[p]) & set(R[o]))]
            rng = np.random.default_rng(0)
            blk[f"{p}-{o}"] = {"dev3": ci(rows["dev3"], rng), "pooled90": ci(sum(rows.values(), []), rng) if len(rows) == 3 else None, "sets": sorted(rows)}
        rep[m] = blk
    cer = json.loads((ROOT / "results/conv_p0/rr16d3_cer.json").read_text())["dev3"]
    rep["cer_dev3"] = {c: round(float(v["cer_mean"]), 4) for c, v in cer.items() if c not in ("per", "examples")}
    ax = json.loads((ROOT / "results/conv_p0/d109v2_rr16ndev3.json").read_text())
    rep["axes_dev3"] = {c: {"pass": ax[f"d139:{c}"]["pass"], "axes": ax[f"d139:{c}"]["axes"], "fail": [k for k, v in ax[f"d139:{c}"]["stats"].items() if not v["pass"]]} for c in FORMS}
    rep["owner_line_descriptive"] = {sc: {p: "MET" if all(rep[m][f"{p}-KNN"][sc]["rank_diff"] <= 0 and rep[m][f"{p}-KNN"][sc]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA")) else "NOT_MET"
                                          for p, o in PAIRS if o == "KNN"} for sc in ("dev3", "pooled90")}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
