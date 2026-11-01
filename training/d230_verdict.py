"""D230 の判定(results/conv_p0/d230_prereg.yaml): 声域の中心(REG)と独立の中心(IND)で、声の高さだけを移した対照(PSHIFT・PSHIFTW)の − KNN の順位差を RRQSDC19L・G03S10ZH と並べる。
    uv run python d230_verdict.py --s <scratchpad> --root <dev の親> --out ../results/conv_p0/d230.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import d181_verdict as V
from rr17c_verdict import ci

ROOT = Path(__file__).resolve().parent.parent
SETS = ("dev", "dev2", "dev3")
FORMS = ("PSHIFT", "PSHIFTW", "RRQSDC19L", "G03S10ZH")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S, RT = Path(a.s), Path(a.root)
    ok = {D: {m["ti"] for m in json.loads((RT / D / "cen_reg/meta.json").read_text()) if m["ok"]} for D in SETS}
    rep: dict = {}
    for ruler in ("REG", "IND"):
        R = {D: V.ranks(S / f"d230{ruler}{D}_lda.npz", ROOT / f"results/conv_p0/d230{ruler}{D}_tri.json") for D in SETS}
        for m in ("LDA", "ECAPA"):
            blk = {"levels_ok": {c: round(float(np.mean([R[D][m][c][k] for D in SETS for k in R[D][m][c] if int(k.split("_t")[1]) in ok[D]])), 2) for c in FORMS + ("KNN",)}}
            for c in FORMS:
                rows = [(D, int(k.split("_t")[1]), R[D][m][c][k] - R[D][m]["KNN"][k]) for D in SETS for k in sorted(set(R[D][m][c]) & set(R[D][m]["KNN"])) if int(k.split("_t")[1]) in ok[D]]
                blk[f"{c}-KNN"] = ci(rows, np.random.default_rng(0))
            for c in ("PSHIFT", "PSHIFTW"):
                rows = [(D, int(k.split("_t")[1]), R[D][m][c][k] - R[D][m]["RRQSDC19L"][k]) for D in SETS for k in sorted(set(R[D][m][c]) & set(R[D][m]["RRQSDC19L"])) if int(k.split("_t")[1]) in ok[D]]
                blk[f"{c}-RRQSDC19L"] = ci(rows, np.random.default_rng(0))
            rep[f"{ruler}_{m}"] = blk
    g = lambda c, m: rep[f"REG_{m}"][f"{c}-KNN"]["rank_diff"]
    robust = all(g(c, m) >= 2 and rep[f"REG_{m}"][f"{c}-RRQSDC19L"]["rank_diff"] >= 2 for c in ("PSHIFT", "PSHIFTW") for m in ("LDA", "ECAPA"))
    game = any(g(c, m) <= 0 for c in ("PSHIFT", "PSHIFTW") for m in ("LDA", "ECAPA"))
    rep["verdict"] = "REG_GAMEABLE" if game else "REG_ROBUST" if robust else "MIXED"
    for r in ("REG", "IND"):
        for m in ("LDA", "ECAPA"):
            print(r, m, "levels", rep[f"{r}_{m}"]["levels_ok"], {c: (rep[f"{r}_{m}"][f"{c}-KNN"]["rank_diff"], rep[f"{r}_{m}"][f"{c}-KNN"]["ci95"]) for c in FORMS})
    print("verdict", rep["verdict"])
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
