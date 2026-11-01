"""D229 の判定(results/conv_p0/d229_prereg.yaml): 声域をそろえた中心(REG)と独立の中心(IND = rr19L の評価)で、各形 − kNN-VC の順位差を並べる。
    uv run python d229_verdict.py --s <scratchpad> --root <dev の親> --out ../results/conv_p0/d229.json
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
FORMS = ("RRQSDC19L", "RRQSD19L", "RRQS19L", "RRQSDC18P", "RRQS18P", "G03S10ZH")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S, RT = Path(a.s), Path(a.root)
    ok = {D: {m["ti"] for m in json.loads((RT / D / "cen_reg/meta.json").read_text()) if m["ok"]} for D in SETS}
    runs = {"IND": {D: V.ranks(S / f"rr19{D}_lda.npz", ROOT / f"results/conv_p0/rr19{D}_tri.json") for D in SETS},
            "REG": {D: V.ranks(S / f"d229{D}_lda.npz", ROOT / f"results/conv_p0/d229{D}_tri.json") for D in SETS}}
    rep: dict = {"targets_ok": {D: len(ok[D]) for D in SETS}, "targets_ok_total": sum(len(v) for v in ok.values())}
    for ruler, R in runs.items():
        for m in ("LDA", "ECAPA"):
            blk = {}
            for c in FORMS:
                for sub in ("all", "ok"):
                    rows = [(D, int(k.split("_t")[1]), R[D][m][c][k] - R[D][m]["KNN"][k]) for D in SETS for k in sorted(set(R[D][m][c]) & set(R[D][m]["KNN"]))
                            if sub == "all" or int(k.split("_t")[1]) in ok[D]]
                    blk[f"{c}-KNN|{sub}"] = ci(rows, np.random.default_rng(0))
            blk["levels_ok"] = {c: round(float(np.mean([R[D][m][c][k] for D in SETS for k in R[D][m][c] if int(k.split("_t")[1]) in ok[D]])), 2) for c in FORMS + ("KNN",)}
            rep[f"{ruler}_{m}"] = blk
    met = lambda ruler, c: all(rep[f"{ruler}_{m}"][f"{c}-KNN|ok"]["rank_diff"] <= 0 and rep[f"{ruler}_{m}"][f"{c}-KNN|ok"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA"))
    rep["owner_line_ok_subset"] = {r: {c: met(r, c) for c in FORMS} for r in ("IND", "REG")}
    drop = all(rep[f"REG_{m}"]["RRQSDC19L-KNN|ok"]["rank_diff"] <= rep[f"IND_{m}"]["RRQSDC19L-KNN|all"]["rank_diff"] - 0.3 for m in ("LDA", "ECAPA"))
    rep["verdict"] = "REG_EXPLAINS" if (met("REG", "RRQSDC19L") or met("REG", "RRQSD19L")) else "REG_PARTIAL" if drop else "REG_NONE"
    print(json.dumps({k: rep[k] for k in ("targets_ok", "owner_line_ok_subset", "verdict")}, indent=1, ensure_ascii=False))
    for r in ("IND", "REG"):
        for m in ("LDA", "ECAPA"):
            print(r, m, {c: (rep[f"{r}_{m}"][f"{c}-KNN|ok"]["rank_diff"], rep[f"{r}_{m}"][f"{c}-KNN|ok"]["ci95"]) for c in ("RRQSDC19L", "RRQSD19L", "G03S10ZH")})
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
