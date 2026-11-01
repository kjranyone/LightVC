"""D215 の判定(学習なし・results/conv_p0/d215_prereg.yaml): RRPS16C・RRPT16C の 4 軸(3 組)と、90 目標の順位の差・CER・余裕。
    uv run python d215_verdict.py --s <scratchpad> --out ../results/conv_p0/d215.json
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
FORMS = ("RRPS16C", "RRPT16C", "RRSN16C", "RRTN16C")
PAIRS = [("RRPS16C", "KNN"), ("RRPT16C", "KNN"), ("RRSN16C", "KNN"), ("RRTN16C", "KNN"), ("G03S10ZH", "KNN"), ("RRPS16C", "G03S10ZH"), ("RRPT16C", "G03S10ZH"),
         ("RRPS16C", "RRSN16C"), ("RRPT16C", "RRTN16C"), ("RRPT16C", "RRPS16C")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.s)
    runs = {D: V.ranks(S / f"d215{D}_lda.npz", ROOT / f"results/conv_p0/d215{D}_tri.json") for D in SETS}
    rep: dict = {}
    for m in ("LDA", "ECAPA"):
        blk: dict = {f"levels_{D}": {c: round(float(np.mean(list(r.values()))), 2) for c, r in sorted(runs[D][m].items())} for D in SETS}
        for p, o in PAIRS:
            rows = {D: [(D, int(k.split("_t")[1]), R[m][p][k] - R[m][o][k]) for k in sorted(set(R[m][p]) & set(R[m][o]))] for D, R in runs.items()}
            rng = np.random.default_rng(0)
            blk[f"{p}-{o}"] = {"pooled90": ci(sum(rows.values(), []), rng), "dev_dev2": ci(rows["dev"] + rows["dev2"], rng), "dev3": ci(rows["dev3"], rng)}
        rep[m] = blk
    cer = {D: json.loads((ROOT / f"results/conv_p0/d215{D}_cer.json").read_text())[D] for D in SETS}
    rep["cer"] = {D: {c: round(float(v["cer_mean"]), 4) for c, v in cer[D].items() if c not in ("per", "examples")} for D in SETS}
    rep["cer_mean_3"] = {c: round(float(np.mean([rep["cer"][D][c] for D in SETS])), 4) for c in rep["cer"]["dev"] if all(c in rep["cer"][D] for D in SETS)}
    ax = {D: json.loads((ROOT / f"results/conv_p0/d109v2_d215{D}.json").read_text()) for D in SETS}
    keys = {c: f"d139:{c}" for c in FORMS} | {"G03S10ZH": "d114:G03S10ZH"}
    rep["axes"] = {c: {D: {"pass": ax[D][kk]["pass"], "fail": [k for k, v in ax[D][kk]["stats"].items() if not v["pass"]],
                           "shimmer": [ax[D][kk]["stats"]["shimmer"]["median"], ax[D][kk]["stats"]["shimmer"]["real_p15_p85"][1]],
                           "lmstd": [ax[D][kk]["stats"]["lmstd"]["median"], ax[D][kk]["stats"]["lmstd"]["real_p15_p85"]]} for D in SETS} for c, kk in keys.items()}
    nat = [c for c in ("RRPS16C", "RRPT16C") if all(rep["axes"][c][D]["pass"] for D in SETS)]
    rep["natural_pass"] = nat
    cm = rep["cer_mean_3"]
    ch = lambda s_, t_: t_ if cm[t_] <= cm[s_] - 0.015 and all(rep[m][f"{t_}-{s_}"]["pooled90"]["rank_diff"] <= 0.25 for m in ("LDA", "ECAPA")) else s_
    rep["choice"] = ch("RRPS16C", "RRPT16C") if len(nat) == 2 else nat[0] if nat else None
    rep["verdict"] = "NAT_OK" if nat else "NAT_NOT"
    rep["owner_line_descriptive_pooled90"] = {p: "MET" if all(rep[m][f"{p}-KNN"]["pooled90"]["rank_diff"] <= 0 and rep[m][f"{p}-KNN"]["pooled90"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA")) else "NOT_MET"
                                              for p, o in PAIRS if o == "KNN"}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
