"""D183(学習なし・判定): 独立の中心の同一性の対比を dev(D181 の IND)と dev2 で並べ、まとめて(60 目標)区間を出す。results/conv_p0/d183_prereg.yaml の fork。
LDA = d4b の --dump・ECAPA = d146 の items / ranks。クラスタ = (組, 目標)。
    uv run python d183_verdict.py --s <scratchpad> --out ../results/conv_p0/d183.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import d181_verdict as V

ROOT = Path(__file__).resolve().parent.parent
CON = {"C1_memory": ("RRSWP", "RRM20"), "C2_R_vs_table": ("RR1C", "TABR"), "C3_dyn_vs_table": ("G05S10H", "TABR"), "C4_data": ("RR8C", "RR1C"),
       "s_mem5": ("RRM05", "RRM20"), "s_dyn_amount": ("G05S10H", "G00S10H"), "s_rc0": ("RRC1C", "TABR"), "s_rc8": ("RRC8C", "RRC1C")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.s)
    sets = {"dev": V.ranks(S / "d181ind_lda.npz", ROOT / "results/conv_p0/d181ind_tri.json"), "dev2": V.ranks(S / "d183dev2_lda.npz", ROOT / "results/conv_p0/d183dev2_tri.json")}
    if (S / "d181ind_r8_lda.npz").exists():
        x = V.ranks(S / "d181ind_r8_lda.npz", ROOT / "results/conv_p0/d181ind_r8_tri.json")
        for m in x:
            sets["dev"][m].update(x[m])
    rng = np.random.default_rng(0)
    rep: dict = {}
    for m in ("LDA", "ECAPA"):
        rep[m] = {}
        for nm, (p, o) in CON.items():
            rows = []
            for sn, R in sets.items():
                if p in R[m] and o in R[m]:
                    for k in sorted(set(R[m][p]) & set(R[m][o])):
                        rows.append((sn, int(k.split("_t")[1]), R[m][p][k] - R[m][o][k]))
            if not rows:
                continue
            blk = {}
            for scope in ("dev", "dev2", "pooled"):
                rr = [r for r in rows if scope == "pooled" or r[0] == scope]
                if not rr:
                    continue
                cl = np.array([f"{r[0]}:{r[1]}" for r in rr]); d = np.array([r[2] for r in rr], float)
                us = sorted(set(cl)); idx = {u: np.flatnonzero(cl == u) for u in us}
                B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
                q = np.percentile([d[ii].mean() for ii in B], [2.5, 97.5])
                blk[scope] = {"diff": round(float(d.mean()), 3), "ci95": [round(float(q[0]), 3), round(float(q[1]), 3)], "n": len(d), "targets": len(us)}
            if "dev" in blk and "dev2" in blk:
                blk["same_sign"] = bool(np.sign(blk["dev"]["diff"]) == np.sign(blk["dev2"]["diff"]))
            pc = blk["pooled"]["ci95"]
            blk["pooled_reading"] = "worse" if pc[0] > 0 else "better" if pc[1] < 0 else "undecided"
            rep[m][f"{nm}({p}-{o})"] = blk
        for sn, R in sets.items():
            if "REFMID" in R[m]:
                rr = np.array(list(R[m]["REFMID"].values()), float)
                rep[m][f"REFMID_{sn}"] = {"top1": round(float((rr == 1).mean()), 3), "mean_rank": round(float(rr.mean()), 2)}
            rep[m][f"levels_{sn}"] = {c: round(float(np.mean(list(v.values()))), 2) for c, v in sorted(R[m].items())}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
