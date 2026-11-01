"""D181(学習なし・物差しの検査・判定): 開発用の組の同一性を、参照の録音の真ん中 8 秒の中心(REF = 従来)と独立の中心(IND = 参照と評価のクリップを除いた残りの真ん中 8 秒)で並べ、
results/conv_p0/d181_prereg.yaml の fork で判定する。LDA = d4b の --dump・ECAPA = d146 の items / ranks。目標単位(30)のブートストラップ 5000 回。
    uv run python d181_verdict.py --s <scratchpad> --out ../results/conv_p0/d181.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def ranks(npz: Path, tri: Path) -> dict:
    z = np.load(npz)
    lda: dict = {}
    for nm, ti, c, s in zip(z["name"], z["ti"], z["cond"], z["score"]):
        lda.setdefault(str(c), {})[str(nm)[: -len(str(c)) - 5]] = int((s > s[int(ti)]).sum()) + 1
    eca = {c: dict(zip(r["items"], r["ranks"])) for c, r in json.loads(tri.read_text())["ECAPA"].items() if isinstance(r, dict) and "items" in r}
    return {"LDA": lda, "ECAPA": eca}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.s)
    R = {}
    for v in ("ref", "ind"):
        R[v] = ranks(S / f"d181{v}_lda.npz", ROOT / f"results/conv_p0/d181{v}_tri.json")
        if (S / f"d181{v}_r8_lda.npz").exists():
            x = ranks(S / f"d181{v}_r8_lda.npz", ROOT / f"results/conv_p0/d181{v}_r8_tri.json")
            for m in x:
                R[v][m].update(x[m])
    conds = sorted(c for c in R["ref"]["LDA"] if c in R["ind"]["LDA"])
    keys = sorted(set.intersection(*(set(R[v][m][c]) for v in R for m in ("LDA", "ECAPA") for c in conds)))
    tg = np.array([int(k.split("_t")[1]) for k in keys])
    us = sorted(set(tg))
    idx = {u: np.flatnonzero(tg == u) for u in us}
    rng = np.random.default_rng(0)
    B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
    ci = lambda v: [round(float(q), 3) for q in np.percentile([v[ii].mean() for ii in B], [2.5, 97.5])]
    rep: dict = {"n": len(keys), "targets": len(us), "conds": conds}
    contrasts = [("G05S10H", "TABR"), ("G00S10H", "TABR"), ("G05S10H", "G00S10H"), ("RR1C", "TABR"), ("KNN", "TABR"), ("RRSWP", "RRM20"), ("RRM10", "RRM20"),
                 ("XRD05", "TABR"), ("RR8C", "RR1C"), ("RR8C", "TABR"), ("XRD05E", "TABR"), ("RR7C", "TABR"), ("RR4", "TABR"), ("RRC1C", "TABR"), ("DB1H", "G00S10H"), ("DB3H", "G00S10H")]
    for v in R:
        for m in ("LDA", "ECAPA"):
            X = {c: np.array([R[v][m][c][k] for k in keys], float) for c in conds}
            blk = {c: {"top1": round(float((x == 1).mean()), 3), "mean_rank": round(float(x.mean()), 2)} for c, x in X.items()}
            for p, o in contrasts:
                if p not in X or o not in X:
                    continue
                d = X[p] - X[o]
                blk[f"{p}-{o}"] = {"rank_diff": round(float(d.mean()), 3), "ci95": ci(d)}
            ref_ = R[v][m].get("REFMID")
            if ref_:
                rr = np.array(list(ref_.values()), float)
                blk["REFMID"] = {"top1": round(float((rr == 1).mean()), 3), "mean_rank": round(float(rr.mean()), 2), "n": len(rr)}
            rep[f"{v}_{m}"] = blk
    ratio = {}
    for m in ("LDA", "ECAPA"):
        dr, di = rep[f"ref_{m}"]["G05S10H-TABR"]["rank_diff"], rep[f"ind_{m}"]["G05S10H-TABR"]["rank_diff"]
        ratio[m] = round(di / dr, 3) if abs(dr) > 1e-9 else None
    rep["delta_ratio_ind_over_ref"] = ratio
    ok = [r is not None for r in ratio.values()]
    if all(ok) and all(r <= 0.5 for r in ratio.values()):
        v = "SHARED"
    elif all(ok) and all(r > 0.5 for r in ratio.values()):
        v = "SPEAKER"
    else:
        v = "MIXED"
    rep["verdict"] = v
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
