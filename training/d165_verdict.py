"""D165(判定): 実女声のパラレルの組で results/conv_p0/d165_prereg.yaml の規則を当てる。
同一性 = d4b_lda_verifier --dump の npz(出力ごとの全目標への得点 → 本人の順位)・CER = d23 の per。目標単位のブートストラップ 5000 回。
    uv run python d165_verdict.py --dump <s>/d165_lda.npz --cer ../results/conv_p0/d165_cer.json --name real165 --out ../results/conv_p0/d165.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--cer", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.dump)
    pi, ti, cond, score = z["pi"], z["ti"], z["cond"], z["score"]
    rank = np.array([1 + int((score[i] > score[i, ti[i]]).sum()) for i in range(len(pi))])
    per = json.loads(Path(a.cer).read_text())[a.name]["per"]
    conds = sorted(set(cond.tolist()))
    keys = sorted({(int(p), int(t)) for p, t in zip(pi, ti)})
    R = {c: {(int(p), int(t)): r for p, t, cc, r in zip(pi, ti, cond, rank) if cc == c} for c in conds}
    keys = [k for k in keys if all(k in R[c] for c in conds)]
    tg = np.array([t for _, t in keys])
    us = sorted(set(tg.tolist()))
    rng = np.random.default_rng(0)
    idx = {u: np.flatnonzero(tg == u) for u in us}
    B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
    ci = lambda v: [round(float(q), 4) for q in np.percentile([v[ii].mean() for ii in B], [2.5, 97.5])]
    hit = {c: np.array([R[c][k] == 1 for k in keys], float) for c in conds}
    mr = {c: np.array([R[c][k] for k in keys], float) for c in conds}
    ck = lambda k: f"p{k[0]:04d}_t{k[1]:03d}"
    cer = {c: np.array([per[c].get(ck(k), np.nan) for k in keys]) for c in conds if c in per}
    rep: dict = {"n": len(keys), "targets": len(us), "chance": round(1 / score.shape[1], 3), "identity": {}, "cer": {}}
    for c in conds:
        rep["identity"][c] = {"top1": round(float(hit[c].mean()), 3), "mean_rank": round(float(mr[c].mean()), 2), "top1_minus_CONV_ci95": ci(hit[c] - hit["CONV"])}
        if c in cer and not np.isnan(cer[c]).any():
            rep["cer"][c] = {"mean": round(float(cer[c].mean()), 4), "median": round(float(np.median(cer[c])), 4), "minus_CONV_ci95": ci(cer[c] - cer["CONV"])}
    x, cv, asf = (rep["identity"][c]["top1"] for c in ("XCD", "CONV", "ASELF"))
    lo_id = rep["identity"]["XCD"]["top1_minus_CONV_ci95"][0]
    hi_cer = rep["cer"]["XCD"]["minus_CONV_ci95"][1]
    if x >= cv + 0.5 * (asf - cv) and hi_cer <= 0.03:
        v = "GENERALIZES"
    elif lo_id > 0:
        v = "PARTIAL"
    elif lo_id <= 0 <= rep["identity"]["XCD"]["top1_minus_CONV_ci95"][1]:
        v = "NO"
    else:
        v = "NO(悪化)"
    rep["verdict"] = v
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
