"""D159(学習なし・判定): diag_rr1(交差ペアの整列 rr)と diag_rr1c(整列 c1)の R を DTW なしの推論(d132)で並べ、results/diag_rr1c/prereg.yaml の fork で判定する。
CER = d23 の per・同一性 = d118 の items / ranks(top-1 の当たり)。男声単位(24)のブートストラップ 5000 回で CONV との差と対の差の 95% 区間。
    uv run python d159_rr_align_verdict.py --cer ../results/conv_p0/d159_cer.json --name tts159 --ident ../results/conv_p0/d159_id.json --out ../results/diag_rr1c/verdict_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cer", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ident", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="RR_rr")
    ap.add_argument("--new", default="RR_c1")
    a = ap.parse_args()
    per = json.loads(Path(a.cer).read_text())[a.name]["per"]
    idn = json.loads(Path(a.ident).read_text())
    conds = [c for c in per if c not in ("SRC",)]
    keys = sorted(set.intersection(*(set(per[c]) for c in ("CONV", a.base, a.new))))
    spk = np.array([k.split("_")[0] for k in keys])
    us = sorted(set(spk))
    rng = np.random.default_rng(0)
    idx = {u: np.flatnonzero(spk == u) for u in us}
    B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
    ci = lambda v: [round(float(q), 4) for q in np.percentile([v[ii].mean() for ii in B], [2.5, 97.5])]
    X = {c: np.array([per[c].get(k, np.nan) for k in keys]) for c in conds}
    rep: dict = {"n": len(keys), "clusters": len(us), "cer": {}, "identity": {}}
    for c in conds:
        v = X[c]
        if np.isnan(v).any():
            continue
        rep["cer"][c] = {"mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4), "n_gt_0.1": int((v > 0.1).sum()),
                         "d_conv": round(float((v - X["CONV"]).mean()), 4), "d_conv_ci95": ci(v - X["CONV"])}
    hit = {}
    for c, r in idn.items():
        if isinstance(r, dict) and "items" in r:
            m = dict(zip(r["items"], r["ranks"]))
            if all(k in m for k in keys):
                hit[c] = np.array([m[k] == 1 for k in keys], float)
                rep["identity"][c] = {"top1": round(float(hit[c].mean()), 3), "mean_rank": round(float(np.mean([m[k] for k in keys])), 2)}
    dc = X[a.base] - X[a.new]
    rep["cer_base_minus_new"] = {"mean": round(float(dc.mean()), 4), "ci95": ci(dc)}
    dh = hit[a.new] - hit[a.base]
    rep["top1_new_minus_base"] = {"mean": round(float(dh.mean()), 4), "ci95": ci(dh)}
    d_b, d_n = rep["cer"][a.base]["d_conv"], rep["cer"][a.new]["d_conv"]
    cer_ok = d_n <= 0.5 * d_b and rep["cer_base_minus_new"]["ci95"][0] > 0
    id_ok = rep["top1_new_minus_base"]["ci95"][0] > 0
    rep["verdict"] = "PAIRS" if cer_ok and id_ok else "MODEL" if not cer_ok and not id_ok and rep["cer_base_minus_new"]["ci95"][0] <= 0 else "MIXED"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
