"""D157(学習なし・判定): 目標本人の c0 + 動き(XCD)の明瞭さの悪化が整列の産物かを results/conv_p0/d157_prereg.yaml の規則で判定する。
入力 = d23_cer の per(組 → CER)・d118 の同一性(--tag)。男声単位(24)のブートストラップ 5000 回で CONV との差 Δ と対の差の 95% 区間。
    uv run python d157_xcd_verdict.py --cer ../results/conv_p0/d157_cer.json --name tts157 --ident ../results/conv_p0/d157_id.json --out ../results/conv_p0/d157.json
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
    a = ap.parse_args()
    per = json.loads(Path(a.cer).read_text())[a.name]["per"]
    conds = ["CONV", "TAB0", "XDYN", "XC0", "XCD", "XDB", "XSELFRT", "XSELFRTCD", "XSELFKCD"]
    keys = sorted(set.intersection(*(set(per[c]) for c in conds)))
    X = {c: np.array([per[c][k] for k in keys]) for c in conds}
    spk = np.array([k.split("_")[0] for k in keys])
    us = sorted(set(spk))
    rng = np.random.default_rng(0)
    idx = {u: np.flatnonzero(spk == u) for u in us}
    B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
    ci = lambda v: [round(float(q), 4) for q in np.percentile([v[ii].mean() for ii in B], [2.5, 97.5])]
    d = {c: X[c] - X["CONV"] for c in conds}
    rep: dict = {"n": len(keys), "clusters": len(us), "cer": {}, "delta": {}}
    for c in conds:
        v = X[c]
        rep["cer"][c] = {"mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4), "p90": round(float(np.percentile(v, 90)), 4),
                         "n_gt_0.1": int((v > 0.1).sum()), "n_gt_0.3": int((v > 0.3).sum())}
        rep["delta"][c] = {"mean": round(float(d[c].mean()), 4), "ci95": ci(d[c])}
    pr = X["XCD"] - X["XSELFRTCD"]
    rep["XCD_minus_XSELFRTCD"] = {"mean": round(float(pr.mean()), 4), "ci95": ci(pr)}
    idn = json.loads(Path(a.ident).read_text())
    rep["identity"] = {c: idn[c] for c in ["REAL_A", "CONV", "TAB0", "XDYN", "XC0", "XCD", "XDB"] if c in idn}
    hi, lo = rep["XCD_minus_XSELFRTCD"]["ci95"][1], rep["XCD_minus_XSELFRTCD"]["ci95"][0]
    if hi <= 0.01 and rep["identity"]["XCD"]["top1"] >= 0.85:
        v = "COMPATIBLE"
    elif lo > 0 and rep["delta"]["XCD"]["mean"] >= 2 * rep["delta"]["XSELFRTCD"]["mean"]:
        v = "CONFLICT"
    else:
        v = "INCONCLUSIVE"
    rep["verdict"] = v
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
