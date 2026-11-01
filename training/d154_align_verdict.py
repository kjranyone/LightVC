"""D154 / D155(学習なし・判定): D152 の別話者の動きの明瞭さの悪化(XDB)が DTW の写しで説明できるかを、results/conv_p0/d154_prereg.yaml の規則で判定する(--rules d155 = d155_prereg.yaml の規則・基準の Δ_XDB は D154 の値)。
入力 = d23_cer の per(組 → CER)・d106 --only ...XSELF* の _self_log_*.json。男声単位(24)のブートストラップ 5000 回で CONV との差 Δ と対の差の 95% 区間。
    uv run python d154_align_verdict.py --cer ../results/conv_p0/d154_cer.json --name tts154 --logs <s>/d154 --out ../results/conv_p0/d154.json
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
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rules", default="d154", choices=["d154", "d155"])
    ap.add_argument("--ref_dxdb", type=float, default=0.0225)
    a = ap.parse_args()
    per = json.loads(Path(a.cer).read_text())[a.name]["per"]
    conds = ["CONV", "XDB", "XDYN", "XSELFK", "XSELFRT"]
    keys = sorted(set.intersection(*(set(per[c]) for c in conds)))
    X = {c: np.array([per[c][k] for k in keys]) for c in conds}
    spk = np.array([k.split("_")[0] for k in keys])
    us = sorted(set(spk))
    rng = np.random.default_rng(0)
    idx = {u: np.flatnonzero(spk == u) for u in us}
    B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
    ci = lambda v: [round(float(q), 4) for q in np.percentile([v[ii].mean() for ii in B], [2.5, 97.5])]
    d = {c: X[c] - X["CONV"] for c in conds}
    rep: dict = {"n": len(keys), "clusters": len(us), "cer": {}, "delta": {}, "paired_vs_XDB": {}}
    for c in conds:
        v = X[c]
        rep["cer"][c] = {"mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4), "p90": round(float(np.percentile(v, 90)), 4),
                         "n_gt_0.1": int((v > 0.1).sum()), "n_gt_0.3": int((v > 0.3).sum())}
        rep["delta"][c] = {"mean": round(float(d[c].mean()), 4), "ci95": ci(d[c])}
        if c not in ("CONV", "XDB"):
            rep["paired_vs_XDB"][c] = {"mean": round(float((d["XDB"] - d[c]).mean()), 4), "ci95": ci(d["XDB"] - d[c])}
    logs = list({r["pair"]: r for f in sorted(Path(a.logs).glob("_self_log_*.json")) for r in json.loads(f.read_text())}.values())
    agg = lambda path: round(float(np.mean([r[path[0]][path[1]] if len(path) == 2 else r[path[0]] for r in logs])), 3)
    rep["paths"] = {"n_pairs": len(logs), "K_ne": agg(("K", "ne")), "K_mae": agg(("K", "mae")), "K_mae_v": agg(("K", "mae_v")),
                    "RT_ne": agg(("RT", "ne")), "RT_mae": agg(("RT", "mae")), "RT_mae_v": agg(("RT", "mae_v")), "RT_p90_v": agg(("RT", "p90_v")), "RT_bias": agg(("RT", "bias")),
                    "RT_rep": agg(("RT", "rep")), "RT_skip": agg(("RT", "skip")), "B_rep": agg(("B", "rep")), "B_skip": agg(("B", "skip")),
                    "drel_K": agg(("drel_K",)), "drel_RT": agg(("drel_RT",)), "drel_B": agg(("drel_B",)), "drel_A": agg(("drel_A",))}
    dx, lo_x = rep["delta"]["XDB"]["mean"], rep["delta"]["XDB"]["ci95"][0]
    k_ok = rep["paths"]["K_ne"] >= 0.10
    if lo_x <= 0:
        v = "INCONCLUSIVE(XDB の悪化が再現しない)"
    elif rep["delta"]["XSELFRT"]["ci95"][1] < 0.5 * dx and rep["paired_vs_XDB"]["XSELFRT"]["ci95"][0] > 0:
        v = "REALIZATION"
    elif k_ok and rep["delta"]["XSELFK"]["ci95"][0] > 0 and rep["delta"]["XSELFK"]["mean"] >= 0.67 * dx and rep["paired_vs_XDB"]["XSELFK"]["ci95"][0] <= 0 <= rep["paired_vs_XDB"]["XSELFK"]["ci95"][1]:
        v = "ALIGNMENT"
    else:
        v = "INCONCLUSIVE"
    if a.rules == "d155":
        r0 = a.ref_dxdb
        dn, cin = rep["delta"]["XDB"]["mean"], rep["delta"]["XDB"]["ci95"]
        if rep["delta"]["XSELFRT"]["ci95"][1] >= 0.5 * r0:
            v = "INCONCLUSIVE(整列器自身の誤差がまだ有害)"
        elif cin[0] > 0 and dn >= 0.67 * r0:
            v = "REALIZATION"
        elif cin[1] < 0.5 * r0:
            v = "ALIGNMENT"
        else:
            v = "PARTIAL"
        rep["ref_dxdb"] = r0
    rep["K_branch_usable"] = k_ok
    rep["verdict"] = v
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
