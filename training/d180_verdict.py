"""D180(学習なし・判定): 素朴な鍵の鋭い読み(XRD05 / XRD02 と内訳 E・C)を表だけ(TABR)と実女声の開発用の組で並べ、results/conv_p0/d180_prereg.yaml の fork で判定する。
LDA = d4b の --dump(本人の順位)・ECAPA = d146 の items / ranks・CER = d23 の per。目標単位(30)のブートストラップ 5000 回で平均順位の利得(差し引く側の順位 − 比べる側の順位)の 95% 区間。
    uv run python d180_verdict.py --lda <S>/d180_lda.npz --tri ../results/conv_p0/d180_tri.json --cer ../results/conv_p0/d180_cer.json --out ../results/conv_p0/d180.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lda", required=True)
    ap.add_argument("--tri", required=True)
    ap.add_argument("--cer", default="")
    ap.add_argument("--cer_name", default="dev180")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conds = ["TABR", "XRD05", "XRD05E", "XRD05C", "XRD02", "XRD02E", "XRD02C"]
    z = np.load(a.lda)
    lda: dict = {}
    for nm, ti, c, s in zip(z["name"], z["ti"], z["cond"], z["score"]):
        lda.setdefault(str(c), {})[str(nm)[: -len(str(c)) - 5]] = int((s > s[int(ti)]).sum()) + 1
    eca = {c: dict(zip(r["items"], r["ranks"])) for c, r in json.loads(Path(a.tri).read_text())["ECAPA"].items() if isinstance(r, dict) and "items" in r}
    keys = sorted(set.intersection(*(set(lda[c]) & set(eca[c]) for c in conds)))
    tg = np.array([int(k.split("_t")[1]) for k in keys])
    us = sorted(set(tg))
    idx = {u: np.flatnonzero(tg == u) for u in us}
    rng = np.random.default_rng(0)
    B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
    ci = lambda v: [round(float(q), 4) for q in np.percentile([v[ii].mean() for ii in B], [2.5, 97.5])]
    rk = {r: {c: np.array([m[c][k] for k in keys], float) for c in conds} for r, m in (("LDA", lda), ("ECAPA", eca))}
    rep: dict = {"n": len(keys), "targets": len(us)}
    pairs = [(c, "TABR") for c in conds[1:]]
    for r in rk:
        rep[r] = {c: {"top1": round(float((v == 1).mean()), 3), "mean_rank": round(float(v.mean()), 2)} for c, v in rk[r].items()}
        for p, o in pairs:
            dr = rk[r][o] - rk[r][p]
            dh = (rk[r][p] == 1).astype(float) - (rk[r][o] == 1)
            rep[r][f"{p}-{o}"] = {"rank_gain": round(float(dr.mean()), 3), "rank_gain_ci95": ci(dr), "top1": round(float(dh.mean()), 4), "top1_ci95": ci(dh)}
    if a.cer:
        per = json.loads(Path(a.cer).read_text())[a.cer_name]["per"]
        rep["cer"] = {c: round(float(np.mean([per[c][k] for k in keys if k in per[c]])), 4) for c in conds if c in per}
    m = rep["LDA"]["XRD05-TABR"]
    v = "SHARP" if m["rank_gain_ci95"][0] > 0 and rep["ECAPA"]["XRD05-TABR"]["rank_gain"] > 0 else "TABLE"
    rep["verdict"] = v
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
