"""D178(学習なし・判定): diag_rr8c(実女声を目標側に置いた交差ペアを半分混ぜた R)を実女声の開発用の組(60 組・30 目標)で diag_rr1c と同じ実行の物差しで並べ、
results/diag_rr8c/prereg.yaml の fork で判定する。CER = d23 の per・LDA = d4b の --dump(出力ごとの 30 目標への得点 → 本人の順位)・ECAPA = d146 の items / ranks。
目標単位(30)のブートストラップ 5000 回で対の差の 95% 区間。表だけ(TABR)との差も併記する(fork の判定には使わない)。
    uv run python d178_verdict.py --cer ../results/conv_p0/d178dev_cer.json --name dev178 --lda <S>/d178dev_lda.npz --tri ../results/conv_p0/d178dev_tri.json --out ../results/diag_rr8c/verdict_eval_dev.json
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
    ap.add_argument("--lda", required=True)
    ap.add_argument("--tri", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="RR1C")
    ap.add_argument("--new", default="RR8C")
    ap.add_argument("--table", default="TABR")
    ap.add_argument("--cer_max", type=float, default=0.10)
    a = ap.parse_args()
    per = json.loads(Path(a.cer).read_text())[a.name]["per"]
    z = np.load(a.lda)
    lda: dict = {}
    for nm, ti, c, s in zip(z["name"], z["ti"], z["cond"], z["score"]):
        lda.setdefault(str(c), {})[str(nm)[: -len(str(c)) - 5]] = int((s > s[int(ti)]).sum()) + 1
    eca = {c: dict(zip(r["items"], r["ranks"])) for c, r in json.loads(Path(a.tri).read_text())["ECAPA"].items() if isinstance(r, dict) and "items" in r}
    conds = [a.base, a.new, a.table]
    keys = sorted(set.intersection(*(set(per[c]) & set(lda[c]) & set(eca[c]) for c in conds)))
    tg = np.array([int(k.split("_t")[1]) for k in keys])
    us = sorted(set(tg))
    idx = {u: np.flatnonzero(tg == u) for u in us}
    rng = np.random.default_rng(0)
    B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
    ci = lambda v: [round(float(q), 4) for q in np.percentile([v[ii].mean() for ii in B], [2.5, 97.5])]
    cer = {c: np.array([per[c][k] for k in keys]) for c in conds}
    rk = {r: {c: np.array([m[c][k] for k in keys], float) for c in conds} for r, m in (("LDA", lda), ("ECAPA", eca))}
    rep: dict = {"n": len(keys), "targets": len(us), "cer": {c: {"mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4), "n_gt_0.1": int((v > 0.1).sum())} for c, v in cer.items()}}
    for r in rk:
        rep[r] = {c: {"top1": round(float((v == 1).mean()), 3), "mean_rank": round(float(v.mean()), 2)} for c, v in rk[r].items()}
        for o in (a.base, a.table):
            dh = (rk[r][a.new] == 1).astype(float) - (rk[r][o] == 1)
            dr = rk[r][o] - rk[r][a.new]
            rep[r][f"{a.new}-{o}"] = {"top1": round(float(dh.mean()), 4), "top1_ci95": ci(dh), "rank_gain": round(float(dr.mean()), 3), "rank_gain_ci95": ci(dr)}
    dc = cer[a.new] - cer[a.base]
    rep["cer_new_minus_base"] = {"mean": round(float(dc.mean()), 4), "ci95": ci(dc)}
    cer_ok = rep["cer"][a.new]["mean"] <= a.cer_max
    id_ok = rep["LDA"][f"{a.new}-{a.base}"]["top1_ci95"][0] > 0 and rep["ECAPA"][f"{a.new}-{a.base}"]["top1"] > 0
    id_any = rep["LDA"][f"{a.new}-{a.base}"]["top1"] > 0 or rep["ECAPA"][f"{a.new}-{a.base}"]["top1"] > 0
    rep["verdict"] = "DATA" if cer_ok and id_ok else "CLARITY_ONLY" if cer_ok else "NONE" if not id_any else "UNREGISTERED(CER > 上限・同一性の点推定は上)"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
