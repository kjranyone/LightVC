"""D654: 発話ごとの hf / tilt の散らばりの源を分ける(開発の組 dev2 / dev3 × k5 / k7 / k12)。出力の hf と、入力(操作者の SRC)の hf・目標本人の参照の録音の hf との相関、
目標ごとの平均の分散の割合。腕 = PR1(D653)・候補 C(生成器・関門 G の hf を通る)・錨 GU。規則は results/conv_p0/d654/prereg.yaml。
    uv run python d654_spread.py
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

from d631_world_swap import C, G, H, ROOT, SETS, keys
from d651_copy_hf import best8, hf_tilt

OUT = ROOT / "results/conv_p0/d654"
ARMS = {"PR1": lambda k: f"PR1_{k}", "C": lambda k: C[k], "GU": lambda k: G[k][0]}


def job(it: tuple) -> dict:
    s, k, key = it
    from d6_unit_cov import load48
    out = {"s": s, "k": k, "key": key, "ti": int(key[7:10])}
    for a, nm in ARMS.items():
        x, _ = sf.read(H / s / "d570" / nm(k) / f"{key}_{nm(k)}.wav")
        out[a] = hf_tilt(np.asarray(x, np.float64))
    x, _ = sf.read(H / s / "d56" / f"{key}_SRC.wav")
    out["SRC"] = hf_tilt(np.asarray(x, np.float64))
    out["REF"] = hf_tilt(best8(load48(H / s / "d37" / "_work" / f"ref_{out['ti']:03d}.wav").astype(np.float64)))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    with Pool(int(os.environ.get("LVC_NPROC", "12"))) as pl:
        recs = pl.map(job, its, chunksize=4)
    rep: dict = {}
    for i, stat in enumerate(("hf", "tilt")):
        rep[stat] = {}
        for a in ARMS:
            rep[stat][a] = {}
            for k in C:
                rk = [r for r in recs if r["k"] == k]
                y = np.array([r[a][i] for r in rk])
                src = np.array([r["SRC"][i] for r in rk])
                ref = np.array([r["REF"][i] for r in rk])
                tis = np.array([f"{r['s']}{r['ti']}" for r in rk])
                gm = {t: y[tis == t].mean() for t in set(tis)}
                between = float(np.var([gm[t] for t in tis]) / (np.var(y) + 1e-12))
                X = np.stack([src, ref, np.ones_like(src)], 1)
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                rep[stat][a][k] = {"p15_p85_width": round(float(np.percentile(y, 85) - np.percentile(y, 15)), 2), "median": round(float(np.median(y)), 2),
                                   "r_src": round(float(np.corrcoef(y, src)[0, 1]), 3), "r_ref": round(float(np.corrcoef(y, ref)[0, 1]), 3),
                                   "beta_src": round(float(beta[0]), 3), "beta_ref": round(float(beta[1]), 3), "target_share": round(between, 3)}
            print(stat, a, json.dumps(rep[stat][a]), flush=True)
        y = np.array([r["REF"][i] for r in recs if r["k"] == 7])
        rep[stat]["REF_width"] = round(float(np.percentile(y, 85) - np.percentile(y, 15)), 2)
        y = np.array([r["SRC"][i] for r in recs if r["k"] == 7])
        rep[stat]["SRC_width"] = round(float(np.percentile(y, 85) - np.percentile(y, 15)), 2)
        print(stat, "REF width", rep[stat]["REF_width"], "SRC width", rep[stat]["SRC_width"], flush=True)
    (OUT / "spread.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
