"""D692 の事後の解析(事前登録の外・判定に使わない): 組ごとの官能指数 v1 の差(B1R / B1X − Y58)・条件の中の dose(Spearman)・目標ごとの Δ_reg の大きさとの関係。
    uv run python d692_posthoc.py --out ../results/conv_p0/d692/posthoc.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from d692_eval import ARMS, COND, H, f0_med, pairs
from kansei_v1 import KanseiV1

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    km = KanseiV1("gen")
    bj = json.loads((ROOT / "results/conv_p0/d692/build.json").read_text())["targets"]
    rep: dict = {"note": "事後の解析・判定に使わない", "cond": {}, "per": {}}
    allx, ally, alld = [], [], []
    for nm, s, kd, d in COND:
        rs = pairs(s, kd)
        keys = [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in rs]
        f = lambda sx: [H / s / "gate" / f"{d}{sx}" / f"{k}_{d}{sx}.wav" for k in keys]
        vb = np.array(km.margin(f("Y58")))
        va = {arm: np.array(km.margin(f(arm))) for arm in ARMS}
        f0t = np.array([float(np.exp(json.loads((H / s / "rr_en" / f"en_{r['ti']:03d}" / "enroll.json").read_text())["mu"])) for r in rs])
        f0o = np.array([f0_med(p) for p in f("Y58")])
        gap = np.log2(f0t / f0o)
        dlow = np.array([bj[f"{s}/{r['ti']:03d}"]["delta_db_lt300"] for r in rs])
        dd = va["B1R"] - vb
        dx = va["B1X"] - vb
        ok = np.isfinite(gap)
        c = {"n": int(ok.sum()), "gap_median_oct": round(float(np.median(gap[ok])), 3),
             "spearman_gap_B1R": round(float(spearmanr(gap[ok], dd[ok])[0]), 3), "p_gap_B1R": round(float(spearmanr(gap[ok], dd[ok])[1]), 4),
             "spearman_dlow_B1R": round(float(spearmanr(dlow, dd)[0]), 3), "p_dlow_B1R": round(float(spearmanr(dlow, dd)[1]), 4),
             "spearman_B1R_vs_negB1X": round(float(spearmanr(dd, -dx)[0]), 3),
             "sym_median": round(float(np.median(dd + dx)), 3)}
        rep["cond"][nm] = c
        for i, k in enumerate(keys):
            rep["per"][f"{nm}/{k}"] = {"Y58": round(float(vb[i]), 3), "B1R": round(float(va["B1R"][i]), 3), "B1X": round(float(va["B1X"][i]), 3),
                                       "f0_T": round(float(f0t[i]), 1), "f0_out": round(float(f0o[i]), 1), "dlow_db": dlow[i]}
        allx += list(gap[ok])
        ally += list(dd[ok])
        alld += list(dlow[ok])
        print(nm, c, flush=True)
    rep["pooled"] = {"spearman_gap_B1R": round(float(spearmanr(allx, ally)[0]), 3), "spearman_dlow_B1R": round(float(spearmanr(alld, ally)[0]), 3), "n": len(allx)}
    print(rep["pooled"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
