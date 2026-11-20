"""段 D5(paper_snv rev41 §6・results/snv_d5/prereg.yaml attribution・学習なし): 開発の組(cv の devA)の声ごとの CER を、台帳の量と並べる。
系ごとに、声ごとの CER(k ごとの試行の中央値)と量ごとの Spearman の相関、族(既定 f1)の [2.5 %, 97.5 %] の内にいる声 対 外にいる声の CER の中央値、順位の上の重回帰の係数を出す。
    cd training && uv run --no-sync python snv_diag/d5_attrib.py [rows=results/snv_d5/dev_rows_devA.json] [fam=f1] [out=results/snv_d5/attrib.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d5_ledger as LD

ROOT = LD.ROOT
EXTRA = ["mora_rate", "rate_lv", "vq", "vrate"]
REG = ["f0_med", "f0_span", "mora_rate", "rate", "floor", "fall", "vq", "tilt", "env_v_b6k"]


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    res = json.loads((ROOT / opt.get("rows", "results/snv_d5/dev_rows_devA.json")).read_text())
    fam = opt.get("fam", "f1")
    cv = json.loads(LD.path_of("cv").read_text())
    keys = [k for ks in LD.AXES.values() for k in ks] + EXTRA
    out: dict = {"fam": fam, "systems": {}}
    rng = {}
    for k in keys:
        x = LD.vals(fam, k)
        rng[k] = (np.percentile(x, 2.5), np.percentile(x, 97.5), np.subtract(*np.percentile(x, [75, 25]))) if len(x) >= 10 else None
    prim = [k for ks in LD.AXES.values() for k in ks if rng[k] is not None]
    for tag, rows in res["rows"].items():
        R = [r for r in rows if r["group"] != "jvnv"]
        spk = sorted({r["spk"] for r in R})
        if len(spk) < 10 or tag == "orig":
            continue
        o: dict = {}
        for kk in sorted({r["k"] for r in R}):
            c = np.array([np.median([r["cer"] for r in R if r["spk"] == s and r["k"] == kk]) for s in spk])
            S = {k: np.array([cv[s].get(k, np.nan) for s in spk], float) for k in keys}
            cor = {}
            for k in keys:
                m = np.isfinite(S[k])
                cor[k] = round(float(spearmanr(S[k][m], c[m])[0]), 3) if m.sum() >= 10 else None
            outd = {k: np.where(np.isfinite(S[k]), np.maximum(0, np.maximum(rng[k][0] - S[k], S[k] - rng[k][1])) / (rng[k][2] + 1e-9), 0.0) for k in prim}
            n_out = sum((outd[k] > 0).astype(int) for k in prim)
            inside = n_out == 0
            X = np.stack([rankdata(np.where(np.isfinite(S[k]), S[k], np.nanmedian(S[k]))) for k in REG], 1)
            X = (X - X.mean(0)) / (X.std(0) + 1e-9)
            y = rankdata(c)
            y = (y - y.mean()) / (y.std() + 1e-9)
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            r2 = 1 - float(((y - X @ beta) ** 2).sum() / (y ** 2).sum())
            top = sorted(((k, cor[k]) for k in keys if cor[k] is not None), key=lambda t: -abs(t[1]))[:8]
            o[f"k{kk}"] = {"n_spk": len(spk), "cer_median_spk": round(float(np.median(c)), 4), "inside_all": {"n": int(inside.sum()), "cer_median": round(float(np.median(c[inside])), 4) if inside.any() else None},
                           "outside_any": {"n": int((~inside).sum()), "cer_median": round(float(np.median(c[~inside])), 4) if (~inside).any() else None},
                           "out_rate_by_stat": {k: round(float((outd[k] > 0).mean()), 3) for k in prim if (outd[k] > 0).any()},
                           "spearman_top": top, "spearman_n_out": round(float(spearmanr(n_out, c)[0]), 3), "rank_reg": {k: round(float(b), 3) for k, b in zip(REG, beta)}, "rank_reg_r2": round(r2, 3),
                           "tertile_by_f0_med": [round(float(np.median(c[(S["f0_med"] >= lo) & (S["f0_med"] <= hi)])), 4) for lo, hi in zip(np.percentile(S["f0_med"], [0, 33.3, 66.7]), np.percentile(S["f0_med"], [33.3, 66.7, 100]))],
                           "tertile_by_mora_rate": [round(float(np.median(c[(S["mora_rate"] >= lo) & (S["mora_rate"] <= hi)])), 4) for lo, hi in zip(np.nanpercentile(S["mora_rate"], [0, 33.3, 66.7]), np.nanpercentile(S["mora_rate"], [33.3, 66.7, 100]))],
                           "tertile_by_floor": [round(float(np.median(c[(S["floor"] >= lo) & (S["floor"] <= hi)])), 4) for lo, hi in zip(np.percentile(S["floor"], [0, 33.3, 66.7]), np.percentile(S["floor"], [33.3, 66.7, 100]))]}
        out["systems"][tag] = o
        print("==", tag, flush=True)
        for kk, d in o.items():
            print(f"  {kk}: 声ごとの中央値 {d['cer_median_spk']}  内 {d['inside_all']}  外 {d['outside_any']}  外れの数との相関 {d['spearman_n_out']}")
            print(f"     相関の上位 {d['spearman_top']}")
            print(f"     順位の回帰(R² {d['rank_reg_r2']}){d['rank_reg']}")
            print(f"     3 分位: 高さ {d['tertile_by_f0_med']} 速さ {d['tertile_by_mora_rate']} 床 {d['tertile_by_floor']}")
    (ROOT / opt.get("out", "results/snv_d5/attrib.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
