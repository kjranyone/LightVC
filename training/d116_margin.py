"""D116(学習なし): D109 v2 の判定の余裕(memory: 合否の余裕 < 計測ぶれ幅なら PASS にしない)。D109 のキャッシュ(<s>/d109_cache.json・発話ごとの統計)から、
本物の分布(閾値)と候補の組(発話)を目標ごとのまとまりで同時にブートストラップ(2000 回)し、各統計の合格の割合と、out_frac・中央値の 5〜95% を出す。
判定 = 全統計で合格の割合 ≥ 0.95 なら「余裕あり」・0.5〜0.95 は INCONCLUSIVE・< 0.5 は失格。出力 = results/conv_p0/d116_<組>.json。
    uv run python d116_margin.py --s <scratchpad> --set d115:S05
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIDE = {"jump": "hi", "range": "both", "micro": "both", "vfrac": "both", "contrast": "both", "amline": "hi", "shimmer": "both", "hf": "both", "tilt": "both", "lmstd": "both",
        "ltas_d": "hi", "dtilt": "both", "dhf": "both"}


def judge(rv: np.ndarray, sv: np.ndarray, side: str) -> tuple:
    lo, hi, l15, h15 = np.percentile(rv, [2.5, 97.5, 15, 85])
    md = float(np.median(sv))
    if side == "hi":
        of, ok = float(np.mean(sv > hi)), md <= h15
    else:
        of, ok = float(np.mean((sv < lo) | (sv > hi))), l15 <= md <= h15
    return bool(of <= 0.15 and ok), of, md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--set", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    S = Path(a.s)
    cache = json.loads((S / "d109_cache.json").read_text())
    pool = [v for k, v in cache.items() if k.startswith("2|pool|")]
    rng = np.random.default_rng(116)
    for st in a.set:
        d, c = st.split(":")
        rows, grp = [], []
        for k, v in cache.items():
            if not k.startswith("2|set|"):
                continue
            p = Path(k.split("|")[2])
            if p.parent.name == d and p.name.endswith(f"_{c}.wav"):
                rows.append(v)
                m = re.search(r"_t(\d+)", p.name)
                grp.append(int(m.group(1)) if m else len(grp))
        grp = np.array(grp)
        ug = np.unique(grp)
        rep = {"n": len(rows), "n_pool": len(pool)}
        allp = np.ones(a.n, bool)
        for s in SIDE:
            R = np.array([r[s] for r in pool], float)
            X = np.array([r[s] for r in rows], float)
            R, X_ok = R[np.isfinite(R)], np.isfinite(X)
            if X_ok.sum() == 0:
                continue
            res = []
            for b in range(a.n):
                rb = R[rng.integers(0, len(R), len(R))]
                gb = ug[rng.integers(0, len(ug), len(ug))]
                xb = np.concatenate([X[(grp == g) & X_ok] for g in gb])
                res.append(judge(rb, xb, SIDE[s]))
            ok = np.array([r[0] for r in res])
            allp &= ok
            rep[s] = {"pass_rate": round(float(ok.mean()), 3), "out_frac_5_95": [round(float(z), 3) for z in np.percentile([r[1] for r in res], [5, 95])],
                      "median_5_95": [round(float(z), 3) for z in np.percentile([r[2] for r in res], [5, 95])], "point": judge(R, X[X_ok], SIDE[s])[0]}
        rep["all_pass_rate"] = round(float(allp.mean()), 3)
        rep["verdict"] = "PASS(余裕あり)" if all(rep[s]["pass_rate"] >= 0.95 for s in SIDE if s in rep) else ("FAIL" if any(rep[s]["pass_rate"] < 0.5 for s in SIDE if s in rep) else "INCONCLUSIVE")
        print(st, rep["verdict"], "all", rep["all_pass_rate"], {s: rep[s]["pass_rate"] for s in SIDE if s in rep}, flush=True)
        (ROOT / f"results/conv_p0/d116_{d}_{c}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
