"""D80a(学習なし): DTW の物差し(dtw_ruler.py)が既存の耳の合否を再現するかを、写し合成の刺激(D77 と同じ組・同じラベル)で確かめる。出力 = results/conv_p0/d80a.json。
    uv run python d80a_dtw_validate.py --s <scratchpad> --workers 10
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from d77_ruler import FAIL, ORDERS, PASS, SETS, rd

ROOT = Path(__file__).resolve().parent.parent


def one(job: tuple) -> list:
    import dtw_ruler as DR
    set_, key, items, srcp = job
    ax = DR.analyse(rd(Path(srcp)))
    return [{"set": set_, "key": key, "cond": c, **DR.dist(DR.analyse(rd(Path(p))), ax)} for c, p in items]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    S = Path(a.s)
    jobs = []
    for d, conds in SETS.items():
        for f in sorted((S / d).glob("*_SRC.wav")):
            key = f.name[: -len("_SRC.wav")]
            jobs.append((d, key, [(c, str(S / d / f"{key}_{c}.wav")) for c in conds if (S / d / f"{key}_{c}.wav").exists()], str(f)))
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(one, jobs):
            rows.extend(r)
    by = {}
    for r in rows:
        by.setdefault((r["set"], r["key"]), {})[r["cond"]] = r
    rep = {}
    for m in ["env", "lm", "lm_v"]:
        pv = [r[m] for r in rows if (r["set"], r["cond"]) in PASS and np.isfinite(r[m])]
        fv = [r[m] for r in rows if (r["set"], r["cond"]) in FAIL and np.isfinite(r[m])]
        auc = float(np.mean([[1.0 if p < f else 0.5 if p == f else 0.0 for f in fv] for p in pv]))
        orders = {s_: {c: round(float(np.nanmean([cs[c][m] for (ss_, k_), cs in by.items() if ss_ == s_ and c in cs])), 4) for c in od} for s_, od in ORDERS.items()}
        rep[m] = {"auc": round(auc, 3), "pass_p90": round(float(np.percentile(pv, 90)), 4), "fail_p10": round(float(np.percentile(fv, 10)), 4), "orders": orders}
        print(m, json.dumps(rep[m], ensure_ascii=False), flush=True)
    (ROOT / "results/conv_p0/d80a.json").write_text(json.dumps({"summary": rep, "rows": rows}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
