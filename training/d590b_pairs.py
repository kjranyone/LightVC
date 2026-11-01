"""D590b: d590 のキャッシュ(<S>/d590_cache.json)から版どうしの shimmer_p / shimmer のファイルごとの対の差(同じ p####_t####)を目標でまとめたブートストラップで。
    uv run python d590b_pairs.py --s <scratch> --pairs OURS5RY58G-OURS5Y58 ... [--stat shimmer_p] --out <json>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--stat", default="shimmer_p")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows: dict = {}
    for s in a.s:
        c = json.loads((Path(s) / "d590_cache.json").read_text())
        for k, v in c.items():
            if not k.startswith("set|"):
                continue
            p = Path(k.split("|")[1])
            m = re.match(r"(p\d+_t(\d+))_(.+)\.wav$", p.name)
            if not m:
                continue
            rows.setdefault(m.group(3), {})[(s, m.group(1))] = (int(m.group(2)), v[a.stat])
    rng = np.random.default_rng(590)
    out = {}
    for pr in a.pairs:
        A, B = pr.split("-")
        ks = sorted(set(rows.get(A, {})) & set(rows.get(B, {})))
        d = np.array([rows[A][k][1] - rows[B][k][1] for k in ks])
        tg = np.array([f"{k[0]}|{rows[A][k][0]}" for k in ks])
        ok = np.isfinite(d)
        d, tg = d[ok], tg[ok]
        ut = np.unique(tg)
        by = {t: d[tg == t] for t in ut}
        bs = []
        for _ in range(4000):
            pick = rng.choice(ut, len(ut))
            bs.append(np.concatenate([by[t] for t in pick]).mean())
        out[pr] = {"n": int(len(d)), "targets": int(len(ut)), "diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
        print(pr, out[pr], flush=True)
    Path(a.out).write_text(json.dumps({"stat": a.stat, "pairs": out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
