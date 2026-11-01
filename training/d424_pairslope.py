"""D424: 2 つの系の目標をまたいだ帯域の形の傾きの差を、目標でまとめた対のブートストラップで(d399 の prof・参照 = <組>/d37/_work/ref_TTT.wav)。
    uv run python d424_pairslope.py --a "A=<glob>" --b "B=<glob>" --set dev --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from d399_tablefollow import BANDS, H, prof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--set", default="dev")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    refs = {int(f.stem[4:7]): prof(f) for f in sorted((H / a.set / "d37" / "_work").glob("ref_*.wav"))}
    Y = {}
    for spec in (a.a, a.b):
        n, pat = spec.split("=", 1)
        by: dict = {}
        for f in sorted(glob.glob(pat)):
            by.setdefault(int(Path(f).name.split("_t")[1][:3]), []).append(prof(f))
        Y[n] = {t: np.mean(v, 0) for t, v in by.items()}
    na, nb = a.a.split("=")[0], a.b.split("=")[0]
    ks = sorted(set(Y[na]) & set(Y[nb]) & set(refs))
    X = np.array([refs[k] for k in ks]); A = np.array([Y[na][k] for k in ks]); B = np.array([Y[nb][k] for k in ks])

    def sl(ix, M, k):
        x, y = X[ix, k] - X[ix, k].mean(), M[ix, k] - M[ix, k].mean()
        return float((x * y).sum() / (x ** 2).sum())
    rng = np.random.default_rng(424)
    rep = {"n_targets": len(ks)}
    for k, (lo, hi) in enumerate(BANDS):
        d0 = sl(np.arange(len(ks)), B, k) - sl(np.arange(len(ks)), A, k)
        bs = [sl(ix, B, k) - sl(ix, A, k) for ix in (rng.integers(0, len(ks), len(ks)) for _ in range(2000))]
        bias = float((B[:, k] - A[:, k]).mean())
        rep[f"{lo}-{hi}"] = {"dslope": round(d0, 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "dbias_db": round(bias, 2)}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
