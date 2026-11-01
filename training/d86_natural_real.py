"""D86(学習なし): d4b 形式(pNNNN_tNNN_<cond>.wav・cen_TTT)の変換を、話者基準の自然さの物差し(D78: 動きの共分散・帯 RMS)で測る。基準 = 目標の登録の参照 <s>/d37/_work/ref_TTT.wav(評価の発話とは別)。
    uv run python d86_natural_real.py --s <scratchpad> --dir <scratchpad>/d85_360k --conds TAB,W10,C10,C05 --out ../results/rvtab_full/eval_360k/natural_real.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from d77_ruler import rd
from d78_spk_ruler import dist, feats, init, ref_feats


def one(job: tuple) -> list:
    S, d, ti, items = job
    r = ref_feats(rd(Path(S) / "d37/_work" / f"ref_{ti:03d}.wav"))
    return [{"key": k, "cond": c, **dist(feats(rd(Path(p))), r)} for k, c, p in items]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    d = Path(a.dir)
    by: dict = {}
    for c in a.conds.split(","):
        for f in sorted(d.glob(f"p*_t*_{c}.wav")):
            m = re.match(r"(p\d+_t(\d+))_", f.name)
            by.setdefault(int(m.group(2)), []).append((m.group(1), c, str(f)))
    for ti in list(by):
        cen = d / f"cen_{ti:03d}.wav"
        if cen.exists():
            by[ti].append((f"cen_{ti:03d}", "REAL", str(cen)))
    rows = []
    with Pool(a.workers, initializer=init) as pool:
        for r in pool.imap_unordered(one, [(a.s, str(d), ti, it) for ti, it in by.items()]):
            rows.extend(r)
    summ = {m: {c: round(float(np.nanmean([r[m] for r in rows if r["cond"] == c])), 4) for c in ["REAL"] + a.conds.split(",")} for m in ["cov", "rms", "ctr"]}
    for m, v in summ.items():
        print(m, json.dumps(v), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
