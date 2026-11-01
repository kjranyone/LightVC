"""D560(推論だけ・男声 → 女声): 目標の参照(<s>/d37/_work/ref_TTT.wav 全体)の発声中(c0n > 0.5)の c0n 平均 = 目標の登録の定数。
D558 の水準ずらしの行き先を実女声の平均 4.40 から目標本人の水準に替えるための材料(Rust の書き出し = 評価の書き出しと同じ設定・条件だけ)。
    uv run python d560_target_level.py --sets dev2 dev3 devreal devvctkm devff23x --out ../results/conv_p0/d560_tlevel.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from d273_data import BIN, job
from d274_evaldump import COM
import train_ppgvoc as P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for s in a.sets:
        S = P.H / s
        W = P.H / "e2e" / f"tref_{s}"
        W.mkdir(parents=True, exist_ok=True)
        n = len(json.loads((S / "devset.json").read_text())["targets"])
        jobs = [([BIN, "artic", "--enroll", S / "d85_work" / f"en_{t:03d}"] + COM + ["-i", S / "d37" / "_work" / f"ref_{t:03d}.wav"], W / f"t{t:03d}.cond", W / f"t{t:03d}.front") for t in range(n)]
        with Pool(3) as pl:
            list(pl.imap_unordered(job, jobs))
        lv = {}
        for t in range(n):
            c = np.fromfile(W / f"t{t:03d}.cond", "<f4").reshape(-1, 31)[:, 0]
            v = c[c > 0.5]
            lv[t] = float(v.mean()) if len(v) > 50 else None
        x = np.array([v for v in lv.values() if v is not None])
        res[s] = {"level": lv, "p10_p50_p90": [round(float(q), 3) for q in np.percentile(x, [10, 50, 90])]}
        print(s, res[s]["p10_p50_p90"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
