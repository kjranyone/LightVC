"""CPU 交換の前後の計時(results/bench_cpu_swap): 測定の経路の CPU の処理 = harvest(1 本ずつ・Pool 8・Pool 12)と WORLD の分析 + 合成(1 本ずつ)。入力は dev2 の SRC 48 本の先頭 8 s(固定)。
    uv run python bench_cpu.py --out ../results/bench_cpu_swap/<label>/cpu.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import H, SR, harvest


def world(x: np.ndarray) -> int:
    f0 = harvest(x)
    t = np.arange(len(f0)) * 0.005
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    return len(pw.synthesize(f0, sp, ap, SR, 5.0))


def pooled(n: int, xs: list) -> list:
    with Pool(n) as pl:
        return pl.map(harvest, xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="約 1 分: harvest 1 本ずつ 8 本・Pool 8 で 48 本・WORLD 2 本・各 2 回")
    a = ap.parse_args()
    from d6_unit_cov import load48
    fs = sorted((H / "dev2" / "d56").glob("p*_SRC.wav"))[:48]
    xs = [load48(f).astype(np.float64)[: 8 * SR] for f in fs]
    rep: dict = {"n_files": len(xs), "sec_audio_each": 8}
    if a.quick:
        jobs = (("harvest_serial_8files", lambda: [harvest(x) for x in xs[:8]]), ("harvest_pool8", lambda: pooled(8, xs)), ("world_serial_2files", lambda: [world(x) for x in xs[:2]]))
    else:
        jobs = (("harvest_serial", lambda: [harvest(x) for x in xs]), ("harvest_pool8", lambda: pooled(8, xs)), ("harvest_pool12", lambda: pooled(12, xs)), ("world_serial_8files", lambda: [world(x) for x in xs[:8]]))
    for nm, fn in jobs:
        ts = []
        for _ in range(2 if a.quick else a.reps):
            t0 = time.perf_counter()
            fn()
            ts.append(round(time.perf_counter() - t0, 3))
        rep[nm] = {"sec": ts, "median": float(np.median(ts))}
        print(nm, rep[nm], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
