"""results/bench_cpu_swap の要約と前後の比較。
    uv run python bench_summary.py before [after]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent.parent / "results/bench_cpu_swap"


def one(lb: str) -> dict:
    d = B / lb
    rows = [json.loads(l) for l in (B / f"train_{lb}" / "train.jsonl").read_text().splitlines() if l.strip() and '"sec"' in l]
    sec = {r["step"]: r["sec"] for r in rows}
    st = sorted(k for k in sec if k >= 100)
    iv = [(sec[b] - sec[a]) / (b - a) for a, b in zip(st[:-1], st[1:])]
    w = json.loads((d / "wall.json").read_text())
    total = w["t_end"] - w["t_start"]
    gpu = []
    for l in (d / "gpu.csv").read_text().splitlines():
        p = [q.strip() for q in l.split(",")]
        if len(p) >= 2 and p[1].isdigit():
            gpu.append(int(p[1]))
    g = np.array(gpu)
    gs = g[np.flatnonzero(g > 0)[0]:] if (g > 0).any() else g
    cpu = []
    for l in (d / "vmstat.txt").read_text().splitlines():
        p = l.split()
        if len(p) > 15 and p[0].isdigit():
            cpu.append(100 - int(p[14]))
    c = json.loads((d / "cpu.json").read_text())
    return {"s_per_step": round((sec[st[-1]] - sec[st[0]]) / (st[-1] - st[0]), 4), "interval_median": round(float(np.median(iv)), 4), "interval_min": round(min(iv), 4), "interval_max": round(max(iv), 4),
            "startup_s": round(total - sec[max(sec)], 1), "gpu_util_mean_training": round(float(gs.mean()), 1) if len(gs) else None, "cpu_busy_mean": round(float(np.mean(cpu)), 1) if cpu else None,
            **{k: v["median"] for k, v in c.items() if isinstance(v, dict)}}


def main() -> int:
    labs = sys.argv[1:]
    res = {lb: one(lb) for lb in labs}
    keys = list(next(iter(res.values())).keys())
    print("項目".ljust(26), *[lb.rjust(12) for lb in labs], *(["後/前".rjust(8)] if len(labs) == 2 else []))
    for k in keys:
        v = [res[lb][k] for lb in labs]
        r = f"{v[1] / v[0]:.3f}".rjust(8) if len(labs) == 2 and v[0] else ""
        print(k.ljust(26), *[str(x).rjust(12) for x in v], r)
    if len(labs) == 2:
        a, b = res[labs[0]], res[labs[1]]
        same = not (b["interval_max"] < a["interval_min"] or a["interval_max"] < b["interval_min"])
        print("学習の s/step:", "区間が重なる = 差なし" if same else "区間が重ならない = 差あり")
    (B / f"summary_{'_'.join(labs)}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
