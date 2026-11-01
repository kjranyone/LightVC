"""D625: S の尾の解剖。LVC_PROF の書き出し([ブロック数, 10] f32 = ブロックの時間 + 段 9 個 ms)を読み、尾(自分の走行の p99 以上)と中(p40〜p60)の段ごとの平均の差を出す。
規則は results/conv_p0/d625/prereg.yaml。
    uv run python d625_prof.py --dir ../results/conv_p0/d625
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ST = ["front", "c1_wait", "f1_wait", "cct", "c2_wait", "cond_rr", "exc", "voc", "h1eq"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    rep: dict = {}
    for f in sorted(Path(a.dir).glob("prof_*.f32")):
        x = np.fromfile(f, "<f4").reshape(-1, 10)
        t, s = x[:, 0], x[:, 1:]
        p40, p60, p99 = np.percentile(t, [40, 60, 99])
        tail, mid = t >= p99, (t >= p40) & (t <= p60)
        ex = s[tail].mean(0) - s[mid].mean(0)
        tot = float(t[tail].mean() - t[mid].mean())
        rr_even = s[0::2, 5].mean(), s[1::2, 5].mean()
        r = {"n": len(t), "mean": round(float(t.mean()), 4), "p99": round(float(p99), 4), "max": round(float(t.max()), 4),
             "stage_mean": dict(zip(ST, np.round(s.mean(0), 4).tolist())), "stage_p99": dict(zip(ST, np.round(np.percentile(s, 99, 0), 4).tolist())),
             "tail_minus_mid": dict(zip(ST, np.round(ex, 4).tolist())), "tail_excess_total": round(tot, 4),
             "share": dict(zip(ST, np.round(ex / max(tot, 1e-9), 3).tolist())), "unaccounted": round(float(tot - ex.sum()), 4),
             "cond_rr_even_odd": [round(float(v), 4) for v in rr_even], "tail_idx_mod2": np.bincount(np.flatnonzero(tail) % 2, minlength=2).tolist()}
        rep[f.stem] = r
        print(f.stem, json.dumps({k: r[k] for k in ("mean", "p99", "max", "tail_excess_total", "unaccounted")}), flush=True)
        print("   mean ", r["stage_mean"], flush=True)
        print("   share", r["share"], "tail parity", r["tail_idx_mod2"], "cond_rr even/odd", r["cond_rr_even_odd"], flush=True)
    Path(a.dir, "prof.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
