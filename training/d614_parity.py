"""D614 の Rust 一致の検査: Rust の条件の書き出し(cond_ppg・d_cond 295)を --h1fix 4 の有無で取り、行 1..25 の差を
train_ppgvoc.h1fix_delta(lf = 列 25・vv = 列 26・σ 4・ft = exp(登録の mu))とフレームごとに比べる。他の列の差は 0 であること。
    uv run python d614_parity.py --dir <書き出しの置き場> --set dev2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import train_ppgvoc as P

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
D_COND = 295


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--set", default="dev2")
    ap.add_argument("--sigma", type=float, default=4.0)
    a = ap.parse_args()
    d = Path(a.dir)
    worst_d, worst_o, nv, n = 0.0, 0.0, 0, 0
    for f0 in sorted(d.glob("*_h0.cond")):
        key = f0.name[:10]
        ti = int(key[7:10])
        mu = json.loads((H / a.set / "d85_work" / f"en_{ti:03d}" / "enroll.json").read_text())["mu"]
        c0 = np.fromfile(f0, "<f4").reshape(-1, D_COND)
        c1 = np.fromfile(d / f"{key}_h{a.sigma:g}.cond", "<f4").reshape(-1, D_COND)
        T = min(len(c0), len(c1))
        c0, c1 = c0[:T], c1[:T]
        lf = torch.from_numpy(c0[:, 25].copy())[None, None]
        vv = torch.from_numpy(c0[:, 26].copy())[None, None]
        ref = P.h1fix_delta(lf, vv, a.sigma, float(np.exp(mu)))[0].numpy().T
        dd = float(np.abs((c1[:, 1:25] - c0[:, 1:25]) - ref).max())
        oth = np.r_[0, np.arange(25, D_COND)]
        do = float(np.abs(c1[:, oth] - c0[:, oth]).max())
        worst_d, worst_o = max(worst_d, dd), max(worst_o, do)
        nv += int((c0[:, 26] > 0.5).sum())
        n += T
        print(key, "frames", T, "voiced", int((c0[:, 26] > 0.5).sum()), "max|Δ − h1fix_delta|", f"{dd:.2e}", "max|other cols|", f"{do:.2e}", "max|δ|", f"{float(np.abs(ref).max()):.3f}", flush=True)
    ok = worst_d < 1e-4 and worst_o == 0.0 and nv > 0
    print(json.dumps({"files": len(list(d.glob("*_h0.cond"))), "frames": n, "voiced": nv, "max_delta_err": worst_d, "max_other_cols": worst_o, "pass": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
