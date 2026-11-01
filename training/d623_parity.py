"""D623 の Rust 一致の検査: Rust の製品の経路の --h1eq 3 あり(B)と無し(A・条件の書き出しつき)を、Python の d619_h1eq.h1eq(A, f0)と比べる。
Rust はブロック t に 1 フレーム前の f0(書き出しのフレーム t − 1)を使う = Python の h1eq に f0 を 1 フレーム遅らせて渡す。大きさは二乗平均でそろえる(ゲインは比で決まり大きさに依らない)。
    uv run python d623_parity.py --dir <置き場> --set dev2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from d619_h1eq import h1eq

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
D_COND = 295


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--set", default="dev2")
    ap.add_argument("--target", type=float, default=3.0)
    ap.add_argument("--fmax", type=float, default=float("inf"))
    a = ap.parse_args()
    d = Path(a.dir)
    worst, rows = 0.0, []
    for fa in sorted(d.glob("*_A.wav")):
        key = fa.name[:10]
        ti = int(key[7:10])
        ft = float(np.exp(json.loads((H / a.set / "d85_work" / f"en_{ti:03d}" / "enroll.json").read_text())["mu"]))
        ya, _ = sf.read(fa)
        yb, _ = sf.read(d / f"{key}_B.wav")
        c = np.fromfile(d / f"{key}_A.cond", "<f4").reshape(-1, D_COND)
        f0 = np.where(c[:, 26] > 0.5, 200.0 * np.exp(c[:, 25].astype(np.float64)), 0.0)
        f0s = np.r_[0.0, f0[:-1]]
        yp, gs = h1eq(ya, f0s, ft, a.target, a.fmax)
        n = min(len(yp), len(yb))
        yp, yb2, ya2 = yp[:n] / np.sqrt((yp[:n] ** 2).mean()), yb[:n] / np.sqrt((yb[:n] ** 2).mean()), ya[:n] / np.sqrt((ya[:n] ** 2).mean())
        e = float(np.abs(yp - yb2).max())
        eff = float(np.abs(ya2 - yb2).max())
        worst = max(worst, e)
        rows.append((key, e, eff, float(gs[f0s[: len(gs)] > 0].mean())))
        print(key, "max|Python − Rust| (RMS 1)", f"{e:.2e}", "max|A − B| (効き)", f"{eff:.2e}", "mean voiced gain dB", round(rows[-1][3], 2), flush=True)
    ok = worst < 1e-3 and len(rows) > 0 and all(r[2] > 0.05 for r in rows)
    print(json.dumps({"files": len(rows), "max_err": worst, "pass": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
