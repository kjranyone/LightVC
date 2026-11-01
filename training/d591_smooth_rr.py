"""D591: R の差分(d579_rrG の dS = R − G の包絡 24 行・5 ms)を時間方向に因果な 2 項の平滑 [1, 2, 1] / 4(t, t−1, t−2・遅れ 1 フレーム)にかけた版を新しい置き場へ書く(元は書き換えない)。
帯ごとの通し: 10 Hz 0.98・20 Hz 0.90・40 Hz 0.65・80 Hz 0.10(200 フレーム / 秒)。dC はそのまま。
    uv run python d591_smooth_rr.py --sets dev2 dev3 --src d579_rrG --dst d579_rrGs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")


def smooth(d: np.ndarray) -> np.ndarray:
    p1 = np.concatenate([d[:, :1], d[:, :-1]], 1)
    p2 = np.concatenate([d[:, :1], d[:, :1], d[:, :-2]], 1)
    return ((d + 2 * p1 + p2) / 4).astype(d.dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--src", default="d579_rrG")
    ap.add_argument("--dst", default="d579_rrGs")
    a = ap.parse_args()
    for s in a.sets:
        src, dst = H / s / a.src, H / s / a.dst
        dst.mkdir(exist_ok=True)
        n = 0
        for f in sorted(src.glob("*.npz")):
            z = np.load(f)
            o = {k: z[k] for k in z.files}
            o["dS"] = smooth(z["dS"])
            np.savez(dst / f.name, **o)
            n += 1
        d0 = np.load(sorted(src.glob("*.npz"))[0])["dS"]
        r = float(np.abs(np.diff(smooth(d0), axis=1)).mean() / np.abs(np.diff(d0, axis=1)).mean())
        print(s, "smoothed", n, "frame-diff ratio (first file)", round(r, 3), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
