"""D230(学習なし・物差しの検査の対照・results/conv_p0/d230_prereg.yaml): 開発用の組の源(d56/p*_SRC.wav)を WORLD で分析し、
f0 だけを目標の参照の声域へ移した PSHIFT と、さらに包絡を一律 1.2 倍に伸ばした PSHIFTW を <s>/d230/ に書く(診断専用・製品の経路でない)。
    uv run python d230_pshift.py --s <dev の組>
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from d6_unit_cov import load48

SR = 48000
WARP = 1.2


def job(args: tuple) -> str:
    src, f0_ref, od = args
    import pyworld as pw
    x = load48(src).astype(np.float64)
    f0, t = pw.harvest(x, SR, f0_floor=60.0, f0_ceil=600.0, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    v = f0 > 0
    r = f0_ref / float(np.median(f0[v]))
    f1 = np.where(v, f0 * r, 0.0)
    stem = Path(src).name.replace("_SRC.wav", "")
    sf.write(Path(od) / f"{stem}_PSHIFT.wav", np.clip(pw.synthesize(f1, sp, ap, SR, 5.0), -1, 1).astype(np.float32), SR)
    nb = sp.shape[1]
    fr = np.arange(nb)
    spw = np.stack([np.interp(fr / WARP, fr, s) for s in sp])
    apw = np.stack([np.interp(fr / WARP, fr, s) for s in ap])
    sf.write(Path(od) / f"{stem}_PSHIFTW.wav", np.clip(pw.synthesize(f1, spw, apw, SR, 5.0), -1, 1).astype(np.float32), SR)
    return f"{stem} r={r:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    S = Path(a.s)
    meta = {m["ti"]: m for m in json.loads((S / "cen_reg/meta.json").read_text())}
    od = S / "d230"
    od.mkdir(exist_ok=True)
    jobs = [(str(f), meta[int(f.name.split("_t")[1][:3])]["f0_ref"], str(od)) for f in sorted((S / "d56").glob("p*_SRC.wav"))]
    with Pool(a.workers) as pool:
        out = pool.map(job, jobs)
    print(len(out), out[:3], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
