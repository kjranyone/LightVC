"""D83: d4b 形式の写し合成(pNNNN_tNNN_<cond>.wav + SRC)の参照 PESQ(D77 の物差し: 耳の合否 AUC 0.996・SRC と RMS を揃える・16k wb)。
    uv run python d83_pesq.py --dir <dir> --conds BASE,FT --out ../results/rvtab_full/d83_pesq_335k.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def one(job: tuple) -> dict:
    import librosa
    from pesq import pesq
    from d77_ruler import rd
    from quality_gate import align
    key, src, items = job
    x = rd(Path(src))
    out = {"key": key}
    for c, p in items:
        y = rd(Path(p))
        _, y = align(x, y)
        y = np.pad(y[: len(x)], (0, max(0, len(x) - len(y))))
        y = y * np.sqrt((x ** 2).mean() / max((y ** 2).mean(), 1e-12))
        out[c] = float(pesq(16000, librosa.resample(x, orig_sr=48000, target_sr=16000), librosa.resample(y.astype(np.float32), orig_sr=48000, target_sr=16000), "wb"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    d = Path(a.dir)
    conds = a.conds.split(",")
    jobs = []
    for f in sorted(d.glob("*_SRC.wav")):
        key = f.name[: -len("_SRC.wav")]
        jobs.append((key, str(f), [(c, str(d / f"{key}_{c}.wav")) for c in conds if (d / f"{key}_{c}.wav").exists()]))
    with Pool(a.workers) as pool:
        rows = pool.map(one, jobs)
    summ = {c: round(float(np.mean([r[c] for r in rows if c in r])), 4) for c in conds}
    if len(conds) == 2:
        summ[f"{conds[1]}_gt_{conds[0]}"] = int(sum(r[conds[1]] > r[conds[0]] for r in rows))
    print(json.dumps(summ), flush=True)
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
