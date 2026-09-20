"""pitch-blind符号器学習用のWORLDピッチシフト音声プール(f0shift_cacheの合成部のみ)。

data/f0shift_wav/<spk>/<stem>.wav (48k int16) + meta.json {stem: st}。
同stemの元音声との時間整列はWORLD合成の長さ保存(±数ms)に依存する。

    CUDA_VISIBLE_DEVICES=0 uv run python f0shift_wavpool.py --n 4000 --workers 10
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/female_real_feat"
F0FIX = ROOT / "data/female_real_f0fix"
OUT = ROOT / "data/f0shift_wav"
CAP_HZ = 700.0

import sys
sys.path.insert(0, str(Path(__file__).parent))
from f0shift_cache import shift_one


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    stems = []
    for spk in sorted(SRC.iterdir()):
        if spk.is_dir():
            for f in spk.glob("*.pt"):
                if (F0FIX / spk.name / f.name).exists():
                    stems.append(f)
    rng.shuffle(stems)
    stems = stems[:a.n]

    jobs, meta = [], {}
    for f in stems:
        f0f = torch.load(F0FIX / f.parent.name / f.name, map_location="cpu",
                         weights_only=False)["f0"].numpy()
        fv = f0f[f0f > 60]
        if len(fv) < 60:
            continue
        med = float(np.median(fv))
        st_max = min(12.0, 12.0 * np.log2(CAP_HZ / med))
        st_min = max(-4.0, -12.0 * np.log2(med / 65.0))
        if st_max - st_min < 1.0:
            continue
        st = rng.uniform(st_min, st_max)
        od = OUT / f.parent.name
        od.mkdir(parents=True, exist_ok=True)
        wav_out = od / (f.stem + ".wav")
        if wav_out.exists():
            continue
        jobs.append((str(f), str(wav_out),
                     str(F0FIX / f.parent.name / f.name), st))
        meta[f.stem] = round(st, 3)
    print(f"  jobs {len(jobs)}", flush=True)
    t0 = time.time()
    errs = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (_, msg) in enumerate(ex.map(shift_one, jobs, chunksize=8)):
            if msg and ":" in msg:
                errs += 1
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(jobs)} err {errs} ({time.time()-t0:.0f}s)",
                      flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta))
    print(json.dumps({"jobs": len(jobs), "errs": errs,
                      "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
