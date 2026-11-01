"""D265 の変換(results/conv_p0/d265_prereg.yaml): devtts の FT(女声 TTS・α1)/ MT(男声 TTS・α1.3)を全体写像と表 γ0.7 + 色で(--causal-ct --floor-apply・rvoc3hi)。
    uv run python d265_conv.py --s <dev の親>
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d257_eq import BIN, job

BASE = ["--rvoc", "results/rvoc3hi/export", "--mix", "0", "--per-p75", "--med-from-operator", "--f4", "results/f4_1/export", "--warp", "--causal-ct", "--floor-apply"]
RUNGS = {"GM": ["--global-map"], "T07": ["--resid", "0.7", "--color"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    a = ap.parse_args()
    F = Path(a.s) / "devtts"
    od = F / "d265"
    od.mkdir(exist_ok=True)
    jobs = []
    for f in sorted((F / "d56").glob("p0*_SRC.wav")):
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        if not (F / "d85_work" / f"op_{pi:04d}" / "enroll.json").exists():
            continue
        al = "1.0" if pi < 100 else "1.3"
        for r, opt in RUNGS.items():
            jobs.append([BIN, "artic", "--enroll", F / "d85_work" / f"en_{ti:03d}", "--operator-enroll", F / "d85_work" / f"op_{pi:04d}"] + BASE + ["--alpha", al] + opt + ["-i", f, "-o", od / f"{f.stem[:-4]}_{r}.wav"])
    with Pool(8) as pl:
        list(pl.imap_unordered(job, jobs))
    print("written", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
