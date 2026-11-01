"""D262(学習なし・results/conv_p0/d262_prereg.yaml): 包絡の差 Δ を下限つきの log-mel の領域で足す(--gmap-floor)と条件の側の不整合が消えるか(devff の FF 60)。
    uv run python d262_floor.py --s <dev の親>
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d257_eq import BIN, job

BASE = ["--rvoc", "results/rvoc3hi/export", "--mix", "0", "--per-p75", "--med-from-operator", "--f4", "results/f4_1/export", "--warp", "--causal-ct", "--global-map", "--gmap-floor", "--alpha", "1"]
RUNGS = {"FL0": ["--gmap-scale", "0", "--no-f0-shift", "--f4-no-offset"], "FL_ONLY": ["--gmap-scale", "1", "--no-f0-shift", "--f4-no-offset"], "FL_ALL": ["--gmap-scale", "1"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    a = ap.parse_args()
    F = Path(a.s) / "devff"
    od = F / "d262"
    od.mkdir(exist_ok=True)
    jobs = []
    for f in sorted((F / "d56").glob("p00*_SRC.wav")):
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        for r, opt in RUNGS.items():
            jobs.append([BIN, "artic", "--enroll", F / "d85_work" / f"en_{ti:03d}", "--operator-enroll", F / "d85_work" / f"op_{pi:04d}"] + BASE + opt + ["-i", f, "-o", od / f"{f.stem[:-4]}_{r}.wav"])
    with Pool(8) as pl:
        list(pl.imap_unordered(job, jobs))
    print("written", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
