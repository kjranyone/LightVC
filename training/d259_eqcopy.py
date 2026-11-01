"""D259(学習なし・results/conv_p0/d259_prereg.yaml): 実女声(SRC)と実女声 × EQ(D257 の SRCEQ)を、新しい前段 + rvoc3hi で自分自身へ写し合成する。
    uv run python d259_eqcopy.py --s <dev の親>
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d257_eq import BIN, job

COPY = ["--rvoc", "results/rvoc3hi/export", "--mix", "1", "--med-from-operator", "--causal-ct", "--per-p75", "--f4", "results/f4_1/export"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    a = ap.parse_args()
    F = Path(a.s) / "devff"
    od = F / "d259"
    od.mkdir(exist_ok=True)
    jobs = []
    for f in sorted((F / "d56").glob("p00*_SRC.wav")):
        pi = int(f.name[1:5])
        op = F / "d85_work" / f"op_{pi:04d}"
        k = f.stem[:-4]
        jobs.append([BIN, "artic", "--enroll", op, "--operator-enroll", op] + COPY + ["-i", f, "-o", od / f"{k}_CSRC.wav"])
        jobs.append([BIN, "artic", "--enroll", op, "--operator-enroll", op] + COPY + ["-i", F / "d257" / f"{k}_SRCEQ.wav", "-o", od / f"{k}_CSRCEQ.wav"])
    with Pool(8) as pl:
        list(pl.imap_unordered(job, jobs))
    print("written", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
