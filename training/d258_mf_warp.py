"""D258(学習なし・results/conv_p0/d258_prereg.yaml): 男声 → 女声(dev の MF 60)で Δ を使わず伸縮 α だけ(1.0 / 1.15 / 1.3)+ 出力の EQ。
    uv run python d258_mf_warp.py --s <dev の親>
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import soundfile as sf
from d257_eq import BIN, SR, eq, job, load48, ltas

ROOT = Path(__file__).resolve().parent.parent
BASE = ["--rvoc", "results/rvoc3hi/export", "--mix", "0", "--per-p75", "--med-from-operator", "--f4", "results/f4_1/export", "--warp", "--causal-ct", "--global-map", "--gmap-scale", "0"]
ALPHAS = {"A100": "1.0", "A115": "1.15", "A130": "1.3"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    a = ap.parse_args()
    F = Path(a.s) / "dev"
    od = F / "d258"
    (od / "_cal").mkdir(parents=True, exist_ok=True)
    srcs = sorted((F / "d56").glob("p00*_SRC.wav"))
    jobs = []
    for f in srcs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        en, op = F / "d85_work" / f"en_{ti:03d}", F / "d85_work" / f"op_{pi:04d}"
        for r, al in ALPHAS.items():
            jobs.append([BIN, "artic", "--enroll", en, "--operator-enroll", op] + BASE + ["--alpha", al, "-i", f, "-o", od / f"{f.stem[:-4]}_{r}.wav"])
            jobs.append([BIN, "artic", "--enroll", en, "--operator-enroll", op] + BASE + ["--alpha", al, "-i", F / "d56/_work" / f"op_{pi:04d}.wav", "-o", od / "_cal" / f"op{pi:04d}_t{ti:03d}_{r}.wav"])
    with Pool(8) as pl:
        list(pl.imap_unordered(job, jobs))
    tl: dict = {}
    for f in srcs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        k = f.stem[:-4]
        if ti not in tl:
            tl[ti] = ltas(load48(F / "d37/_work" / f"ref_{ti:03d}.wav"))
        for r in ALPHAS:
            g = tl[ti] - ltas(load48(od / "_cal" / f"op{pi:04d}_t{ti:03d}_{r}.wav"))
            sf.write(od / f"{k}_{r}Q.wav", eq(load48(od / f"{k}_{r}.wav"), g, 1.0), SR)
    print("written", len(srcs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
