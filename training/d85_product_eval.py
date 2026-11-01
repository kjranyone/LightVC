"""D85(学習なし): 実の女声の目標(評価の 30 目標)への変換を Rust の製品経路で作り、話者照合(d4b の学習外 LDA)・自然さ(D78 の共分散・目標の別の発話が基準)・CER(D23)で測る。
元 = D56 の新しい男声 60 文(<s>/d56 の SRC)・目標の登録 = <s>/d37/_work/ref_TTT.wav を作り直す(動きの共分散 res_cov つき)・操作者の登録 = <s>/d56/_work/op_PPPP.wav を作り直す。
条件 = --conds "名前=引数"(; 区切り)・共通 = artic --mix 0 --per-p75 --med-from-operator --warp --alpha 1.3 --operator-enroll + --extra。出力 = d4b 形式(cen_TTT・pNNNN_tNNN_<cond>・SRC)。
    uv run python d85_product_eval.py --s <scratchpad> --out_dir <scratchpad>/d85_360k --extra "--rvoc results/rvtab_full/export_360k" --conds "TAB=--resid 0;W10=--resid 1.0;C10=--resid 1.0 --color;C05=--resid 0.5 --color"
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"


def run(cmd: list) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FAIL {cmd} {r.stderr[-400:]}")


def job(j: tuple) -> str:
    cmd, out = j
    if not Path(out).exists():
        run(cmd)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    S = Path(a.s)
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    work = S / "d85_work"; work.mkdir(exist_ok=True)
    srcs = sorted((S / "d56").glob("p*_SRC.wav"))
    en_jobs = []
    for ti in range(30):
        en_jobs.append(([BIN, "artic-enroll", "-r", S / "d37/_work" / f"ref_{ti:03d}.wav", "-o", work / f"en_{ti:03d}"], str(work / f"en_{ti:03d}" / "enroll.json")))
    for f in srcs:
        pi = int(f.name[1:5])
        en_jobs.append(([BIN, "artic-enroll", "-r", S / "d56/_work" / f"op_{pi:04d}.wav", "-o", work / f"op_{pi:04d}"], str(work / f"op_{pi:04d}" / "enroll.json")))
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(job, en_jobs))
    conds = {c.split("=", 1)[0]: c.split("=", 1)[1].split() for c in a.conds.split(";") if c}
    cv = []
    for f in srcs:
        k = f.name[: -len("_SRC.wav")]
        m = re.match(r"p(\d+)_t(\d+)", k)
        pi, ti = int(m.group(1)), int(m.group(2))
        for c, args in conds.items():
            cv.append(([BIN, "artic", "-i", f, "-o", od / f"{k}_{c}.wav", "--enroll", work / f"en_{ti:03d}", "--operator-enroll", work / f"op_{pi:04d}", "--mix", "0", "--per-p75", "--med-from-operator", "--warp", "--alpha", "1.3"] + args + a.extra.split(), str(od / f"{k}_{c}.wav")))
        shutil.copy(f, od / f.name)
    for c in sorted((S / "d56").glob("cen_*.wav")):
        shutil.copy(c, od / c.name)
    with Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, cv)):
            if i % 50 == 0:
                print("conv", i, len(cv), flush=True)
    print("done", len(cv), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
