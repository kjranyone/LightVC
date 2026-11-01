"""D122(学習なし): D121 の c0 の表(c0 も『目標の表 + 操作者の残差』)を Rust の製品経路(実女声 30 目標・男声 60 組)で。
登録をし直す(t0 = 単位ごとの c0 の話者平均からの相対値を足した Rust の artic-enroll・表そのものは従来と同じ)→ C07S05T = C07S05B(γ0.7 + 色 + β0.5 + C07S05 の較正)+ `--c0-table`。
出力 = <s>/d122/pNNNN_tNNN_C07S05T.wav(+ 比較の C07S05B・SRC)。関門・CER・LDA は呼び出し側で。
    uv run python d122_c0_table.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
EXTRA = ["--rvoc", "results/rvtab_x/export", "--f4", "results/f4_1/export", "--f0-conf", "0.5", "--resid", "0.7", "--color", "--soft", "0.5", "--c0-table"]


def job(cmd: list) -> None:
    out = cmd[cmd.index("-o") + 1]
    done = Path(out) / "enroll.json" if cmd[1] == "artic-enroll" else Path(out)
    if not done.exists():
        subprocess.run([str(c) for c in cmd], cwd=ROOT, env={"PATH": "/usr/bin:/bin"}, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    S = Path(a.s)
    work, cal, od = S / "d122_work", S / "d114/_cal_C07S05", S / "d122"
    work.mkdir(exist_ok=True); od.mkdir(exist_ok=True)
    srcs = sorted((S / "d56").glob("p*_SRC.wav"))
    en = [[BIN, "artic-enroll", "-r", S / "d37/_work" / f"ref_{t:03d}.wav", "-o", work / f"en_{t:03d}"] for t in range(30)]
    en += [[BIN, "artic-enroll", "-r", S / "d56/_work" / f"op_{int(f.name[1:5]):04d}.wav", "-o", work / f"op_{int(f.name[1:5]):04d}"] for f in srcs]
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(job, en))
    jobs = []
    for f in srcs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        k = f.stem[:-4]
        jobs.append([BIN, "artic", "-i", f, "-o", od / f"{k}_C07S05T.wav", "--enroll", work / f"en_{ti:03d}", "--operator-enroll", work / f"op_{pi:04d}", "--mix", "0", "--per-p75",
                     "--med-from-operator", "--warp", "--alpha", "1.3", "--env-bias", cal / f"bias_{pi:04d}_t{ti:03d}.f32"] + EXTRA)
        shutil.copy(S / "d114" / f"{k}_C07S05B.wav", od / f"{k}_C07S05B.wav")
        shutil.copy(f, od / f.name)
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(job, jobs))
    print("done", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
