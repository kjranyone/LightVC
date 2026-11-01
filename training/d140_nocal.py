"""D140(学習なし): 鋭さ β × 登録の較正 × 動きの量 γ の組み合わせのうち較正なしの構成を Rust の製品経路で鳴らす(較正ありは d114_env_bias_calib.py --cfg)。
D139: 表だけ(β1・較正なし)は同一性 0.40 だが、柔らかい引き + 較正の表だけは 0.12 = 自然さのための処置が同一性を削った疑いの帰属。
出力 = <s>/d114/pNNNN_tNNN_<cfg>N.wav(N = 較正なし)。
    uv run python d140_nocal.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import subprocess
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
EXTRA = ["--rvoc", "results/rvtab_x/export", "--f4", "results/f4_1/export", "--f0-conf", "0.5"]
CFG = {"G07S03": ["--resid", "0.7", "--color", "--soft", "0.3"], "G00S03": ["--resid", "0", "--soft", "0.3"], "G03S03": ["--resid", "0.3", "--color", "--soft", "0.3"],
       "G03S10": ["--resid", "0.3", "--color", "--soft", "1"]}


def job(cmd: list) -> None:
    out = cmd[cmd.index("-o") + 1]
    if not Path(out).exists():
        subprocess.run([str(c) for c in cmd], cwd=ROOT, env={"PATH": "/usr/bin:/bin"}, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    S = Path(a.s)
    work, od = S / "d85_work", S / "d114"
    jobs = []
    for f in sorted((S / "d56").glob("p*_SRC.wav")):
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        for c, args in CFG.items():
            jobs.append([BIN, "artic", "-i", f, "-o", od / f"{f.stem[:-4]}_{c}N.wav", "--enroll", work / f"en_{ti:03d}", "--operator-enroll", work / f"op_{pi:04d}",
                         "--mix", "0", "--per-p75", "--med-from-operator", "--warp", "--alpha", "1.3"] + EXTRA + args)
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(job, jobs))
    print("done", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
