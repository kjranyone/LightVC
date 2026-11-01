"""段 1 の基準(current/e2e_ppg.md)の評価の入力: dev の男声 → 女声 60 組の C1 の事後・条件(高さと周期性を目標へ = 操作者の登録の中央値 → 目標)と、目標 30 人の参照の自分自身への写し合成の条件(要約の元)を Rust で書き出す(描画は省く)。
    uv run python d274_evaldump.py --s <dev の親> --set dev --out <dev の親>/e2e
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d273_data import BIN, job

COM = ["--mix", "1", "--causal-ct", "--per-p75", "--f4", "results/f4_1/export", "--rvoc", "results/rvoc3hi/export"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--set", default="dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="p00*_SRC.wav", help="操作者の入力(devtts は p0*_SRC.wav = 女声 TTS p00・男声 TTS p01)")
    a = ap.parse_args()
    D, O = Path(a.s) / a.set, Path(a.out) / f"eval_{a.set}"
    O.mkdir(parents=True, exist_ok=True)
    jobs, tis = [], set()
    for f in sorted((D / "d56").glob(a.glob)):
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        tis.add(ti)
        jobs.append(([BIN, "artic", "--enroll", D / "d85_work" / f"en_{ti:03d}", "--operator-enroll", D / "d85_work" / f"op_{pi:04d}", "--med-from-operator"] + COM + ["-i", f], O / f"{f.stem[:-4]}.cond", O / f"{f.stem[:-4]}.front"))
    for ti in sorted(tis):
        en = D / "d85_work" / f"en_{ti:03d}"
        jobs.append(([BIN, "artic", "--enroll", en, "--operator-enroll", en, "--med-from-operator"] + COM + ["-i", D / "d37/_work" / f"ref_{ti:03d}.wav"], O / f"ref_{ti:03d}.cond", O / f"ref_{ti:03d}.front"))
    with Pool(6) as pl:
        list(pl.imap_unordered(job, jobs))
    print("eval dumped", sum(Path(c).exists() for _, c, _ in jobs), "/", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
