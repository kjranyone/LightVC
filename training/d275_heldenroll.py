"""train_ppgvoc.py の評価(保留の実女声 30 人の自己再構成)用に、各話者の参照(評価のクリップ A を除く female-dataset のファイルを名前順に 20 s 以上つなぐ・48 kHz)を作り、
製品の登録(Rust artic-enroll)で表を作る。男声 → 女声の評価(d85_work/en_TTT)と同じ作り方。
    uv run python d275_heldenroll.py
"""
from __future__ import annotations

import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48
from d273_data import BIN
from train_ppgvoc import H, ROOT, e2e_held

O = H / "e2e" / "held_en"


def one(job: tuple) -> str:
    i, spk, ka = job
    en = O / f"en_{i:03d}"
    if (en / "enroll.json").exists():
        return "skip"
    stem_a = ka[len(spk) + 1:]
    xs, tot = [], 0.0
    for w in sorted((ROOT / "female-dataset" / spk).glob("*.wav")):
        if w.stem == stem_a:
            continue
        x = load48(w)
        xs.append(x)
        tot += len(x) / 48000
        if tot >= 20.0:
            break
    if tot < 20.0:
        return f"short {spk} {tot:.1f}"
    ref = O / f"ref_{i:03d}.wav"
    sf.write(ref, np.concatenate(xs), 48000)
    r = subprocess.run([str(BIN), "artic-enroll", "-r", str(ref), "-o", str(en)], cwd=ROOT, env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    return "ok" if r.returncode == 0 else f"err {spk} {r.stderr[-300:]}"


def main() -> int:
    O.mkdir(parents=True, exist_ok=True)
    jobs = [(i, spk, ka) for i, (spk, ka, _) in enumerate(e2e_held())]
    with Pool(6) as pl:
        res = list(pl.imap(one, jobs))
    print({r.split()[0]: sum(x.split()[0] == r.split()[0] for x in res) for r in res}, [r for r in res if not r.startswith(("ok", "skip"))], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
