"""交差の男声の操作者(MalePool の 77 人)の登録(Rust artic-enroll)。参照 = 話者の書き出し済みの 48 kHz のクリップを名前順につないで 20 s 以上(足りなければある分)。
動きの入力(表 + γ·(操作者の包絡 − Σp T_操作者))を交差の標本で学習するための操作者の表。出力 = <dir>/male_en/<spk>/enroll.json・enroll.bin。
    uv run python d291_maleenroll.py --dirs <dev の親>/xmale <dev の親>/condgan
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d273_data import BIN

ROOT = Path(__file__).resolve().parent.parent


def one(job: tuple) -> str:
    spk, wavs, out = job
    en = out / spk
    if (en / "enroll.json").exists():
        return "skip"
    xs, tot = [], 0.0
    for w in wavs:
        x, sr = sf.read(str(w), dtype="float32")
        assert sr == 48000, (w, sr)
        xs.append(x if x.ndim == 1 else x.mean(1))
        tot += len(xs[-1]) / sr
        if tot >= 20.0:
            break
    ref = out / f"{spk}_ref.wav"
    sf.write(ref, np.concatenate(xs), 48000)
    r = subprocess.run([str(BIN), "artic-enroll", "-r", str(ref), "-o", str(en)], cwd=ROOT, env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    return f"ok {tot:.1f}" if r.returncode == 0 else f"err {spk} {r.stderr[-200:]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    a = ap.parse_args()
    jobs = []
    for d in map(Path, a.dirs):
        out = d / "male_en"
        out.mkdir(exist_ok=True)
        by: dict = {}
        for m in json.loads((d / "meta.json").read_text())["male"]:
            if m["key"].startswith("m0") or m["spk"].startswith("m0"):
                continue
            w = d / "wav" / f"{m['key']}.wav"
            if w.exists():
                by.setdefault(m["spk"], []).append(w)
        jobs += [(s, sorted(v), out) for s, v in sorted(by.items())]
    with Pool(6) as pl:
        res = list(pl.imap(one, jobs))
    print(len(jobs), {r.split()[0]: sum(x.split()[0] == r.split()[0] for x in res) for r in res}, [r for r in res if r.startswith("err")][:3], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
