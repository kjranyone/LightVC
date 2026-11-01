"""D31: 製品の条件(D29 と同じ組・同じ元の文・同じ目標の登録)で、操作者の登録(単位ごとの話者差を元の声の包絡に足す = D30 の DELTA)を Rust の製品経路で測る。
操作者の参照 = 同じ男声話者の neutral のうち元の文に使っていない文(五十音などの読み上げも含む = 登録用の台本の想定)を 22s 以上。
出力 = d4b の形式: D29 の cen・SRC・TAB・M35 を写し、DLT(--operator・混ぜ 0)と DLT35(混ぜ 0.35 = 元の包絡 + 0.65·話者差)を加える。
    uv run python d31_operator.py --d29 <scratchpad>/d29 --out_dir <scratchpad>/d31 --n_tg 10
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
SKIP = {"t00", "t01", "t12"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d29", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_tg", type=int, default=10)
    ap.add_argument("--per", type=int, default=2)
    a = ap.parse_args()
    d29 = Path(a.d29)
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    work = od / "_work"; work.mkdir(exist_ok=True)
    males = sorted(p.name for p in (ROOT / "data/male_tts_corpus").iterdir() if p.is_dir())
    rng = random.Random(3)
    pi = 0
    for ti in range(a.n_tg):
        shutil.copy(d29 / f"cen_{ti:03d}.wav", od / f"cen_{ti:03d}.wav")
        endir = d29 / "_work" / f"enroll_{ti:03d}"
        assert (endir / "enroll.json").exists(), endir
        for j in range(a.per):
            m = males[(ti * a.per + j) % len(males)]
            mdir = ROOT / "data/male_tts_corpus" / m
            files = [f for f in sorted(mdir.glob("t*_neutral.wav")) if f.name[:3] not in SKIP]
            rng.shuffle(files)
            used, tl = [], 0.0
            for f in files:
                if tl >= 8.0:
                    break
                x = load48(f); used.append(f.name); tl += len(x) / 48000
            src = np.concatenate([load48(mdir / f) for f in used])[: 8 * 48000].astype(np.float32)
            srcp = d29 / f"p{pi:04d}_t{ti:03d}_SRC.wav"
            s29, _ = sf.read(srcp, dtype="float32")
            assert len(s29) == len(src) and np.abs(s29 - src).max() < 1e-3, f"D29 の元の文の再現が合わない {srcp}"
            opd = work / f"op_{m}"
            if not (opd / "enroll.json").exists():
                rest = [f for f in sorted(mdir.glob("t*_neutral.wav")) if f.name not in used]
                xs, tot = [], 0.0
                for f in rest:
                    if tot >= 22.0:
                        break
                    x = load48(f); xs.append(x); tot += len(x) / 48000
                assert tot >= 20.0, f"{m} の操作者の参照が足りない {tot:.1f}s"
                opw = work / f"op_{m}.wav"
                sf.write(opw, np.concatenate(xs).astype(np.float32), 48000)
                op_args = ["--operator", str(opw), "--save-operator", str(opd)]
            else:
                op_args = ["--operator-enroll", str(opd)]
            for c in ("SRC", "TAB", "M35"):
                shutil.copy(d29 / f"p{pi:04d}_t{ti:03d}_{c}.wav", od / f"p{pi:04d}_t{ti:03d}_{c}.wav")
            for name, mix in (("DLT", 0.0), ("DLT35", 0.35)):
                outp = od / f"p{pi:04d}_t{ti:03d}_{name}.wav"
                cmd = [str(BIN), "artic", "-i", str(srcp), "-o", str(outp), "--mix", str(mix), "--enroll", str(endir)] + op_args
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                if r.returncode != 0:
                    print("FAIL", cmd, r.stderr[-500:], flush=True)
                    return 1
                op_args = ["--operator-enroll", str(opd)]
            pi += 1
        print("target", ti, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
