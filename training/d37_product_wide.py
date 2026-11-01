"""D37: 製品の条件を広げる(評価プールの女声 --n_tg 人 × 2 文・日本語男声 TTS の自然な文 8s・全て Rust の製品経路)。最初の 10 目標は D29 と同じ元の文。
操作者の登録 = 元の文ごとに、同じ男声話者の neutral のうち その元の文に使っていない文 22s 以上(Rust artic-enroll)。男声の母集団 = D33 の 47 人の登録から操作者を除く平均。
  TAB = 柔らかい引き(混ぜ 0)・M35 = 混ぜ 0.35・W120 = 伸縮つき話者差 α 1.2(操作者の生の表)・W120S = α 1.2 + 操作者の表を κ 30 で縮約・W130S = α 1.3 + κ 30。
出力 = d4b の形式(cen・pNNNN_tNNN_<cond>・SRC)。
    uv run python d37_product_wide.py --d33 <scratchpad>/d33 --out_dir <scratchpad>/d37 --n_tg 30
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import diag_pool as P
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
SKIP = {"t00", "t01", "t12"}
KAPPA = 30.0


def run(cmd: list) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL {cmd} {r.stderr[-500:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d33", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_tg", type=int, default=30)
    ap.add_argument("--per", type=int, default=2)
    a = ap.parse_args()
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    work = od / "_work"; work.mkdir(exist_ok=True)
    pool = P.load_pool(150, 3, "cpu")
    males = sorted(p.name for p in (ROOT / "data/male_tts_corpus").iterdir() if p.is_dir())
    pop = {d.name[3:]: np.fromfile(d / "enroll.bin", dtype="<f4").reshape(24, -1).astype(np.float64) for d in sorted((Path(a.d33) / "_work").glob("en_male_*"))}
    assert len(pop) == len(males), len(pop)
    rng = random.Random(3)
    pi = 0
    for ti in range(a.n_tg):
        t = pool[ti]
        ws = sorted((ROOT / "female-dataset" / t["spk"]).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= 20.0:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        endir = work / f"enroll_{ti:03d}"
        if not (endir / "enroll.json").exists():
            refp = work / f"ref_{ti:03d}.wav"
            sf.write(refp, np.concatenate([load48(w) for w in ref]).astype(np.float32), 48000)
            run([BIN, "artic-enroll", "-r", refp, "-o", endir])
        sf.write(od / f"cen_{ti:03d}.wav", t["cen"], 48000)
        for j in range(a.per):
            m = males[(ti * a.per + j) % len(males)]
            mdir = ROOT / "data/male_tts_corpus" / m
            files = [f for f in sorted(mdir.glob("t*_neutral.wav")) if f.name[:3] not in SKIP]
            rng.shuffle(files)
            used, xs, tl = [], [], 0.0
            for f in files:
                if tl >= 8.0:
                    break
                x = load48(f); xs.append(x); used.append(f.name); tl += len(x) / 48000
            srcp = od / f"p{pi:04d}_t{ti:03d}_SRC.wav"
            sf.write(srcp, np.concatenate(xs)[: 8 * 48000].astype(np.float32), 48000)
            opd = work / f"op_{pi:04d}"
            if not (opd / "enroll.json").exists():
                rest = [f for f in sorted(mdir.glob("t*_neutral.wav")) if f.name not in used]
                ys, tot = [], 0.0
                for f in rest:
                    if tot >= 22.0:
                        break
                    y = load48(f); ys.append(y); tot += len(y) / 48000
                opw = work / f"op_{pi:04d}.wav"
                sf.write(opw, np.concatenate(ys).astype(np.float32), 48000)
                run([BIN, "artic-enroll", "-r", opw, "-o", opd])
            j_ = json.loads((opd / "enroll.json").read_text())
            top = np.fromfile(opd / "enroll.bin", dtype="<f4").reshape(24, -1).astype(np.float64)
            n = np.asarray(j_["counts"], np.float64)[None]
            popm = np.mean([v for k, v in pop.items() if k != m], 0)
            shd = work / f"opS_{pi:04d}"
            shd.mkdir(exist_ok=True)
            ((n * top + KAPPA * popm) / (n + KAPPA)).astype("<f4").tofile(shd / "enroll.bin")
            (shd / "enroll.json").write_text(json.dumps({"D": 24, "K": top.shape[1], "mu": j_["mu"], "per_mean": j_["per_mean"], "counts": []}))
            conds = {"TAB": ["--mix", "0"], "M35": ["--mix", "0.35"],
                     "W120": ["--mix", "0", "--operator-enroll", opd, "--warp", "--alpha", "1.2"],
                     "W120S": ["--mix", "0", "--operator-enroll", shd, "--warp", "--alpha", "1.2"],
                     "W130S": ["--mix", "0", "--operator-enroll", shd, "--warp", "--alpha", "1.3"]}
            for name, args in conds.items():
                run([BIN, "artic", "-i", srcp, "-o", od / f"p{pi:04d}_t{ti:03d}_{name}.wav", "--enroll", endir] + args)
            pi += 1
        print("target", ti, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
