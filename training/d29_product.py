"""D29: 製品の条件(日本語男声 TTS の自然な文 → 評価プールの女声)を Rust の製品経路(lightvc-app artic・参照 WAV からの登録・柔らかい引き β 1・平滑なし)で変換し、
つまみ mix = 0 / 0.2 / 0.35 の明瞭さ(CER)と同一性(LDA)を測る素材を作る。出力 = d4b の形式(cen_TTT・pNNNN_tTTT_{TAB,M20,M35}・SRC)。
男声の文は五十音などの読み上げ(t00・t01・t12)を除く。
    uv run python d29_product.py --out_dir <scratchpad>/d29 --n_tg 10
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_tg", type=int, default=10)
    ap.add_argument("--per", type=int, default=2)
    a = ap.parse_args()
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    work = od / "_work"; work.mkdir(exist_ok=True)
    pool = P.load_pool(150, 3, "cpu")
    males = sorted(p.name for p in (ROOT / "data/male_tts_corpus").iterdir() if p.is_dir())
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
        xr = np.concatenate([load48(w) for w in ref]).astype(np.float32)
        refp = work / f"ref_{ti:03d}.wav"
        sf.write(refp, xr, 48000)
        sf.write(od / f"cen_{ti:03d}.wav", t["cen"], 48000)
        endir = work / f"enroll_{ti:03d}"
        for j in range(a.per):
            m = males[(ti * a.per + j) % len(males)]
            files = [f for f in sorted((ROOT / "data/male_tts_corpus" / m).glob("t*_neutral.wav")) if f.name[:3] not in SKIP]
            rng.shuffle(files)
            xs, tl = [], 0.0
            for f in files:
                if tl >= 8.0:
                    break
                x = load48(f); xs.append(x); tl += len(x) / 48000
            src = np.concatenate(xs)[: 8 * 48000].astype(np.float32)
            srcp = od / f"p{pi:04d}_t{ti:03d}_SRC.wav"
            sf.write(srcp, src, 48000)
            for name, mix in (("TAB", 0.0), ("M20", 0.2), ("M35", 0.35)):
                outp = od / f"p{pi:04d}_t{ti:03d}_{name}.wav"
                cmd = [str(BIN), "artic", "-i", str(srcp), "-o", str(outp), "--mix", str(mix)]
                cmd += ["--enroll", str(endir)] if (endir / "enroll.json").exists() else ["--reference", str(refp), "--save-enroll", str(endir)]
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                if r.returncode != 0:
                    print("FAIL", cmd, r.stderr[-500:], flush=True)
                    return 1
            pi += 1
        print("target", ti, t["spk"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
