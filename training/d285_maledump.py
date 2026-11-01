"""diag_ppgvoc3(交差の標本)の男声の操作者の入力: data/male_tts_corpus(Irodori v3・VCTK の男声の日本語クローン 47 人・評価の v4 Large の男声 m000〜m047 とは別人)を
48 kHz にして Rust の前段(d273_data の SELF: --causal-ct・高さの移動なし)で条件と C1 の事後を書き出す(描画は省く)。devtts の男声の操作者と同じ話者 = devtts の MT は学習済みの話者になる。
    uv run python d285_maledump.py --out <dev の親>/xmale --per_spk 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d273_data import BIN, SELF, job

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_spk", type=int, default=8)
    a = ap.parse_args()
    from d6_unit_cov import load48
    O = Path(a.out)
    (O / "wav").mkdir(parents=True, exist_ok=True)
    (O / "male").mkdir(parents=True, exist_ok=True)
    rng = random.Random(285)
    jobs, meta = [], []
    for sd in sorted(d for d in (ROOT / "data/male_tts_corpus").iterdir() if d.is_dir()):
        ws = sorted(sd.glob("*.wav"))
        for w in rng.sample(ws, min(a.per_spk, len(ws))):
            k = f"{sd.name}_{w.stem}"
            p48 = O / "wav" / f"{k}.wav"
            if not p48.exists():
                sf.write(p48, load48(w), 48000)
            jobs.append(([BIN, "artic", "-i", p48] + SELF, O / "male" / f"{k}.cond", O / "male" / f"{k}.front"))
            meta.append({"key": k, "spk": sd.name})
    with Pool(6) as pl:
        list(pl.imap_unordered(job, jobs))
    ok = [m for m in meta if (O / "male" / f"{m['key']}.cond").exists()]
    (O / "meta.json").write_text(json.dumps({"male": ok}, indent=1))
    print("dumped", len(ok), "/", len(jobs), "speakers", len({m["spk"] for m in ok}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
