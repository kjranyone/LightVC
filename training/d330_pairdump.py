"""diag_ppgvoc19(同じ文の男声 / 女声の操作者の変換をそろえる損失)の入力: data/pair_tts_v4l(pmNNN・pfNNN が同じ 16 文を読む)を 48 kHz にして
Rust の前段(d273_data の SELF: --causal-ct・高さの移動なし)で条件と C1 の事後を書き出す(描画は省く)。
出力 = <out>/pair/<spk>_tNN.{cond,front}・<out>/meta.json({"pair": [{key, spk, g(m/f), text}]})。
    uv run python d330_pairdump.py --out <dev の親>/xpair
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d273_data import SELF, job

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=str(ROOT / "data/pair_tts_v4l"))
    a = ap.parse_args()
    from d6_unit_cov import load48
    O, S = Path(a.out), Path(a.src)
    (O / "wav").mkdir(parents=True, exist_ok=True)
    (O / "pair").mkdir(parents=True, exist_ok=True)
    jobs, meta = [], []
    for sd in sorted(d for d in S.iterdir() if d.is_dir() and d.name[:2] in ("pm", "pf")):
        for w in sorted(sd.glob("t*.wav")):
            k = f"{sd.name}_{w.stem}"
            p48 = O / "wav" / f"{k}.wav"
            if not p48.exists():
                sf.write(p48, load48(w), 48000)
            jobs.append(([BIN, "artic", "-i", p48] + SELF, O / "pair" / f"{k}.cond", O / "pair" / f"{k}.front"))
            meta.append({"key": k, "spk": sd.name, "g": sd.name[1], "text": w.stem})
    with Pool(6) as pl:
        list(pl.imap_unordered(job, jobs))
    ok = [m for m in meta if (O / "pair" / f"{m['key']}.cond").exists()]
    (O / "meta.json").write_text(json.dumps({"pair": ok}, indent=1))
    print("dumped", len(ok), "/", len(jobs), "m", sum(m["g"] == "m" for m in ok), "f", sum(m["g"] == "f" for m in ok), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
