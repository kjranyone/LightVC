"""D481: 交差の標本の男声の束に日本語の男声を足す材料(diag_ppgvoc46 用)。data/pair_tts_v4l の男声 pm000〜pm039(学習の操作者の側として作った Irodori-TTS-v4-Large・評価の操作者 m000〜m047 とは別の声と文)の
16 文を MalePool の形(<out>/meta.json の male・<out>/wav・<out>/male/*.cond / *.front = Rust の書き出し d273_data と同じ設定)に。登録は d291_maleenroll.py --dirs <out> で作る。
    uv run python d481_xmale_ja.py --out <dev の親>/xmale_ja
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d273_data import BIN, SELF, job
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    O = Path(a.out)
    for d in ("wav", "male"):
        (O / d).mkdir(parents=True, exist_ok=True)
    meta, jobs = {"male": []}, []
    for s in sorted((ROOT / "data/pair_tts_v4l").glob("pm*")):
        for w in sorted(s.glob("t*.wav")):
            k = f"{s.name}_{w.stem}"
            wp = O / "wav" / f"{k}.wav"
            if not wp.exists():
                sf.write(wp, load48(w).astype(np.float32), 48000)
            jobs.append(([BIN, "artic"] + SELF + ["-i", wp], O / "male" / f"{k}.cond", O / "male" / f"{k}.front"))
            meta["male"].append({"key": k, "spk": s.name, "src": "tts_ja"})
    with Pool(3) as pl:
        list(pl.imap_unordered(job, jobs))
    ok = sum((O / "male" / f"{m['key']}.cond").exists() for m in meta["male"])
    (O / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print("clips", len(meta["male"]), "dumped", ok, "speakers", len({m["spk"] for m in meta["male"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
