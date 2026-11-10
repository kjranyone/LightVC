"""段 D2 の下見の amendment_1(results/snv_d2_pilot/prereg.yaml)の物差し: SPARC 多言語版の EMA を書き出す(物差しとしてだけ・../sparc_env で動かす)。
対象 = 下見の 320 本(data/snv_d2_pilot/{act,read}/<声>/sNN.wav)・段 D の内容のうち学習の声が 8 声読んだ文 30(data/snv_d/<声>/sNN.wav)・較正の伸縮の複写(下見の朗読の最初の声の s00 を速さ 0.8〜1.2 で 8 本)。
出力 = data/snv_d2_pilot/sparc/{pilot_act,pilot_read,d,cal}/....npz(ema [T][12]・50 Hz)。
    cd .. && ../sparc_env/bin/python training/snv_diag/d2_sparc_dump.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/snv_d2_pilot/sparc"


def main() -> int:
    from sparc import load_model
    m = load_model("multi", device="cuda")
    jobs = [(p, OUT / f"pilot_{p.parts[-3]}/{p.parent.name}/{p.stem}.npz") for p in sorted((ROOT / "data/snv_d2_pilot").glob("*/*/s*.wav")) if p.parts[-3] in ("act", "read")]
    dm = json.loads((ROOT / "data/snv_d/manifest.json").read_text())
    split = json.loads((ROOT / "results/snv_i3_m5/split.json").read_text())
    tr = set(split["train"])
    by: dict = {}
    for v in dm["voices"]:
        if v["voice"] not in tr:
            continue
        for f in v["files"]:
            p = ROOT / f"data/snv_d/{v['voice']}/{f['s']}.wav"
            if p.exists():
                by.setdefault(f["content"], []).append((v["voice"], p))
    cs = sorted(c for c, it in by.items() if len(it) >= 8)
    random.Random(0).shuffle(cs)
    for c in cs[:30]:
        for v, p in by[c][:8]:
            jobs.append((p, OUT / f"d/{c}/{v}.npz"))
    man = json.loads((ROOT / "data/snv_d2_pilot/manifest.json").read_text())
    src = ROOT / f"data/snv_d2_pilot/read/{man['voices'][0]['voice']}/s00.wav"
    x, sr = librosa.load(str(src), sr=16000)
    (OUT / "cal").mkdir(parents=True, exist_ok=True)
    for r in (0.8, 0.87, 0.94, 1.0, 1.06, 1.13, 1.2, 0.9):
        q = OUT / f"cal/r{r}.wav"
        sf.write(q, librosa.effects.time_stretch(x, rate=r), 16000)
        jobs.append((q, OUT / f"cal/r{r}.npz"))
    for p, q in jobs:
        if q.exists():
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        c = m.encode(str(p))
        np.savez(q, ema=np.asarray(c["ema"], dtype=np.float32))
    print("done", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
