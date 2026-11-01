"""A1 の診断(合否は変えない・記録だけ): 目標の top-1 が天井の 3 分の 1 に留まる在処を SPARC の経路ごとに分ける。全て 10 s の切り出し。
D0 = 目標本人の元の音声・D1 = その SPARC の再合成(演奏も構造も本人)・D1b = 本人の演奏 + 参照から取った構造・D2 = 別の女声の演奏 → 目標・D3 = Irodori の男声の演奏 → 目標。
    cd .. && ../sparc_env/bin/python training/snv_a1_diag_convert.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/snv_a1/diag"
A0 = ROOT / "results/snv_a0"
A1 = ROOT / "results/snv_a1"
SR = 16000
GROUPS = ((3, 4, 5), (8, 9, 10), (11, 12, 13))


def clip10(voice: str, g: tuple) -> np.ndarray:
    y = np.concatenate([librosa.load(str(A0 / f"wav/{voice}_{j}.wav"), sr=SR, mono=True)[0] for j in g])
    return y[: 10 * SR].astype(np.float32)


def main() -> int:
    from sparc import load_model
    m = load_model("multi", device="cuda")
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    meta = json.loads((A1 / "convert_meta.json").read_text())
    fv = meta["targets"]
    mv = [f"pm{k:03d}" for k in range(12)]
    tcode = {v: m.encode(str(A1 / f"src/tref_{v}.wav")) for v in fv}
    rows = []
    for i, v in enumerate(fv):
        for gi, g in enumerate(GROUPS):
            p = OUT / f"wav/D0_{v}_{gi}.wav"
            sf.write(p, clip10(v, g), SR)
            c = m.encode(str(p))
            sf.write(OUT / f"wav/D1_{v}_{gi}.wav", m.decode(**copy.deepcopy(c)), SR)
            sf.write(OUT / f"wav/D1b_{v}_{gi}.wav", m.convert(src_code=c, trg_code=tcode[v]), SR)
            other = fv[(i + 1 + gi) % len(fv)]
            q = OUT / f"wav/src_{other}_{gi}.wav"
            if not q.exists():
                sf.write(q, clip10(other, g), SR)
            sf.write(OUT / f"wav/D2_{v}_{gi}.wav", m.convert(src_code=m.encode(str(q)), trg_code=tcode[v]), SR)
            male = mv[(i + gi) % len(mv)]
            r = OUT / f"wav/srcm_{male}.wav"
            if not r.exists():
                sf.write(r, clip10(male, (3, 4, 5)), SR)
            sf.write(OUT / f"wav/D3_{v}_{gi}.wav", m.convert(src_code=m.encode(str(r)), trg_code=tcode[v]), SR)
            rows.append({"target": v, "g": gi, "d2_src": other, "d3_src": male})
        print("diag", v, flush=True)
    (OUT / "meta.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
