"""N1(diag)の下ごしらえ: N0r の男 12 声の発声(操作者の入力)と男の登録(A0 の j0〜j2 の連結)を、A2 と同じ BigVGAN の log-mel にし、SPARC 用の 44.1 kHz の wav を書き出す。
出力 = data/snv_e1_feat/mel/n0r/{voice}_{cat}_{k}.npy・mel/a0enroll/{pm}.npy・wav44/n0r__*.wav・wav44/a0enroll__{pm}.wav。
    cd training && uv run --no-sync python snv_n1_prep.py && cd .. && ../sparc_env/bin/python training/snv_n1_sparc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from snv_a2_mel import A0, OUT, SR, logmel

N0R = Path(__file__).resolve().parent.parent / "results/snv_n0r/wav"


def main() -> int:
    its = [it for it in json.loads((N0R / "items.json").read_text()) if it["sex"] == "M"]
    w44 = OUT.parent / "wav44"
    for it in its:
        name = f"{it['voice']}_{it['cat']}_{it['k']}"
        y = librosa.load(str(N0R / f"{name}.wav"), sr=SR, mono=True)[0]
        (OUT / "n0r").mkdir(parents=True, exist_ok=True)
        np.save(OUT / f"n0r/{name}.npy", logmel(y).astype(np.float16))
        sf.write(w44 / f"n0r__{name}.wav", y, SR)
    for v in sorted({it["voice"] for it in its}):
        y = np.concatenate([librosa.load(str(A0 / f"wav/{v}_{j}.wav"), sr=SR, mono=True)[0] for j in (0, 1, 2)])
        np.save(OUT / f"a0enroll/{v}.npy", logmel(y).astype(np.float16))
        sf.write(w44 / f"a0enroll__{v}.wav", y, SR)
    print("inputs", len(its), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
