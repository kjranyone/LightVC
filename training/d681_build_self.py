"""D681: 自己変換の組 d681_self を dev2 の目標から作る(操作者 = 目標本人の登録の録音 ref_TTT・入力 = 本人の別録音 cen_TTT の音量が最大の 8 s・前置き = ref の末尾 12 s)。
出力 = <H>/d681_self/{d37/_work/ref_TTT.wav, d56/{key}_SRC.wav, d56/_work/{op_PPPP, pre_key}.wav, d85_work/{en_TTT, op_PPPP}, devset.json}・key = pTTTT_tTTT。
    uv run python d681_build_self.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from d566_mf_v3_ear import best8, load

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
D, S = H / "dev2", H / "d681_self"
SR = 48000


def main() -> int:
    for d in ("d37/_work", "d56/_work", "d85_work"):
        (S / d).mkdir(parents=True, exist_ok=True)
    pairs = []
    for t in range(30):
        ref, cen = D / "d37/_work" / f"ref_{t:03d}.wav", D / "d37" / f"cen_{t:03d}.wav"
        key = f"p{t:04d}_t{t:03d}"
        shutil.copy(ref, S / "d37/_work" / ref.name)
        if not (S / "d85_work" / f"en_{t:03d}").exists():
            shutil.copytree(D / "d85_work" / f"en_{t:03d}", S / "d85_work" / f"en_{t:03d}")
        shutil.copy(ref, S / "d56/_work" / f"op_{t:04d}.wav")
        opd = S / "d85_work" / f"op_{t:04d}"
        if not (opd / "enroll.bin").exists():
            r = subprocess.run([str(ROOT / "target/release/lightvc-app"), "artic-enroll", "-r", str(ref), "-o", str(opd)], cwd=ROOT, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
            assert r.returncode == 0, r.stderr[-300:]
        x = best8(load(cen))
        sf.write(S / "d56" / f"{key}_SRC.wav", x.astype(np.float32), SR)
        pre = load(ref)[-12 * SR:]
        sf.write(S / "d56/_work" / f"pre_{key}.wav", np.concatenate([np.pad(pre, (12 * SR - len(pre), 0)), x]).astype(np.float32), SR)
        pairs.append({"pi": t, "ti": t, "kind": "SELF"})
    (S / "devset.json").write_text(json.dumps({"targets": json.loads((D / "devset.json").read_text())["targets"], "pairs": pairs, "kind": "SELF"}))
    print("self pairs", len(pairs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
