"""D704(paper_mf E-C1d (c)・規則は results/conv_p0/d704/prereg.yaml): 2 人目の日本語の本物の男声 namikawa.mp3(リポジトリの根・コーパスの外・205 s・44.1 kHz の mp3 196 kbps・
評価専用 = 学習に使わない)から外郎売(d487_uiro)と同じ構成の関門の組 H/d704_nmk を作る: gate_mf_run.build(参照 22 s・8 s の区間・目標 = dev2 の 30 人を巡回)+ 入力 pre_<key>.wav
(操作者の参照の末尾 12 s + 元の声 8 s)+ rr_en は d487_uiro の rr_en(同じ dev2 の目標・同じ並び)への symlink。
    uv run python d704_nmkset.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48
from gate_mf_run import PRE, build

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
ROOT = Path(__file__).resolve().parent.parent
SR = 48000


def main() -> int:
    S = H / "d704_nmk"
    assert not (S / "devset.json").exists(), S
    I = H / "d704_nmk_in" / "namikawa"
    I.mkdir(parents=True, exist_ok=True)
    w = I / "namikawa_full.wav"
    if not w.exists():
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(ROOT / "namikawa.mp3"), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
        sf.write(str(w), np.frombuffer(raw, dtype="<f4"), SR, subtype="FLOAT")
    log = build(H / "d704_nmk_in", S, 30, "dev2")
    n0 = int(PRE * SR)
    for r in log:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        pre = load48(S / "d56/_work" / f"op_{r['pi']:04d}.wav")[-n0:]
        sf.write(S / "d56/_work" / f"pre_{k}.wav", np.concatenate([np.pad(pre, (n0 - len(pre), 0)), load48(S / "d56" / f"{k}_SRC.wav")]).astype(np.float32), SR)
    (S / "rr_en").symlink_to(H / "d487_uiro" / "rr_en")
    a = json.loads((H / "d487_uiro" / "devset.json").read_text())["targets"]
    b = json.loads((S / "devset.json").read_text())["targets"]
    assert a == b
    print(json.dumps({"pairs": len(log), "targets_used": sorted({r["ti"] for r in log}), "seconds": round(len(load48(w)) / SR, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
