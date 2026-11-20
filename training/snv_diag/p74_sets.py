"""段 P74 step a の測り直し(results/snv_p74/prereg.yaml の step_a_remeasure・学習なし): 脆い源の組の声を足す。
d_rv6b = E1 の検証の男声 6 + 学習の男声の先頭 48(段 P73 の d_rv6 の 18 声を含む)× t05・t06・登録 t00〜t02 に、合成の残響(RT60 0.6 s・DRR 0 dB・p73_deg.py と同じ式と種)を当てた版。
段 P73 の 18 声は data/snv_p73/deg/rv6 の音をそのまま指す。足す 36 声は data/snv_p74/deg/rv6/<声>/<発話>.wav。台帳は data/snv_p73/sets.json の d_rv6b(probe_rig.ext_items が読む)。
    cd training && uv run --no-sync python snv_diag/p74_sets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p73_deg as DG
import snv_d5_chain as CH

ROOT = DG.ROOT


def main() -> int:
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    voices = [v for v in split["val"] if v.startswith("em")] + [v for v in split["train"] if v.startswith("em")][:48]
    old = {x["voice"] for x in json.loads((DG.D / "sets.json").read_text())["d_rv6"]}
    rel = lambda v, t: f"data/snv_p73/deg/rv6/{v}/{t}.wav" if v in old else f"data/snv_p74/deg/rv6/{v}/{t}.wav"
    for v in voices:
        for t in DG.TS:
            q = ROOT / rel(v, t)
            if q.exists():
                continue
            q.parent.mkdir(parents=True, exist_ok=True)
            x, sr = sf.read(ROOT / f"data/snv_e1/{v}/{t}.wav", dtype="float32")
            assert sr == CH.SR
            sf.write(q, DG.reverb(x, 0.6, 0.0, CH.seed_of(f"p73rir|{v}")), sr)
    w = lambda v, t: {"wav": rel(v, t), "ema": f"{DG.SP}/e1/{v}/{t}.npz"}
    sets = json.loads((DG.D / "sets.json").read_text())
    sets["d_rv6b"] = [{"voice": v, "src": [w(v, "t05"), w(v, "t06")], "enr": [w(v, t) for t in ("t00", "t01", "t02")]} for v in voices]
    (DG.D / "sets.json").write_text(json.dumps(sets, ensure_ascii=False, indent=1))
    print(len(voices), sum(v in old for v in voices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
