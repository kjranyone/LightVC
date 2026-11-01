"""A2(diag): 演奏の入力の代役 = SPARC の特徴(EMA 12・高さ・大きさ・周期性・50 Hz)を書き出す。SPARC の依存は ../sparc_env(uv)。
対象 = E1 試作のコーパス全部(data/snv_e1)・A0 の目標の登録(results/snv_a1/src/tref_*.wav)・A1 の診断の 10 s の切り出し(D0・別の女声の源・男声の源)・本物の男声の区間。
出力 = data/snv_e1_feat/sparc/<名前>.npz。
    cd .. && ../sparc_env/bin/python training/snv_a2_sparc_feats.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_e1_feat/sparc"


def jobs() -> list[tuple[str, Path]]:
    out = [(f"e1/{p.parent.name}/{p.stem}", p) for p in sorted((ROOT / "data/snv_e1").glob("e*/t*.wav"))]
    out += [(f"a1src/{p.stem}", p) for p in sorted((ROOT / "results/snv_a1/src").glob("*.wav"))]
    out += [(f"a1diag/{p.stem}", p) for p in sorted((ROOT / "results/snv_a1/diag/wav").glob("*.wav")) if p.stem.startswith(("D0_", "src_", "srcm_"))]
    return out


def main() -> int:
    from sparc import load_model
    m = load_model("multi", device="cuda")
    js = jobs()
    print("jobs", len(js), flush=True)
    for i, (name, p) in enumerate(js):
        q = OUT / f"{name}.npz"
        if q.exists():
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        c = m.encode(str(p))
        n = min(len(c["ema"]), len(c["pitch"]), len(c["loudness"]), len(c["periodicity"]))
        np.savez(q, ema=c["ema"][:n].astype(np.float32), pitch=c["pitch"][:n, 0].astype(np.float32), loudness=c["loudness"][:n, 0].astype(np.float32),
                 periodicity=c["periodicity"][:n, 0].astype(np.float32), pitch_stats=np.asarray(c["pitch_stats"], dtype=np.float32))
        if i % 500 == 0:
            print(i, name, flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
