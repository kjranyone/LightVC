"""diag_snv_a1x の FF の物差しに要る A0F の j3〜j5 の SPARC の特徴(物差しとしてだけ・../sparc_env で動かす)。
出力 = data/snv_e1_feat/sparc/a0/<声>_<j>.npz(ema・pitch・loudness・periodicity)。
    cd .. && ../sparc_env/bin/python training/snv_diag/a0_sparc_more.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from sparc import load_model
    m = load_model("multi", device="cuda")
    n = 0
    for p in sorted((ROOT / "results/snv_a0/wav").glob("*_[345].wav")):
        q = ROOT / f"data/snv_e1_feat/sparc/a0/{p.stem}.npz"
        if q.exists():
            continue
        c = m.encode(str(p))
        k = min(len(c["ema"]), len(c["pitch"]), len(c["loudness"]), len(c["periodicity"]))
        np.savez(q, ema=c["ema"][:k].astype(np.float32), pitch=c["pitch"][:k, 0].astype(np.float32), loudness=c["loudness"][:k, 0].astype(np.float32), periodicity=c["periodicity"][:k, 0].astype(np.float32))
        n += 1
    print("wrote", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
