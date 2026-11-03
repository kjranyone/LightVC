"""N1(diag): SPARC の特徴(A2 の演奏の代役)を N0r の男の発声と男の登録に書き出す。依存は ../sparc_env(uv)。
入力 = data/snv_e1_feat/wav44/n0r__*.wav・a0enroll__pm*.wav(snv_n1_prep.py)。出力 = data/snv_e1_feat/sparc/n0r/<名前>.npz・sparc/a0enroll/<pm>.npz。
    cd .. && ../sparc_env/bin/python training/snv_n1_sparc.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/snv_e1_feat"


def main() -> int:
    from sparc import load_model
    m = load_model("multi", device="cuda")
    js = [(f"n0r/{p.stem[5:]}", p) for p in sorted((FEAT / "wav44").glob("n0r__*.wav"))]
    js += [(f"a0enroll/{p.stem[10:]}", p) for p in sorted((FEAT / "wav44").glob("a0enroll__pm*.wav"))]
    for name, p in js:
        q = FEAT / f"sparc/{name}.npz"
        if q.exists():
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        c = m.encode(str(p))
        n = min(len(c["ema"]), len(c["pitch"]), len(c["loudness"]), len(c["periodicity"]))
        np.savez(q, ema=c["ema"][:n].astype(np.float32), pitch=c["pitch"][:n, 0].astype(np.float32), loudness=c["loudness"][:n, 0].astype(np.float32),
                 periodicity=c["periodicity"][:n, 0].astype(np.float32), pitch_stats=np.asarray(c["pitch_stats"], dtype=np.float32))
    print("done", len(js), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
