"""A1 の診断の読み(合否は変えない・記録だけ): D0〜D3 の目標の top-1(ECAPA + LDA・24 声)と、D2・D3 で出力が源の声に読まれる割合・動き m。
    cd training && uv run --no-sync python snv_a1_diag_measure.py
"""
from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from snv_a0_measure import ROOT, Embedders, load, to16
from snv_a0b_measure import fit_lda, proj, splits

OUT = ROOT / "results/snv_a1/diag"
A0 = ROOT / "results/snv_a0"


def main() -> int:
    rows = json.loads((OUT / "meta.json").read_text())
    sp = splits()
    pool = sp["pool"]
    emb = Embedders()
    X, y = [], []
    for i, s in enumerate(sp["lda"]):
        for p in pool[s]:
            X.append(emb(to16(*load(p)))["ecapa"])
            y.append(i)
    res = {"note": "診断(記録だけ)。results/snv_a1/verdict.yaml の合否は変えない"}
    for name, T in (("ecapa+lda", fit_lda(np.stack(X), np.array(y))), ("ecapa", None)):
        voices = sorted({r["target"] for r in rows}) + [f"pm{k:03d}" for k in range(12)]
        E = {v: proj(emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)]))["ecapa"], T) for v in voices}
        fv = sorted({r["target"] for r in rows})
        out = {}
        for d in ("D0", "D1", "D1b", "D2", "D3"):
            top, as_src, ms = [], [], []
            for r in rows:
                e = proj(emb(sf.read(str(OUT / f"wav/{d}_{r['target']}_{r['g']}.wav"))[0].astype(np.float32))["ecapa"], T)
                sc = {u: float(e @ E[u]) for u in fv}
                best = max(sc, key=sc.get)
                top.append(best == r["target"])
                if d == "D2":
                    as_src.append(best == r["d2_src"])
                    ms.append(sc[r["target"]] - sc[r["d2_src"]])
                if d == "D3":
                    ms.append(float(e @ E[r["target"]] - e @ E[r["d3_src"]]))
            out[d] = {"top1": float(np.mean(top)), "n": len(top)}
            if as_src:
                out[d]["read_as_source"] = float(np.mean(as_src))
            if ms:
                out[d]["m_pos_frac"] = float(np.mean(np.array(ms) > 0))
                out[d]["m_mean"] = float(np.mean(ms))
        res[name] = out
        print(name, json.dumps(out), flush=True)
    (OUT / "diag_measure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
