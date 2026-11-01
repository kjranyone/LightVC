"""D82(学習なし): D80 の変換(製品経路・TTS の目標 A)を、話者基準の自然さの物差し(D78: 動きの共分散の Förstner 距離 cov・帯ごとの RMS rms・AUC 0.939)で測る。
基準 = A の文 tK 以外の neutral 9 文。比べる = A の tK_neutral(本物)・別の話し方(本物)・SELFCOPY・各変換。出力 = results/conv_p0/d82.json。
注意: 色づけ(C*)は条件の共分散を目標に合わせる = cov は構成上下がりうる(単独で合否に使わない)。
    uv run python d82_conv_natural.py --d80 <scratchpad>/d80 --workers 10
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48
from d77_ruler import rd
from d78_spk_ruler import dist, feats, init, ref_feats

ROOT = Path(__file__).resolve().parent.parent
FEM = ROOT / "data/female_tts_corpus"


def one(job: tuple) -> list:
    d80, pair, items = job
    M, A, tk = pair.split("_")
    k = int(tk[1:])
    r = ref_feats(np.concatenate([load48(FEM / A / f"t{i:02d}_neutral.wav") for i in range(10) if i != k]).astype(np.float32))
    out = []
    for c, p in items:
        x = load48(p).astype(np.float32) if "female_tts_corpus" in p else rd(Path(p))
        out.append({"pair": pair, "cond": c, **dist(feats(x), r)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d80", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--conds", default="SELFCOPY,TAB,W07,W10,C10,C07")
    a = ap.parse_args()
    d = Path(a.d80)
    pairs = sorted({f.name.rsplit("_", 1)[0] for f in d.glob("m*_TAB.wav")})
    jobs = []
    for pair in pairs:
        M, A, tk = pair.split("_")
        items = [("REAL", str(FEM / A / f"{tk}_neutral.wav"))] + [(f"STYLE_{s}", str(FEM / A / f"{tk}_{s}.wav")) for s in ["soft", "warm"]]
        items += [(c, str(d / f"{pair}_{c}.wav")) for c in a.conds.split(",") if (d / f"{pair}_{c}.wav").exists()]
        jobs.append((str(d), pair, items))
    rows = []
    with Pool(a.workers, initializer=init) as pool:
        for r in pool.imap_unordered(one, jobs):
            rows.extend(r)
    conds = ["REAL", "STYLE_soft", "STYLE_warm"] + a.conds.split(",")
    summ = {m: {c: round(float(np.nanmean([r[m] for r in rows if r["cond"] == c])), 4) for c in conds} for m in ["cov", "rms", "ctr", "mod"]}
    for m, v in summ.items():
        print(m, json.dumps(v), flush=True)
    (ROOT / "results/conv_p0/d82.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
