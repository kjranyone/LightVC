"""D84(学習なし): パラレル変換の話者らしさ = 同じ文を読んだ目標 A と別の女性(偽者 3 人)の本物への DTW 距離の比べ。時刻の合わない動きの罰は全員に等しくかかる = 話者だけを見る。
d80 形式の出力(<M>_<A>_tK_<cond>.wav)を、A の tK_neutral と偽者 B1..B3 の tK_neutral(女声 TTS・d80 と同じ並びの 24 人後ろ)と比べる。
指標: 当たり(A が 4 人中で最も近い)の率・余白(偽者の平均距離 − A への距離)。基準: FLOOR(A の別の話し方)・元の男声。出力 = results/conv_p0/<tag>.json。
    uv run python d84_identity_margin.py --dir <scratchpad>/d83_345k/conv --tag d84_345k
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
FEM = ROOT / "data/female_tts_corpus"
MPAR = ROOT / "data/male_tts_v4l_par"
STYLES = ["breathy", "low_tension", "soft", "warm"]
G: dict = {}


def ana(p: Path) -> dict:
    import dtw_ruler as DR
    return DR.analyse(load48(p).astype(np.float32))


def one(job: tuple) -> list:
    import dtw_ruler as DR
    from d77_ruler import rd
    d, pair, imps, conds = job
    M, A, tk = pair.split("_")
    k = int(tk[1:])
    refs = [ana(FEM / A / f"{tk}_neutral.wav")] + [ana(FEM / B / f"{tk}_neutral.wav") for B in imps]
    items = {"SRC": MPAR / M / f"f{k:02d}.wav", "FLOOR_soft": FEM / A / f"{tk}_soft.wav"}
    items.update({c: Path(d) / f"{pair}_{c}.wav" for c in conds})
    out = []
    for c, p in items.items():
        if not p.exists():
            continue
        y = DR.analyse(rd(p) if str(p).startswith(str(d)) else load48(p).astype(np.float32))
        ds = [DR.dist(y, r)["env"] for r in refs]
        out.append({"pair": pair, "cond": c, "dA": ds[0], "dB": float(np.mean(ds[1:])), "hit": bool(ds[0] < min(ds[1:]))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conds", default="SELFCOPY,TAB,W10,C10")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    d = Path(a.dir)
    fems = sorted(x.name for x in FEM.iterdir() if x.is_dir() and all((x / f"t{i:02d}_{s}.wav").exists() for i in range(10) for s in ["neutral"] + STYLES))
    random.Random(80).shuffle(fems)
    pairs = sorted({f.name.rsplit("_", 1)[0] for f in d.glob("m*_TAB.wav")})
    jobs = []
    for pair in pairs:
        mi = int(pair.split("_")[0][1:])
        jobs.append((str(d), pair, [fems[mi + 24], fems[mi + 48], fems[mi + 72]], a.conds.split(",")))
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(one, jobs):
            rows.extend(r)
    summ = {}
    for c in ["SRC", "FLOOR_soft"] + a.conds.split(","):
        rs = [r for r in rows if r["cond"] == c]
        if rs:
            summ[c] = {"hit": round(float(np.mean([r["hit"] for r in rs])), 3), "margin": round(float(np.mean([r["dB"] - r["dA"] for r in rs])), 4),
                       "dA": round(float(np.mean([r["dA"] for r in rs])), 4), "dB": round(float(np.mean([r["dB"] for r in rs])), 4), "n": len(rs)}
            print(c, json.dumps(summ[c]), flush=True)
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
