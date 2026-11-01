"""D534(D533 の続き・推論だけ): 息の端のピッチ(F1 のみの有声)を因果に減らす F1 の有声の判定の設定を、教師の有声(f0hi.teacher_f0・非因果)との一致で選ぶ。
設定ごとに devff5 の定常状態の条件を書き出し(d467_steady と同じ前置き・COM + 設定のフラグ)、元の 8 s のフレームで 偽の有声(条件が有声・教師が無声)と取りこぼし(教師が有声・条件が無声)の率、
有声の切り替わりの回数(毎秒)を出す。時刻のずれは D532 の TF0_info.json の発話ごとの lag。
    uv run python d534_voicing_sweep.py --cfgs BASE= H73=--voicing-hyst+0.7,0.3 C03=--f0-conf+0.3 --out ../results/conv_p0/d534
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import f0hi as FH
import train_ppgvoc as P
from d273_data import BIN, job
from d274_evaldump import COM
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent


def teacher_v(k: str, S: Path, n: int, lag: int) -> np.ndarray:
    ft, _ = FH.teacher_f0(load48(S / "d467" / f"{k}.wav").astype(np.float64), n=n + 8)
    fs = np.zeros(n, np.float32)
    lo, hi = max(0, -lag), min(n, len(ft) - lag)
    fs[lo:hi] = ft[lo + lag:hi + lag]
    return fs > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", nargs="+", required=True)
    ap.add_argument("--set", default="devff5")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = P.H / a.set
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    lags = json.loads((ROOT / "results/conv_p0/d532/TF0_info.json").read_text())
    pairs = [r for r in json.loads((S / "devset.json").read_text())["pairs"] if f"p{r['pi']:04d}_t{r['ti']:03d}" in lags]
    cfgs = {c.split("=")[0]: [x for x in c.split("=", 1)[1].split("+") if x] for c in a.cfgs}
    tv_cache = O / f"teacher_v_{a.set}.npz"
    tv = dict(np.load(tv_cache)) if tv_cache.exists() else {}
    rep = {}
    for nm, fl in cfgs.items():
        ed = P.H / "e2e" / (f"eval_{a.set}_pre" if nm == "BASE" else f"eval_{a.set}_pre_{nm}")
        ed.mkdir(parents=True, exist_ok=True)
        jobs = []
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            jobs.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{r['ti']:03d}", "--operator-enroll", S / "d85_work" / f"op_{r['pi']:04d}", "--med-from-operator"] + COM + fl + ["-i", S / "d467" / f"{k}.wav"], ed / f"{k}.cond", ed / f"{k}.front"))
        with Pool(3) as pl:
            list(pl.imap_unordered(job, jobs))
        fv, ms, sw = [], [], []
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            c, _ = P.load_dump(ed, k)
            if k not in tv:
                tv[k] = teacher_v(k, S, len(c), lags[k]["lag"])
            t_ = tv[k][2400:4000]
            v = c[2400:4000, 26] > 0.5
            n = min(len(v), len(t_))
            v, t_ = v[:n], t_[:n]
            fv.append(float((v & ~t_).sum() / max(1, (~t_).sum())))
            ms.append(float((~v & t_).sum() / max(1, t_.sum())))
            sw.append(float(np.abs(np.diff(v.astype(int))).sum() / (n / 200)))
        rep[nm] = {"false_voiced": round(float(np.median(fv)), 4), "miss": round(float(np.median(ms)), 4), "switch_per_s": round(float(np.median(sw)), 2), "n": len(fv)}
        print(nm, rep[nm], flush=True)
        np.savez(tv_cache, **tv)
    tsw = [float(np.abs(np.diff(v[2400:4000].astype(int))).sum() / 8) for v in tv.values()]
    rep["TEACHER_switch_per_s"] = round(float(np.median(tsw)), 2)
    print("teacher switch/s", rep["TEACHER_switch_per_s"], flush=True)
    (O / "sweep.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
