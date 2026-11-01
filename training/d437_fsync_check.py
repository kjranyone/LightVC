"""D437: フレーム同期の型の統計(fsync.py)が amline(d109 のガビ軸・耳の「ガビガビ」)と同じ順序を作るか(学習の損失に使う前の物差しの確認)。
devff の入力そのもの(実女声)・製品・kNN-VC・B / D / D2 / 作り直し 40(15k)の女声 → 女声の書き出しで、各発話の先頭 8 s の fsync と amline を並べる。
    uv run python d437_fsync_check.py --out ../results/conv_p0/d437.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    from scipy.stats import spearmanr
    from d6_unit_cov import load48
    from eval_nvoc import amline
    from fsync import fline, fsync
    sets = {"SRC": sorted((H / "devff/d56").glob("p*_SRC.wav"))[::3], "PROD": sorted((H / "devff/d114").glob("p*_G03S10ZH.wav"))[::3],
            "KNN": sorted((H / "devff/d135_real").glob("p*_KNN.wav"))[::3]}
    for r, nm in (("27", "B"), ("36", "D"), ("41", "D2"), ("40", "R40")):
        ev = sorted((ROOT / f"results/diag_ppgvoc{r}").glob("ev_*"))[-1]
        sets[nm] = sorted(ev.glob("f*_FF.wav"))
    rep, allf, alla = {}, [], []
    for nm, fs in sets.items():
        fv, av = [], []
        for f in fs[: a.n]:
            x = load48(f)[: 8 * 48000]
            fv.append(float(fline(torch.from_numpy(x)[None])[0]))
            av.append(amline(x))
        rep[nm] = {"n": len(fv), "fline_db_median": round(float(np.median(fv)), 3), "amline_median": round(float(np.median(av)), 3)}
        allf += fv
        alla += av
        print(nm, rep[nm], flush=True)
    rep["spearman_utt"] = round(float(spearmanr(allf, alla)[0]), 3)
    print("発話の順位相関 fsync 対 amline", rep["spearman_utt"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
