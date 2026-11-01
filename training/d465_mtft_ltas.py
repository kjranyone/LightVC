"""D465: 男声の名残を音色の量で: 同じ目標・同じ文の 男声 TTS 入力の変換(MT = p00*)と 女声 TTS 入力の変換(FT = p01*)の長時間スペクトル(build_ltas.ltas_np の 12 帯・全帯の合計に対する相対 dB)の差。
目標ごとに対にした差の平均と目標でまとめたブートストラップ。入力そのもの(dev の SRC 対 devf の SRC)の差も並べる(入力の男女差がどれだけ出力に残るか)。
    uv run python d465_mtft_ltas.py --conv P43OP=../results/conv_p0/d452m/conv --out ../results/conv_p0/d465.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_ltas import BANDS, ltas_np
from d6_unit_cov import load48

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", nargs="+", required=True, help="NAME=<dir with p00*_CV.wav (male) and p01*_CV.wav (female)>")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(465)
    bands = [f"{lo}-{hi}" for lo, hi in zip(BANDS[:-1], BANDS[1:])]
    rep: dict = {"bands": bands}

    def paired(fm: dict, ff: dict) -> dict:
        keys = sorted(set(fm) & set(ff))
        d = np.array([ltas_np(load48(fm[k])[: 8 * 48000]) - ltas_np(load48(ff[k])[: 8 * 48000]) for k in keys])
        tg = np.array([int(k.split("_t")[1][:3]) for k in keys])
        us = sorted(set(tg))
        idx = {u: np.flatnonzero(tg == u) for u in us}
        bs = np.array([d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean(0) for _ in range(1000)])
        return {"n": len(keys), "diff_db": np.round(d.mean(0), 2).tolist(), "ci_lo": np.round(np.percentile(bs, 2.5, 0), 2).tolist(), "ci_hi": np.round(np.percentile(bs, 97.5, 0), 2).tolist()}

    srcm = {f.name[:10]: f for f in (H / "dev" / "d56").glob("p00*_SRC.wav")}
    srcf = {f.name[:10]: f for f in (H / "devf" / "d56").glob("p00*_SRC.wav")}
    rep["SRC_M-F"] = paired(srcm, srcf)
    print("SRC_M-F", rep["SRC_M-F"]["diff_db"], flush=True)
    for e in a.conv:
        nm, d = e.split("=", 1)
        fm = {f.name[:10]: f for f in Path(d).glob("p00*_CV.wav")}
        ff = {f"p{int(f.name[1:5]) - 100:04d}{f.name[5:10]}": f for f in Path(d).glob("p01*_CV.wav")}
        rep[f"{nm}_MT-FT"] = paired(fm, ff)
        print(f"{nm}_MT-FT", rep[f"{nm}_MT-FT"]["diff_db"], "lo", rep[f"{nm}_MT-FT"]["ci_lo"], "hi", rep[f"{nm}_MT-FT"]["ci_hi"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
