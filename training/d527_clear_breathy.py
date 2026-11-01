"""D527(耳の所見「F0 がしっかり出ず・かすれ」の型の切り分け): 入力の息っぽさの四分位でフレームを分け、出力の非周期性 − 入力の非周期性(WORLD D4C・1〜3 kHz・dB)の中央値を
はっきり(下位 25%)・中間・息っぽい(上位 25%)のフレームで並べる。はっきりしたフレームで出力が大きく上 = 『はっきりすべき所でかすれる』型。群 = 引数。
    uv run python d527_clear_breathy.py --set devff23x --groups P43=<dir>:OP15 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P
from d522_aper_track import ap_series


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = P.H / a.set
    G = {g.split("=")[0]: (Path(g.split("=")[1].split(":")[0]), g.split(":")[-1]) for g in a.groups}
    acc: dict = {g: {"clear": [], "mid": [], "breathy": [], "in_clear": [], "in_breathy": []} for g in G}
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        src = S / "d56" / f"{k}_SRC.wav"
        if not src.exists():
            continue
        a_in, v_in = ap_series(src)
        for g, (d, sfx) in G.items():
            p = d / f"{k}_{sfx}.wav"
            if not p.exists():
                continue
            a_o, v_o = ap_series(p)
            T = min(len(a_in), len(a_o))
            m = v_in[:T] & v_o[:T]
            if m.sum() < 40:
                continue
            xi, xo = a_in[:T, 0][m], a_o[:T, 0][m]
            q1, q3 = np.percentile(xi, [25, 75])
            acc[g]["clear"].append(float(np.median(xo[xi <= q1] - xi[xi <= q1])))
            acc[g]["mid"].append(float(np.median(xo[(xi > q1) & (xi < q3)] - xi[(xi > q1) & (xi < q3)])))
            acc[g]["breathy"].append(float(np.median(xo[xi >= q3] - xi[xi >= q3])))
            acc[g]["in_clear"].append(float(np.median(xi[xi <= q1])))
            acc[g]["in_breathy"].append(float(np.median(xi[xi >= q3])))
    rep = {g: {m: round(float(np.median(v)), 2) for m, v in d.items()} for g, d in acc.items()}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
