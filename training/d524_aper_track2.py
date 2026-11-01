"""D524 = D522 の一般化(組と群を引数に): 入力 SRC と各群の出力の WORLD D4C の非周期性(1〜3 kHz・3〜6 kHz)の時系列を両方有声のフレームで比べる(相関・はっきり ⇔ かすれの差)。
群 = NAME=<ディレクトリ>:<接尾辞>(<dir>/<pNNNN_tNNN>_<接尾辞>.wav)。発話ごとの値の中央値と、群の対の差(B − A)の操作者でまとめたブートストラップの 95% 区間。
    uv run python d524_aper_track2.py --set devff23x --groups P43=<dir>:OP15 P53=<dir>:OP15 --pair P53 P43 --out <json>
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
    ap.add_argument("--pair", nargs=2, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = P.H / a.set
    G = {g.split("=")[0]: (Path(g.split("=")[1].split(":")[0]), g.split(":")[-1]) for g in a.groups}
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    rows: dict = {g: {} for g in G}
    spk = {}
    for r in pairs:
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
            rows[g][k] = (float(np.corrcoef(xi, xo)[0, 1]), float(xo[xi >= q3].mean() - xo[xi <= q1].mean()), float(xi[xi >= q3].mean() - xi[xi <= q1].mean()))
            spk[k] = r.get("spk", k)
    rep = {g: {"corr_1_3k": round(float(np.median([v[0] for v in d.values()])), 3), "breathy_minus_clear_out": round(float(np.median([v[1] for v in d.values()])), 2),
               "breathy_minus_clear_in": round(float(np.median([v[2] for v in d.values()])), 2), "n": len(d)} for g, d in rows.items()}
    if a.pair:
        b, c = a.pair
        ks = sorted(set(rows[b]) & set(rows[c]))
        names = sorted({spk[k] for k in ks})
        cl = np.array([names.index(spk[k]) for k in ks])
        us = np.unique(cl)
        idx = {u: np.flatnonzero(cl == u) for u in us}
        rng = np.random.default_rng(524)
        for j, nm in ((0, "corr"), (1, "contrast")):
            d = np.array([rows[b][k][j] - rows[c][k][j] for k in ks])
            bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
            rep[f"{b}-{c}_{nm}"] = [round(float(d.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
