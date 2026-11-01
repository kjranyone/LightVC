"""D533 = D519 の一般化(群を引数に)+ 取りこぼし(wonly = WORLD が入力を有声・F1 の条件が無声・出力の周期は入力の f0 を声域の差で移した値): F1 の条件だけが有声で WORLD が入力を無声とするフレーム(F1 のみ = 息の端)と両方有声のフレームで、出力の周期性(正規化自己相関の最大・条件の f0 の周期 ±10%・25 ms 窓・800 Hz 以下)を
入力 SRC と各群で比べる。フレームの種類と条件の f0 は F1(eval_<set>_pre の書き出し)で決める(群が別の有声で描画されていても同じフレームで比べる)。発話ごとの F1 のみの 群 − 入力 の中央値と対の差(B − A)。
    uv run python d533_breath_pitch.py --set devff5 --groups P43=<dir>:OP15 TF0=<dir>:TF0 CP=<dir>:CP --pair TF0 P43 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P
from d519_overvoiced import load16, periodicity, SR


def main() -> int:
    import pyworld as pw
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--pair", nargs=2, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = P.H / a.set
    G = {g.split("=")[0]: (Path(g.split("=")[1].split(":")[0]), g.split(":")[-1]) for g in a.groups}
    ed = P.H / "e2e" / f"eval_{a.set}_pre"
    agg = {nm: {g: [] for g in ["SRC"] + list(G)} for nm in ("both", "f1only", "wonly")}
    per = {g: {} for g in G}
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (ed / f"{k}.cond").exists():
            continue
        c, _ = P.load_dump(ed, k)
        c = c[12 * 200:12 * 200 + 8 * 200]
        fc = np.where(c[:, 26] > 0.5, 200.0 * np.exp(c[:, 25]), 0.0)
        xs = load16(S / "d56" / f"{k}_SRC.wav")
        f0w, tw = pw.dio(xs, SR, f0_floor=90, f0_ceil=1100, frame_period=5.0)
        f0w = pw.stonemask(xs, f0w, tw, SR)
        T = min(len(fc), len(f0w))
        fc, f0w = fc[:T], f0w[:T]
        vc, vw = fc > 0, f0w > 0
        both = vc & vw
        shift = np.median(np.log2(fc[both] / f0w[both])) if both.sum() > 20 else 0.0
        p_in = periodicity(xs, np.where(vw, f0w, np.where(vc, fc / 2 ** shift, 0.0)))[:T]
        f1o = vc & ~vw
        wo = vw & ~vc
        fo = np.where(vc, fc, np.where(vw, f0w * 2 ** shift, 0.0))
        pv = {"SRC": p_in}
        for g, (d, sfx) in G.items():
            pv[g] = periodicity(load16(d / f"{k}_{sfx}.wav"), fo)[:T]
        for nm, m in (("both", both), ("f1only", f1o), ("wonly", wo)):
            for g, v in pv.items():
                v = v[m]
                agg[nm][g].extend(v[~np.isnan(v)].tolist())
        if f1o.sum() > 5:
            for g in G:
                per[g][k] = float(np.nanmedian(pv[g][f1o] - p_in[f1o]))
    rep = {nm: {g: [round(float(np.median(v)), 3), len(v)] for g, v in d.items()} for nm, d in agg.items()}
    rep["both_frac_below_0.6"] = {g: round(float(np.mean(np.array(v) < 0.6)), 4) for g, v in agg["both"].items()}
    rep["f1only_minus_src_per_utt"] = {g: round(float(np.median(list(d.values()))), 3) for g, d in per.items()}
    if a.pair:
        b, c_ = a.pair
        ks = sorted(set(per[b]) & set(per[c_]))
        d = np.array([per[b][k] - per[c_][k] for k in ks])
        rng = np.random.default_rng(533)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        rep[f"{b}-{c_}_f1only"] = [round(float(d.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
