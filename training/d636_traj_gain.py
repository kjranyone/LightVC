"""D636: 構音の条件の包絡の軌道の振れ幅の不足(回帰の平均化)が PT の明瞭さの不足を説明するか(開発の組 dev2 / dev3)。
G13 / G16 = 包絡の行の中域の振れ(因果な 1 極 16 Hz の低域 − 因果な指数平均 τ 0.25 s)を g 倍して PT と同じ描画。記録 = PT と PM(候補の音を条件の領域で表した行)の振れの比。
規則は results/conv_p0/d636/prereg.yaml。
    uv run python d636_traj_gain.py --stage render
    uv run python d636_traj_gain.py --stage rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

from d631_world_swap import C, FEATS, G, H, ROOT, SETS, analyze, keys, measure
from d632_f0env_swap import fit, lag, shift, warp
from d634_artic_render import PRE, env_from_rows, synth, vdir
from d635_split import norm, rows_of

OUT = ROOT / "results/conv_p0/d636"
GAINS = {"G13": 1.3, "G16": 1.6}
A_LP, A_EMA = float(np.exp(-2 * np.pi * 16.0 * 0.005)), float(np.exp(-0.005 / 0.25))


def onepole(x: np.ndarray, a: float) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * y[t - 1] + (1.0 - a) * x[t]
    return y


def mid(x: np.ndarray) -> np.ndarray:
    return onepole(x, A_LP) - onepole(x, A_EMA)


def gain(rows: np.ndarray, g: float) -> np.ndarray:
    return rows + (g - 1.0) * mid(rows)


def render(it: tuple) -> dict:
    s, k, key = it
    d = H / s / "d570"
    from d6_unit_cov import load48
    vall = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27).astype(np.float64)
    v = vall[PRE:PRE + 1600]
    xc = load48(d / C[k] / f"{key}_{C[k]}.wav").astype(np.float64)
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    lg = lag(xc, xs)
    _, sps, aps = (shift(m, lg) for m in analyze(xs))
    n = len(v)
    apg = fit(warp(aps, G[k][1]), n)
    pin = fit(sps, n).mean(1, keepdims=True)
    f0 = np.where(v[:, 26] > 0.5, 200.0 * np.exp(v[:, 25]), 0.0)
    rec = {"key": f"{s}|{k}|{key}"}
    if s == "dev2" and k == 7 and key in keys("dev2")[:4]:
        rec["g1_identity"] = bool(np.array_equal(gain(vall[:, 1:25], 1.0), vall[:, 1:25]))
    for a, g in GAINS.items():
        o = d / f"{a}_{k}" / f"{key}_{a}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            r = gain(vall[:, 1:25], g)[PRE:PRE + 1600]
            sf.write(o, synth(f0, norm(env_from_rows(r), pin), apg), 48000, subtype="PCM_16")
    pt, pm = v[:, 1:25], rows_of(xc, n)
    voiced = f0 > 0
    mp, mm = mid(vall[:, 1:25])[PRE:PRE + 1600][voiced], mid(pm)[voiced]
    rec["mid_std_ratio_PT_PM"] = [float(x) for x in mp.std(0) / (mm.std(0) + 1e-9)]
    rec["mid_corr"] = [float(np.corrcoef(mp[:, i], mm[:, i])[0, 1]) for i in range(24)]
    rec["mean_offset"] = [float(x) for x in (pt[voiced] - pm[voiced]).mean(0)]
    rec["pm_mean_abs"] = [float(x) for x in np.abs(pm[voiced]).mean(0)]
    return rec


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["render", "rough"])
    a = ap_.parse_args()
    npr = int(os.environ.get("LVC_NPROC", "12"))
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    if a.stage == "render":
        with Pool(npr) as pl:
            recs = pl.map(render, its, chunksize=2)
        (OUT / "render_rec.json").write_text(json.dumps(recs, indent=0))
        sr = np.array([r["mid_std_ratio_PT_PM"] for r in recs])
        cr = np.array([r["mid_corr"] for r in recs])
        print("g1 identity", all(r["g1_identity"] for r in recs if "g1_identity" in r), flush=True)
        print("mid std ratio PT/PM median per coef", np.round(np.median(sr, 0), 2).tolist(), "overall", round(float(np.median(sr)), 3), flush=True)
        print("mid corr median per coef", np.round(np.median(cr, 0), 2).tolist(), flush=True)
        mo = np.array([r["mean_offset"] for r in recs])
        pa = np.array([r["pm_mean_abs"] for r in recs])
        print("mean offset median per coef", np.round(np.median(mo, 0), 3).tolist(), "PM |row| median", np.round(np.median(pa, 0), 3).tolist(), flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0], "PT": f"PT_{k}"} | {x: f"{x}_{k}" for x in GAINS}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d634/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("PT",) + tuple(GAINS):
        rep[arm] = {}
        for k in C:
            rep[arm][k] = {}
            nm = sysn(k)
            for f, (_, hi) in FEATS.items():
                xs = np.array([res[str(H / s / "d570" / nm[arm] / f"{key}_{nm[arm]}.wav")][f] for s, kk, key in its if kk == k])
                gs = np.array([res[str(H / s / "d570" / nm["G"] / f"{key}_{nm['G']}.wav")][f] for s, kk, key in its if kk == k])
                m = np.isfinite(xs) & np.isfinite(gs)
                gb = float(np.mean(gs[m] > xs[m])) if hi else float(np.mean(gs[m] < xs[m]))
                rep[arm][k][f] = {"G_better": round(gb, 3), "median_X": round(float(np.median(xs[m])), 4), "median_G": round(float(np.median(gs[m])), 4)}
            print(arm, k, {f: (v["G_better"], v["median_X"], v["median_G"]) for f, v in rep[arm][k].items()}, flush=True)
        rep[arm]["flutter_guard"] = bool(all(rep[arm][k]["pflutter"]["G_better"] <= 0.60 for k in C))
    (OUT / "rough.json").write_text(json.dumps(rep, indent=1))
    print("flutter guard", {arm: rep[arm]["flutter_guard"] for arm in rep}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
