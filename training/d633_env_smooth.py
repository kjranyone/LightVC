"""D633: 候補の包絡の時間方向の揺れを変調の周波数で制限する(WORLD・f0 = 候補・非周期性 = 錨をずれでそろえる)。S0 / S40 / S20 / S10。錨との対の比較は D631 と同じ 4 統計。
規則は results/conv_p0/d633/prereg.yaml。
    uv run python d633_env_smooth.py --stage render
    uv run python d633_env_smooth.py --stage rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pyworld as pw
import scipy.signal as ss
import soundfile as sf

from d631_world_swap import C, FEATS, FP, G, H, ROOT, SETS, analyze, keys, measure
from d632_f0env_swap import fit, lag, shift, warp

OUT = ROOT / "results/conv_p0/d633"
FC = {"S0": None, "S40": 40.0, "S20": 20.0, "S10": 10.0}


def render(it: tuple) -> int:
    s, k, key = it
    d = H / s / "d570"
    outs = {a: d / f"{a}_{k}" / f"{key}_{a}_{k}.wav" for a in FC}
    if all(o.exists() for o in outs.values()):
        return 0
    from d6_unit_cov import load48
    xc = load48(d / C[k] / f"{key}_{C[k]}.wav").astype(np.float64)
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    lg = lag(xc, xs)
    f0c, spc, _ = analyze(xc)
    _, _, aps = analyze(xs)
    apg = fit(warp(shift(aps, lg), G[k][1]), len(f0c))
    ls = np.log(np.maximum(spc, 1e-20))
    for a, fc in FC.items():
        sp = spc if fc is None else np.exp(ss.sosfiltfilt(ss.butter(2, fc, fs=1000.0 / FP, output="sos"), ls, axis=0))
        y = pw.synthesize(f0c, np.ascontiguousarray(sp), np.ascontiguousarray(apg), 48000, FP)
        outs[a].parent.mkdir(parents=True, exist_ok=True)
        sf.write(outs[a], (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), 48000, subtype="PCM_16")
    return 1


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["render", "rough"])
    a = ap_.parse_args()
    npr = int(os.environ.get("LVC_NPROC", "12"))
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    if a.stage == "render":
        with Pool(npr) as pl:
            print("rendered", sum(pl.map(render, its, chunksize=2)), "of", len(its), flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0]} | {arm: f"{arm}_{k}" for arm in FC}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in FC:
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
        rep[arm]["parity"] = bool(all(rep[arm][k][f]["G_better"] <= 0.60 for k in C for f in FEATS))
    (OUT / "rough.json").write_text(json.dumps(rep, indent=1))
    print("parity", {arm: rep[arm]["parity"] for arm in FC}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
