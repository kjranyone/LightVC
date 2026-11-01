"""D669: 大きさの音節の動きを弱める程度の得失(基準 PR5・開発の組 dev2 / dev3)。c0 の出力 = s + γ·(c0₃₀ − s)・s = 発声のフレームだけで更新する因果な 1 極 2 Hz・無音のフレームはそのまま。
G7 = γ 0.7・G5 = γ 0.5(全 k)。規則は results/conv_p0/d669/prereg.yaml。
    uv run python d669_level_comp.py --stage render | rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

import d637_envnet_eval as E7
import pae as PA
from causal_synth import synth_causal
from d631_world_swap import C, FEATS, G, H, ROOT, SETS, keys, measure
from d634_artic_render import PRE
from d636_traj_gain import onepole
from d639_knob_hf import ap_from_per
from d652_render_cal import curve as curve1
from d653_causal_render import parts
from d663_shape_power import env4
from d666_target_cal import curve3

OUT = ROOT / "results/conv_p0/d669"
A2 = float(np.exp(-2 * np.pi * 2.0 * 0.005))
A30 = float(np.exp(-2 * np.pi * 30.0 * 0.005))
GAMMA = {"G7": 0.7, "G5": 0.5}


def comp(c0n: np.ndarray, gamma: float) -> np.ndarray:
    c0 = 10.0 * c0n + PA.C0_SIL
    c30 = onepole(c0[:, None], A30)[:, 0]
    out = c30.copy()
    s = None
    for t in range(len(c0)):
        if c0n[t] <= 0.05:
            continue
        s = c30[t] if s is None else A2 * s + (1 - A2) * c30[t]
        out[t] = s + gamma * (c30[t] - s)
    return out


def render(it: tuple) -> int:
    s, k, key = it
    pr = parts(s, k, key)
    rows, _, f0 = pr["PR1"]
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float64)
    cv = curve1() * curve3(s, int(key[7:10]))
    ap = ap_from_per(pr["per"], f0)
    for a, g in GAMMA.items():
        o = H / s / "d570" / f"{a}_{k}" / f"{key}_{a}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            c0 = comp(rc[:, 0], g)[PRE:PRE + 1600]
            sf.write(o, synth_causal(f0, env4(c0, rows) * cv, ap), 48000, subtype="PCM_16")
    return 1


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["render", "rough"])
    a = ap_.parse_args()
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    npr = int(os.environ.get("LVC_NPROC", "12"))
    if a.stage == "render":
        with Pool(npr) as pl:
            print("rendered", sum(pl.map(render, its, chunksize=2)), flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0], "PR5": f"PR5_{k}"} | {x: f"{x}_{k}" for x in GAMMA}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d666/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep = {}
    for arm in ("PR5",) + tuple(GAMMA):
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
        rep[arm]["parity4"] = bool(all(rep[arm][k][f]["G_better"] <= 0.60 for k in C for f in FEATS))
    (OUT / "rough.json").write_text(json.dumps(rep, indent=1))
    print("parity4", {arm: rep[arm]["parity4"] for arm in rep}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
