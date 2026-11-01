"""D663: 形のパワーの係数の速い揺れだけを外す描画 PR4。フレームのパワー = exp(2·c0 / √128)·S̄(t)・S̄ = 形のパワーの係数 S(t) = mean_f exp(2·形) を因果な 1 極(25 Hz)でならしたもの
(PR2 = S をそのまま・PR3 = S を定数)。表の引きの較正は PR2 の曲線 2(D659)・描画の較正は曲線 1(D652)。規則は results/conv_p0/d663/prereg.yaml。
    uv run python d663_shape_power.py --stage render | knob | rough | dev
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

from d631_world_swap import C, FEATS, G, H, ROOT, SETS, analyze, keys, measure
from d634_artic_render import DCT, FC, FQ, synth
from d636_traj_gain import onepole
from d638_artic_gate_g import env_shift
from d639_knob_hf import ap_from_per
from d651_copy_hf import best8, hf_tilt
from d652_render_cal import curve as curve1
from d653_causal_render import parts
from d659_table_cal import curve2

OUT = ROOT / "results/conv_p0/d663"
A25 = float(np.exp(-2 * np.pi * 25.0 * 0.005))


def env4(c0: np.ndarray, rows: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    lm = (rows * 10.0) @ DCT[1:25]
    e = np.exp(2.0 * np.stack([np.interp(FQ, FC * alpha, r) for r in lm]))
    s = e.mean(1)
    sb = np.exp(onepole(np.log(s)[:, None], A25)[:, 0])
    return e / s[:, None] * sb[:, None] * np.exp(2.0 * c0 / np.sqrt(128.0))[:, None]


def render(it: tuple) -> int:
    s, k, key = it
    o = H / s / "d570" / f"PR4_{k}" / f"{key}_PR4_{k}.wav"
    o.parent.mkdir(parents=True, exist_ok=True)
    if not o.exists():
        pr = parts(s, k, key)
        rows, c0, f0 = pr["PR1"]
        sf.write(o, synth(f0, env4(c0, rows) * curve1() * curve2(), ap_from_per(pr["per"], f0)), 48000, subtype="PCM_16")
    return 1


def knob(key: str) -> dict:
    pr = parts("dev2", 7, key)
    rows, c0, f0 = pr["PR1"]
    ap = ap_from_per(pr["per"], f0)
    cv = curve1() * curve2()
    ys = {a: synth(f0, env4(c0, rows, a) * cv, ap) for a in (1.0, 0.9, 1.1)}
    sp = {a: analyze(y.astype(np.float64))[1] for a, y in ys.items()}
    return {"key": key, **{f"render_{a}": env_shift(sp[1.0], sp[a], f0 > 0) for a in (0.9, 1.1)}}


def dev_job(it: tuple) -> dict:
    s, k, key = it
    from d6_unit_cov import load48
    out = {"k": k}
    for a in ("PR2", "PR3", "PR4"):
        x, _ = sf.read(H / s / "d570" / f"{a}_{k}" / f"{key}_{a}_{k}.wav")
        out[a] = hf_tilt(np.asarray(x, np.float64))
    out["REF"] = hf_tilt(best8(load48(H / s / "d37" / "_work" / f"ref_{int(key[7:10]):03d}.wav").astype(np.float64)))
    return out


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["render", "knob", "rough", "dev"])
    a = ap_.parse_args()
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    npr = int(os.environ.get("LVC_NPROC", "12"))
    if a.stage == "render":
        with Pool(npr) as pl:
            print("rendered", sum(pl.map(render, its, chunksize=2)), flush=True)
        return 0
    if a.stage == "knob":
        with Pool(12) as pl:
            recs = pl.map(knob, keys("dev2")[:12])
        cals = json.loads((ROOT / "results/conv_p0/d640/knob_smooth.json").read_text())["cals_render"]
        med = {x: round(float(np.median([r[f"render_{x}"] for r in recs])), 3) for x in (0.9, 1.1)}
        rep = {"items": recs, "median_render": med, "cals_render": cals, "knob_ok_cals": bool(all(abs(med[x] - cals[str(x)]) <= 0.015 for x in (0.9, 1.1)))}
        (OUT / "knob.json").write_text(json.dumps(rep, indent=1))
        print("knob", {k_: v_ for k_, v_ in rep.items() if k_ != "items"}, flush=True)
        return 0
    if a.stage == "dev":
        with Pool(npr) as pl:
            recs = pl.map(dev_job, its, chunksize=4)
        rep: dict = {}
        for k in C:
            rk = [r for r in recs if r["k"] == k]
            rep[k] = {x: {"d_hf": round(float(np.median([r[x][0] - r["REF"][0] for r in rk])), 2), "d_tilt": round(float(np.median([r[x][1] - r["REF"][1] for r in rk])), 3)} for x in ("PR2", "PR3", "PR4")}
            print("target-relative", k, json.dumps(rep[k]), flush=True)
        (OUT / "target_rel.json").write_text(json.dumps(rep, indent=1))
        return 0
    sysn = lambda k: {"G": G[k][0], "PR3": f"PR3_{k}", "PR4": f"PR4_{k}"}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d662/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep = {}
    for arm in ("PR3", "PR4"):
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
