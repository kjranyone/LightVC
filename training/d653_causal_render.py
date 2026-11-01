"""D653: 製品の因果な構音の描画を Python で揃える。包絡 = envnet3 + 遅い補正を外す構造(τ 2 s)+ 遅い揺らぎ + R の ΔS + 描画の定数の較正(D652)・非周期性 = 条件の 4 帯の周期性。
大きさ = 条件の c0n(c0 = 10·c0n + C0_SIL を DCT の 0 次として行に足す・入力のオフラインの分析を使わない)・f0 = 条件の f0。
PR0 = そのまま・PR1 = 大きさ(対数)と f0(対数・有声の続きの中)に因果な 1 極 30 Hz の慣性・PR1S = PR1 + 表の引きの事後を温度 0.5 で鋭く(診断)。規則は results/conv_p0/d653/prereg.yaml。
    uv run python d653_causal_render.py --stage render | knob | rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import torch

import d637_envnet_eval as E7
import pae as PA
from d631_world_swap import C, FEATS, G, H, ROOT, SETS, analyze, keys, measure
from d634_artic_render import DCT, FC, FQ, PRE, synth, vdir
from d636_traj_gain import onepole
from d638_artic_gate_g import env_shift
from d639_knob_hf import ap_from_per

OUT = ROOT / "results/conv_p0/d653"
E7.NET = ROOT / "results/diag_envnet3/last.pt"
CURVE = 10.0 ** (np.array(json.loads((ROOT / "results/conv_p0/d652/curve.json").read_text())["db"]) / 10.0)
A30 = float(np.exp(-2 * np.pi * 30.0 * 0.005))
ARMS = ("PR0", "PR1", "PR1S")


def env_full(c0: np.ndarray, rows: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    lm = c0[:, None] * DCT[0][None] + (rows * 10.0) @ DCT[1:25]
    fc = FC * alpha
    return np.exp(2.0 * np.stack([np.interp(FQ, fc, r) for r in lm])) * CURVE


def smooth_voiced(lf: np.ndarray, v: np.ndarray) -> np.ndarray:
    y = lf.copy()
    for t in range(1, len(lf)):
        if v[t] and v[t - 1]:
            y[t] = A30 * y[t - 1] + (1.0 - A30) * lf[t]
    return y


def parts(s: str, k: int, key: str) -> dict:
    torch.set_num_threads(1)
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float64)
    vc = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27).astype(np.float64)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T.astype(np.float32)))[None]
    with torch.no_grad():
        y = E7.net()(t(rc[:, 31:231]), t(rc[:, 0:1]), t(rc[:, 25:26]), t(rc[:, 26:27]), t(rc[:, 1:25]), t(rc[:, 231:295]))[0].numpy().T.astype(np.float64)
    tab = rc[:, 1:25]
    d = y - tab
    hp = d - onepole(d, float(np.exp(-0.005 / 2.0)))
    post = vc[:, 1:25] - rc[:, 295:]
    T = np.fromfile(H / s / "d85_work" / f"en_{int(key[7:10]):03d}" / "enroll.bin", "<f4").reshape(24, -1).astype(np.float64)
    p = np.maximum(rc[:, 31:231], 1e-8)
    lt = (p @ T.T) / 10.0
    q = np.exp(np.log(p) / 0.5)
    q /= q.sum(1, keepdims=True)
    tab_s = tab - lt + (q @ T.T) / 10.0
    c0 = 10.0 * rc[:, 0] + PA.C0_SIL
    v = rc[:, 26] > 0.5
    lf = np.where(v, rc[:, 25], 0.0)
    c0s = onepole(c0[:, None], A30)[:, 0]
    lfs = smooth_voiced(lf, v)
    sl = slice(PRE, PRE + 1600)
    f0 = lambda x: np.where(v[sl], 200.0 * np.exp(x[sl]), 0.0)
    return {"PR0": ((tab + hp + post)[sl], c0[sl], f0(lf)), "PR1": ((tab + hp + post)[sl], c0s[sl], f0(lfs)), "PR1S": ((tab_s + hp + post)[sl], c0s[sl], f0(lfs)),
            "per": rc[sl, 27:31]}


def render(it: tuple) -> int:
    s, k, key = it
    pr = parts(s, k, key)
    for a in ARMS:
        o = H / s / "d570" / f"{a}_{k}" / f"{key}_{a}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            rows, c0, f0 = pr[a]
            sf.write(o, synth(f0, env_full(c0, rows), ap_from_per(pr["per"], f0)), 48000, subtype="PCM_16")
    return 1


def knob(key: str) -> dict:
    pr = parts("dev2", 7, key)
    rows, c0, f0 = pr["PR1"]
    ap = ap_from_per(pr["per"], f0)
    ys = {a: synth(f0, env_full(c0, rows, a), ap) for a in (1.0, 0.9, 1.1)}
    sp = {a: analyze(y.astype(np.float64))[1] for a, y in ys.items()}
    return {"key": key, **{f"render_{a}": env_shift(sp[1.0], sp[a], f0 > 0) for a in (0.9, 1.1)}}


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["render", "knob", "rough"])
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
    sysn = lambda k: {"G": G[k][0], "EN3HR": f"EN3HR_{k}"} | {x: f"{x}_{k}" for x in ARMS}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d652/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("EN3HR",) + ARMS:
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
