"""D649: 内容の網の補正(網 − 表の引き)の遅い成分を因果な指数平均で引く(EN3H = τ 2 s・EN3Q = τ 0.5 s)。包絡の長い時間の形は目標の表が運び、網は速い音素の細部だけを足す構造。
envnet3・非周期性 = 条件の 4 帯の周期性・ほかは EN3P と同じ描画。規則は results/conv_p0/d649/prereg.yaml。
    uv run python d649_slow_hp.py --stage render | rough
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
from d631_world_swap import C, FEATS, G, H, ROOT, SETS, keys, measure
from d634_artic_render import FQ, PRE, env_from_rows, synth, vdir
from d635_split import norm
from d636_traj_gain import onepole
from d648_envnet3_eval import base

OUT = ROOT / "results/conv_p0/d649"
E7.NET = ROOT / "results/diag_envnet3/last.pt"
TAUS = {"EN3H": 2.0, "EN3Q": 0.5}
HI, LO = (FQ >= 4000) & (FQ < 10000), (FQ >= 100) & (FQ < 4000)


def render(it: tuple) -> dict:
    s, k, key = it
    torch.set_num_threads(1)
    _, f0, pin, ap = base(s, k, key)
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float32)
    vc = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T))[None]
    with torch.no_grad():
        y = E7.net()(t(rc[:, 31:231]), t(rc[:, 0:1]), t(rc[:, 25:26]), t(rc[:, 26:27]), t(rc[:, 1:25]), t(rc[:, 231:295]))[0].numpy().T.astype(np.float64)
    tab = rc[:, 1:25].astype(np.float64)
    post = (vc[:, 1:25] - rc[:, 295:]).astype(np.float64)
    d = y - tab
    sl = slice(PRE, PRE + 1600)
    m = (rc[sl, 0] > 0.05) & (f0 > 0)
    rec = {"key": f"{s}|{k}|{key}"}
    for a, tau in TAUS.items():
        r = (tab + d - onepole(d, float(np.exp(-0.005 / tau))) + post)[sl]
        e = norm(env_from_rows(r), pin)
        rec[a] = float(10 * np.log10(e[m][:, HI].sum() / e[m][:, LO].sum()))
        o = H / s / "d570" / f"{a}_{k}" / f"{key}_{a}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            sf.write(o, synth(f0, e, ap), 48000, subtype="PCM_16")
    e = norm(env_from_rows((y + post)[sl]), pin)
    rec["EN3P"] = float(10 * np.log10(e[m][:, HI].sum() / e[m][:, LO].sum()))
    e = norm(env_from_rows((tab + post)[sl]), pin)
    rec["TAB"] = float(10 * np.log10(e[m][:, HI].sum() / e[m][:, LO].sum()))
    return rec


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["render", "rough"])
    a = ap_.parse_args()
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    npr = int(os.environ.get("LVC_NPROC", "12"))
    if a.stage == "render":
        with Pool(npr) as pl:
            recs = pl.map(render, its, chunksize=2)
        (OUT / "render_rec.json").write_text(json.dumps(recs, indent=0))
        for k in C:
            rk = [r for r in recs if r["key"].split("|")[1] == str(k)]
            print("voiced hi/lo dB", k, {x: round(float(np.median([r[x] for r in rk])), 2) for x in ("TAB", "EN3P") + tuple(TAUS)}, flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0], "EN3P": f"EN3P_{k}"} | {x: f"{x}_{k}" for x in TAUS}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d648/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("EN3P",) + tuple(TAUS):
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
