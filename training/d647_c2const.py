"""D647: envnet2 の C2 の入力を学習の平均のベクトル(定数)にした構音の描画 EN2PC(非周期性 = 条件の 4 帯の周期性・ほかは EN2P と同じ)。C2 が包絡の網の明瞭さに要るかを確かめる。
規則は results/conv_p0/d647/prereg.yaml。
    uv run python d647_c2const.py --stage stats | render | rough
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
from d634_artic_render import PRE, env_from_rows, synth, vdir
from d635_split import norm
from d644_envnet2_eval import base

OUT = ROOT / "results/conv_p0/d647"
E7.NET = ROOT / "results/diag_envnet2/last.pt"


def render(it: tuple) -> int:
    s, k, key = it
    o = H / s / "d570" / f"EN2PC_{k}" / f"{key}_EN2PC_{k}.wav"
    o.parent.mkdir(parents=True, exist_ok=True)
    if o.exists():
        return 1
    torch.set_num_threads(1)
    _, f0, pin, ap = base(s, k, key)
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float32)
    vc = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)
    c2m = np.broadcast_to(np.array(json.loads((OUT / "c2_mean.json").read_text())["c2_mean"], np.float32), (len(rc), 64))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T))[None]
    with torch.no_grad():
        y = E7.net()(t(rc[:, 31:231]), t(rc[:, 0:1]), t(rc[:, 25:26]), t(rc[:, 26:27]), t(rc[:, 1:25]), t(c2m))[0].numpy().T
    rows = (y + vc[:, 1:25] - rc[:, 295:])[PRE:PRE + 1600].astype(np.float64)
    sf.write(o, synth(f0, norm(env_from_rows(rows), pin), ap), 48000, subtype="PCM_16")
    return 1


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["stats", "render", "rough"])
    a = ap_.parse_args()
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    npr = int(os.environ.get("LVC_NPROC", "12"))
    if a.stage == "stats":
        from d643_infer_inputs import train_stats
        ts = train_stats()
        (OUT / "c2_mean.json").write_text(json.dumps({"c2_mean": ts["c2_mean"]}))
        print("c2 mean saved", round(float(np.linalg.norm(ts["c2_mean"])), 4), flush=True)
        return 0
    if a.stage == "render":
        with Pool(npr) as pl:
            print("rendered", sum(pl.map(render, its, chunksize=2)), flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0], "EN2P": f"EN2P_{k}", "EN2PC": f"EN2PC_{k}"}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d644/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("EN2P", "EN2PC"):
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
