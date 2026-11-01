"""D667: 周期性の定数を目標の p75 から平均(per_mean)に替える PR5M(息の成分を少し増やす・k5 の shimmer の切り分け)。ほかは PR5(D666)と同じ。製品では --per-p75 を外すのと同じ。
規則は results/conv_p0/d667/prereg.yaml。
    uv run python d667_per_mean.py --stage render | rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

from causal_synth import synth_causal
from d631_world_swap import C, FEATS, G, H, ROOT, SETS, keys, measure
from d639_knob_hf import ap_from_per
from d652_render_cal import curve as curve1
from d653_causal_render import parts
from d663_shape_power import env4
from d666_target_cal import curve3

OUT = ROOT / "results/conv_p0/d667"


def render(it: tuple) -> int:
    s, k, key = it
    o = H / s / "d570" / f"PR5M_{k}" / f"{key}_PR5M_{k}.wav"
    o.parent.mkdir(parents=True, exist_ok=True)
    if o.exists():
        return 1
    ti = int(key[7:10])
    pr = parts(s, k, key)
    rows, c0, f0 = pr["PR1"]
    pm = np.array(json.loads((H / s / "d85_work" / f"en_{ti:03d}" / "enroll.json").read_text())["per_mean"], np.float64)
    vo = (pr["per"].max(1) > 0).astype(np.float64)
    per = np.clip(pm, 0.0, 1.5)[None] * vo[:, None]
    sf.write(o, synth_causal(f0, env4(c0, rows) * curve1() * curve3(s, ti), ap_from_per(per, f0)), 48000, subtype="PCM_16")
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
    sysn = lambda k: {"G": G[k][0], "PR5": f"PR5_{k}", "PR5M": f"PR5M_{k}"}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d666/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep = {}
    for arm in ("PR5", "PR5M"):
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
