"""D644: 漏れを塞いだ内容の網(diag_envnet2)を製品の Rust の条件(D637 の生の条件)で ENP と同じ描画にかける(EN2P = 表の引き + envnet2 + 遅い揺らぎ + R の ΔS・非周期性 = 条件の 4 帯の周期性)。
開発の組 dev2 / dev3。規則は results/conv_p0/d644/prereg.yaml。
    uv run python d644_envnet2_eval.py --stage render | knob | rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

import d637_envnet_eval as E7
from d631_world_swap import C, FEATS, G, H, ROOT, SETS, analyze, keys, measure
from d632_f0env_swap import fit, lag, shift
from d634_artic_render import PRE, env_from_rows, synth, vdir
from d635_split import norm
from d638_artic_gate_g import env_shift
from d639_knob_hf import ap_from_per

OUT = ROOT / "results/conv_p0/d644"
E7.NET = ROOT / "results/diag_envnet2/last.pt"


def base(s: str, k: int, key: str) -> tuple:
    d = H / s / "d570"
    from d6_unit_cov import load48
    v = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)[PRE:PRE + 1600].astype(np.float64)
    xc = load48(d / C[k] / f"{key}_{C[k]}.wav").astype(np.float64)
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    lg = lag(xc, xs)
    _, sps, _ = (shift(m, lg) for m in analyze(xs))
    pin = fit(sps, len(v)).mean(1, keepdims=True)
    f0 = np.where(v[:, 26] > 0.5, 200.0 * np.exp(v[:, 25]), 0.0)
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319)[PRE:PRE + 1600].astype(np.float64)
    return E7.env_rows(s, k, key)["ENR"].astype(np.float64), f0, pin, ap_from_per(rc[:, 27:31], f0)


def render(it: tuple) -> int:
    s, k, key = it
    o = H / s / "d570" / f"EN2P_{k}" / f"{key}_EN2P_{k}.wav"
    o.parent.mkdir(parents=True, exist_ok=True)
    if not o.exists():
        rows, f0, pin, ap = base(s, k, key)
        sf.write(o, synth(f0, norm(env_from_rows(rows), pin), ap), 48000, subtype="PCM_16")
    return 1


def knob(key: str) -> dict:
    rows, f0, pin, ap = base("dev2", 7, key)
    ys = {a: synth(f0, norm(env_from_rows(rows, a), pin), ap) for a in (1.0, 0.9, 1.1)}
    sp = {a: analyze(y.astype(np.float64))[1] for a, y in ys.items()}
    return {"key": key, **{f"render_{a}": env_shift(sp[1.0], sp[a], f0 > 0) for a in (0.9, 1.1)}, "alpha1_identity": bool(np.array_equal(env_from_rows(rows, 1.0), env_from_rows(rows)))}


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["render", "knob", "rough"])
    a = ap_.parse_args()
    npr = int(os.environ.get("LVC_NPROC", "12"))
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    if a.stage == "render":
        with Pool(npr) as pl:
            print("rendered", sum(pl.map(render, its, chunksize=2)), flush=True)
        return 0
    if a.stage == "knob":
        with Pool(12) as pl:
            recs = pl.map(knob, keys("dev2")[:12])
        cals = json.loads((ROOT / "results/conv_p0/d640/knob_smooth.json").read_text())["cals_render"]
        med = {x: round(float(np.median([r[f"render_{x}"] for r in recs])), 3) for x in (0.9, 1.1)}
        rep = {"items": recs, "median_render": med, "cals_render": cals, "alpha1_identity": all(r["alpha1_identity"] for r in recs)}
        rep["knob_ok_cals"] = bool(rep["alpha1_identity"] and all(abs(med[x] - cals[str(x)]) <= 0.015 for x in (0.9, 1.1)))
        (OUT / "knob.json").write_text(json.dumps(rep, indent=1))
        print("knob", {k_: v_ for k_, v_ in rep.items() if k_ != "items"}, flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0], "ENP": f"ENP_{k}", "EN2P": f"EN2P_{k}"}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d639/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("ENP", "EN2P"):
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
