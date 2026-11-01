"""D632: 錨の非周期性の上で f0 と包絡の出どころを入れ替える(WFG = 錨の f0 + 候補の包絡・WEG = 候補の f0 + 錨の包絡)。錨との対の比較は D631 と同じ 4 統計。
規則は results/conv_p0/d632/prereg.yaml。
    uv run python d632_f0env_swap.py --stage render
    uv run python d632_f0env_swap.py --stage rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf

from d631_world_swap import C, FEATS, FP, G, H, ROOT, SETS, analyze, keys, measure

OUT = ROOT / "results/conv_p0/d632"


def warp(m: np.ndarray, a: float) -> np.ndarray:
    fq = np.linspace(0, 24000, m.shape[1])
    return np.stack([np.interp(fq / a, fq, r) for r in m])


def fit(m: np.ndarray, n: int) -> np.ndarray:
    return m[:n] if len(m) >= n else np.concatenate([m, np.repeat(m[-1:], n - len(m), 0)])


def lag(a: np.ndarray, b: np.ndarray) -> int:
    e = lambda x: np.log(np.convolve(x ** 2, np.ones(240), "valid")[::240] + 1e-9)
    ea, eb = e(a), e(b)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    r = [float(np.dot(ea[max(0, l):n + min(0, l)], eb[max(0, -l):n - max(0, l)])) for l in range(-10, 11)]
    return int(np.argmax(r)) - 10


def shift(m: np.ndarray, l: int) -> np.ndarray:
    if l > 0:
        return np.concatenate([np.repeat(m[:1], l, 0), m])[: len(m)]
    if l < 0:
        return np.concatenate([m[-l:], np.repeat(m[-1:], -l, 0)])
    return m


def render(it: tuple) -> int:
    s, k, key = it
    d = H / s / "d570"
    oa, ob = d / f"WFG{k}" / f"{key}_WFG{k}.wav", d / f"WEG{k}" / f"{key}_WEG{k}.wav"
    from d6_unit_cov import load48
    xc = load48(d / C[k] / f"{key}_{C[k]}.wav").astype(np.float64)
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    lg = lag(xc, xs)
    if oa.exists() and ob.exists():
        return lg
    f0c, spc, _ = analyze(xc)
    f0s, sps, aps = (shift(m, lg) for m in analyze(xs))
    a = G[k][1]
    f0g = fit(f0s[:, None], len(f0c))[:, 0] * 2 ** (k / 12)
    apg, spg = fit(warp(aps, a), len(f0c)), fit(warp(sps, a), len(f0c))
    for o, f0, sp in ((oa, f0g, spc), (ob, f0c, spg)):
        y = pw.synthesize(np.ascontiguousarray(f0), np.ascontiguousarray(sp), np.ascontiguousarray(apg), 48000, FP)
        o.parent.mkdir(parents=True, exist_ok=True)
        sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), 48000, subtype="PCM_16")
    return lg


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["render", "rough"])
    a = ap_.parse_args()
    npr = int(os.environ.get("LVC_NPROC", "12"))
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    if a.stage == "render":
        with Pool(npr) as pl:
            lg = np.array(pl.map(render, its, chunksize=2))
        rec = {"lag_frames_median": float(np.median(lg)), "lag_abs_le1": float(np.mean(np.abs(lg) <= 1)), "lag_hist": {int(v): int((lg == v).sum()) for v in np.unique(lg)}}
        (OUT / "lag.json").write_text(json.dumps(rec, indent=1))
        print("rendered", len(its), "lag", rec, flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0], "WAP": f"WAP{k}", "WFG": f"WFG{k}", "WEG": f"WEG{k}"}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("WAP", "WFG", "WEG"):
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
    pw_, pe = rep["WFG"]["parity"], rep["WEG"]["parity"]
    rep["fork"] = "either" if pw_ and pe else "f0" if pw_ else "env" if pe else "both"
    (OUT / "rough.json").write_text(json.dumps(rep, indent=1))
    print("parity", {arm: rep[arm]["parity"] for arm in ("WAP", "WFG", "WEG")}, "fork", rep["fork"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
