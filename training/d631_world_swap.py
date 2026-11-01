"""D631: 候補のザラつきの切り分け(開発の組 dev2 / dev3)。WRT = 候補の WORLD の往復・WAP = 候補の f0 と包絡 + 錨の非周期性(入力の D4C を α で伸ばす)。
錨との対の比較(D629 hc_hi・D630 phc_lo / phc_hi / pflutter)で『錨が良い側』の割合を読む。規則は results/conv_p0/d631/prereg.yaml。
    uv run python d631_world_swap.py --stage render
    uv run python d631_world_swap.py --stage rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import scipy.signal as ss
import soundfile as sf

import d629_smooth as D9
import d630_pitchsync as D0

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/conv_p0/d631"
SETS = ("dev2", "dev3")
C = {5: "OURS5RY58GQ3", 7: "OURS7RY58GQ3", 12: "OURS12RY58G"}
G = {5: ("GU5_a112", 1.12), 7: ("GU7_a112", 1.12), 12: ("GU12_a120", 1.20)}
FP = 5.0
FEATS = {"hc_hi": (D9, True), "phc_lo": (D0, True), "phc_hi": (D0, True), "pflutter": (D0, False)}


def analyze(x: np.ndarray) -> tuple:
    f0, t = pw.harvest(ss.resample_poly(x, 1, 3), 16000, f0_floor=50.0, f0_ceil=800.0, frame_period=FP)
    return f0, pw.cheaptrick(x, f0, t, 48000), pw.d4c(x, f0, t, 48000)


def render(it: tuple) -> int:
    s, k, key = it
    d = H / s / "d570"
    oa, ob = d / f"WRT{k}" / f"{key}_WRT{k}.wav", d / f"WAP{k}" / f"{key}_WAP{k}.wav"
    if oa.exists() and ob.exists():
        return 0
    from d6_unit_cov import load48
    xc = load48(d / C[k] / f"{key}_{C[k]}.wav").astype(np.float64)
    f0, sp, ap = analyze(xc)
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    _, _, aps = analyze(xs)
    a = G[k][1]
    fq = np.linspace(0, 24000, aps.shape[1])
    apg = np.stack([np.interp(fq / a, fq, r) for r in aps])
    n = len(f0)
    apg = apg[:n] if len(apg) >= n else np.concatenate([apg, np.repeat(apg[-1:], n - len(apg), 0)])
    for o, A in ((oa, ap), (ob, apg)):
        y = pw.synthesize(f0, np.ascontiguousarray(sp), np.ascontiguousarray(A), 48000, FP)
        o.parent.mkdir(parents=True, exist_ok=True)
        sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), 48000, subtype="PCM_16")
    return 1


def keys(s: str) -> list:
    return sorted(f.name[:10] for f in (H / s / "d570" / C[5]).glob(f"p*_{C[5]}.wav"))


def measure(path: str) -> tuple:
    x = D9.load(Path(path))
    r9, r0 = D9.feats(x), D0.feats(x)
    return path, {"hc_hi": r9.get("hc_hi", float("nan")), "phc_lo": r0["phc_lo"], "phc_hi": r0["phc_hi"], "pflutter": r0["pflutter"]}


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
    sysn = lambda k: {"C": C[k], "G": G[k][0], "WRT": f"WRT{k}", "WAP": f"WAP{k}"}
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()})
    with Pool(npr) as pl:
        res = dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps(res, indent=0))
    rep: dict = {}
    for arm in ("C", "WRT", "WAP"):
        rep[arm] = {}
        for k in C:
            rep[arm][k] = {}
            for f, (_, hi) in FEATS.items():
                xs, gs = [], []
                for s, kk, key in its:
                    if kk != k:
                        continue
                    nm = sysn(k)
                    xs.append(res[str(H / s / "d570" / nm[arm] / f"{key}_{nm[arm]}.wav")][f])
                    gs.append(res[str(H / s / "d570" / nm["G"] / f"{key}_{nm['G']}.wav")][f])
                xs, gs = np.array(xs), np.array(gs)
                m = np.isfinite(xs) & np.isfinite(gs)
                gb = float(np.mean(gs[m] > xs[m])) if hi else float(np.mean(gs[m] < xs[m]))
                rep[arm][k][f] = {"G_better": round(gb, 3), "median_X": round(float(np.median(xs[m])), 4), "median_G": round(float(np.median(gs[m])), 4), "n": int(m.sum())}
            print(arm, k, {f: (v["G_better"], v["median_X"], v["median_G"]) for f, v in rep[arm][k].items()}, flush=True)
        rep[arm]["parity"] = bool(all(rep[arm][k][f]["G_better"] <= 0.60 for k in C for f in FEATS))
    (OUT / "rough.json").write_text(json.dumps(rep, indent=1))
    print("parity", {arm: rep[arm]["parity"] for arm in rep}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
