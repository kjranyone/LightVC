"""D657: 声の強さと声門の傾斜の結びつき(大きい声ほど明るい)を構音の描画に足す。学習の側の実女声(開発の組に依らない)で、有声の発声のフレームの
実の包絡の形 − 表の引きの形(帯ごと・log-mel)を、相対の大きさ r(t) = 大きさ − 因果な指数平均(τ 1 s)に回帰: d_b(t) = a_b + β_b·r(t)(固定の定数)。
描画: EFT = PR1(D653)の有声のフレームの行 + (a + β·r(t)) を DCT 1〜24 へ・EFT0 = a だけ(記録)。r は条件の c0n から同じ式で。規則は results/conv_p0/d657/prereg.yaml。
    uv run python d657_effort_tilt.py --stage fit | render | knob | rough | dev
"""
from __future__ import annotations

import argparse
import json
import os
import random
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import torch

import d637_envnet_eval as E7
from d631_world_swap import C, FEATS, G, H, ROOT, SETS, analyze, keys, measure
from d634_artic_render import DCT, PRE, synth
from d638_artic_gate_g import env_shift
from d639_knob_hf import ap_from_per
from d651_copy_hf import best8, hf_tilt
from d653_causal_render import env_full, parts

OUT = ROOT / "results/conv_p0/d657"
D24 = DCT[1:25]
AT = float(np.exp(-0.005 / 1.0))
DB = 20.0 / np.log(10.0) / np.sqrt(128.0)


def rel_level(c0: np.ndarray, sp: np.ndarray) -> np.ndarray:
    lv = c0 * DB
    r = np.zeros_like(lv)
    m = None
    for t in range(len(lv)):
        if not sp[t]:
            continue
        m = lv[t] if m is None else AT * m + (1.0 - AT) * lv[t]
        r[t] = lv[t] - m
    return r


def fit_job(row: tuple) -> tuple | None:
    import nvoc as N
    import pae as PA
    import pyworld as pw
    import scipy.signal as ss
    import train_rvoc as TR
    torch.set_num_threads(1)
    wav, _, sr, dur, spk = row[:5]
    x = TR.read_span(wav, sr, int(0.5 * N.SR), 8 * N.SR).astype(np.float64)
    if len(x) < 8 * N.SR or np.sqrt((x ** 2).mean()) < 1e-3:
        return None
    Tn = len(x) // N.HOP - 2
    seg = Tn * N.HOP
    xa = np.concatenate([np.zeros(PA.A), x[:seg], np.zeros(PA.A)])
    f0, _ = pw.harvest(ss.resample_poly(x[:seg + N.HOP], 1, 3), 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
    f0 = np.pad(f0, (0, max(0, Tn - len(f0))))[:Tn]
    env = PA.envelope(xa, f0).astype(np.float64)
    return spk, x[:seg], f0, env


def fit() -> None:
    import train_ppgvoc as P
    import train_c1 as T1
    import c1_content as CC
    held = P.e2e_held()
    rows, _, _ = P.load_rows({h[0] for h in held}, True)
    rows = [r for r in rows if r[3] >= 9.0]
    random.Random(657).shuffle(rows)
    with Pool(12) as pl:
        got = [g for g in pl.map(fit_job, rows[:200]) if g is not None]
    dev = "cuda"
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"]); c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    Xs, Ys = [], []
    for spk, x, f0, env in got:
        T = np.load(ROOT / "data/rvoc_tables_real" / f"{spk}.npz")["T"].astype(np.float64)
        Tn = len(f0)
        xc = torch.from_numpy(np.concatenate([np.zeros(T1.PRE), x]).astype(np.float32))[None].to(dev)
        with torch.no_grad():
            p = torch.softmax(c1n(mfc(xc))[..., -Tn:], 1)[0].cpu().numpy().T.astype(np.float64)
        c0 = env[0]
        lv = c0 * DB
        sp = lv > lv.max() - 40
        r = rel_level(c0, sp)
        v = sp & (f0 > 0)
        d = env[1:25].T @ D24 - (p @ T.T) @ D24
        Xs.append(r[v]); Ys.append(d[v])
    r = np.concatenate(Xs)
    d = np.concatenate(Ys)
    X = np.stack([np.ones_like(r), r], 1)
    coef = np.linalg.lstsq(X, d, rcond=None)[0]
    a, b = coef[0], coef[1]
    a = a - a.mean()
    b = b - b.mean()
    import librosa
    fc = librosa.mel_frequencies(130, fmin=0.0, fmax=24000.0)[1:-1]
    hi = fc >= 4000
    rep = {"files": len(got), "frames": int(len(r)), "r_std_db": round(float(r.std()), 2),
           "a_rows": [float(q) for q in (a @ D24.T) / 10.0], "b_rows": [float(q) for q in (b @ D24.T) / 10.0],
           "a_hi_minus_lo_db": round(float(20 / np.log(10) * (a[hi].mean() - a[~hi].mean())), 3),
           "b_hi_minus_lo_db_per_10db": round(float(10 * 20 / np.log(10) * (b[hi].mean() - b[~hi].mean())), 3)}
    (OUT / "fit.json").write_text(json.dumps(rep, indent=1))
    print("fit", {k: v for k, v in rep.items() if not k.endswith("_rows")}, flush=True)


def rows_eft(s: str, k: int, key: str, with_b: bool) -> tuple:
    fj = json.loads((OUT / "fit.json").read_text())
    a, b = np.array(fj["a_rows"]), np.array(fj["b_rows"])
    pr = parts(s, k, key)
    rows, c0, f0 = pr["PR1"]
    lv = c0 * DB
    sp = lv > lv.max() - 40
    r = rel_level(c0, sp)
    v = (f0 > 0)[:, None]
    add = a[None] + (r[:, None] * b[None] if with_b else 0.0)
    return rows + v * add, c0, f0, pr["per"]


def render(it: tuple) -> int:
    s, k, key = it
    for a_, wb in (("EFT", True), ("EFT0", False)):
        o = H / s / "d570" / f"{a_}_{k}" / f"{key}_{a_}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            rows, c0, f0, per = rows_eft(s, k, key, wb)
            sf.write(o, synth(f0, env_full(c0, rows), ap_from_per(per, f0)), 48000, subtype="PCM_16")
    return 1


def knob(key: str) -> dict:
    rows, c0, f0, per = rows_eft("dev2", 7, key, True)
    ap = ap_from_per(per, f0)
    ys = {a: synth(f0, env_full(c0, rows, a), ap) for a in (1.0, 0.9, 1.1)}
    sp = {a: analyze(y.astype(np.float64))[1] for a, y in ys.items()}
    return {"key": key, **{f"render_{a}": env_shift(sp[1.0], sp[a], f0 > 0) for a in (0.9, 1.1)}}


def dev_job(it: tuple) -> dict:
    s, k, key = it
    from d6_unit_cov import load48
    out = {"k": k}
    for a in ("PR1", "EFT0", "EFT"):
        x, _ = sf.read(H / s / "d570" / f"{a}_{k}" / f"{key}_{a}_{k}.wav")
        out[a] = hf_tilt(np.asarray(x, np.float64))
    out["REF"] = hf_tilt(best8(load48(H / s / "d37" / "_work" / f"ref_{int(key[7:10]):03d}.wav").astype(np.float64)))
    return out


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["fit", "render", "knob", "rough", "dev"])
    a = ap_.parse_args()
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    npr = int(os.environ.get("LVC_NPROC", "12"))
    if a.stage == "fit":
        fit()
        return 0
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
            rep[k] = {x: {"d_hf": round(float(np.median([r[x][0] - r["REF"][0] for r in rk])), 2), "d_tilt": round(float(np.median([r[x][1] - r["REF"][1] for r in rk])), 3)} for x in ("PR1", "EFT0", "EFT")}
            print("target-relative", k, json.dumps(rep[k]), flush=True)
        (OUT / "target_rel.json").write_text(json.dumps(rep, indent=1))
        return 0
    sysn = lambda k: {"G": G[k][0], "PR1": f"PR1_{k}", "EFT0": f"EFT0_{k}", "EFT": f"EFT_{k}"}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d653/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep = {}
    for arm in ("PR1", "EFT0", "EFT"):
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
