"""D666: 目標ごとの表の引きのパワーの較正(登録の時の定数)と k5 の shimmer の切り分け。
目標の参照(登録と同じ ref_XXX.wav)で、実の行のパワー(c0 = 対数の平均・正規化なし)対 表の引き(本人の事後 × Rust の登録の表)を PR4 と同じ約束(形の平均パワーを 25 Hz でならす × exp(2·c0 / √128))で比べ、
周波数ごとの比 → 1/6 oct で平滑 → 500 Hz を 0 dB → [−6, 12] dB = 目標ごとの曲線 3(全体の曲線 2 の代わり)。PR5 = PR4C の曲線 2 を目標ごとの曲線 3 に・PR5F = PR5 + f0 の慣性 15 Hz(shimmer の切り分け)。
規則は results/conv_p0/d666/prereg.yaml。
    uv run python d666_target_cal.py --stage cal | render | knob | rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

from causal_synth import synth_causal
from d631_world_swap import C, FEATS, G, H, ROOT, SETS, analyze, keys, measure
from d634_artic_render import FQ
from d636_traj_gain import onepole
from d638_artic_gate_g import env_shift
from d639_knob_hf import ap_from_per
from d652_render_cal import curve as curve1
from d653_causal_render import parts
from d659_table_cal import act_mask, c1_post, raw_env
from d663_shape_power import env4

OUT = ROOT / "results/conv_p0/d666"
A15 = float(np.exp(-2 * np.pi * 15.0 * 0.005))


def cal() -> None:
    import nvoc as N
    import pae as PA
    import pyworld as pw
    import scipy.signal as ss
    from d6_unit_cov import load48
    rep = {}
    for s in SETS:
        for ti in sorted({int(k[7:10]) for k in keys(s)}):
            x = load48(H / s / "d37" / "_work" / f"ref_{ti:03d}.wav").astype(np.float64)[: 30 * N.SR]
            Tn = len(x) // N.HOP - 2
            seg = Tn * N.HOP
            xa = np.concatenate([np.zeros(PA.A), x[:seg], np.zeros(PA.A)])
            f0, _ = pw.harvest(ss.resample_poly(x[:seg + N.HOP], 1, 3), 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
            f0 = np.pad(f0, (0, max(0, Tn - len(f0))))[:Tn]
            env = PA.envelope(xa, f0).astype(np.float64)
            T = np.fromfile(H / s / "d85_work" / f"en_{ti:03d}" / "enroll.bin", "<f4").reshape(24, -1).astype(np.float64)
            p = c1_post(x[:seg], Tn)
            er = raw_env(env[0], env[1:25].T / 10.0)
            eo = env4(env[0], (p @ T.T) / 10.0)
            m = act_mask(er)
            pr, po = er[m].sum(0), eo[m].sum(0)
            db = 10 * np.log10((pr / pr.sum() + 1e-30) / (po / po.sum() + 1e-30))
            lf = np.log2(np.maximum(FQ, 20.0))
            sm = np.array([db[np.abs(lf - lf[i]) <= 1 / 12].mean() for i in range(len(FQ))])
            sm = np.clip(sm - np.interp(500.0, FQ, sm), -6.0, 12.0)
            sm[FQ < 60] = sm[np.argmax(FQ >= 60)]
            rep[f"{s}|{ti}"] = [float(v) for v in sm]
    (OUT / "curve3_by_target.json").write_text(json.dumps(rep))
    a = np.array(list(rep.values()))
    pick = lambda f: np.interp(f, FQ, np.median(a, 0))
    print("cal targets", len(rep), "median curve3 dB", {f: round(float(pick(f)), 2) for f in (500, 1000, 2000, 4000, 8000)}, "sd at 4 kHz", round(float(np.std([np.interp(4000, FQ, r) for r in a])), 2), flush=True)


def curve3(s: str, ti: int) -> np.ndarray:
    return 10.0 ** (np.array(json.loads((OUT / "curve3_by_target.json").read_text())[f"{s}|{ti}"]) / 10.0)


def render(it: tuple) -> int:
    s, k, key = it
    pr = parts(s, k, key)
    rows, c0, f0 = pr["PR1"]
    cv = curve1() * curve3(s, int(key[7:10]))
    e = env4(c0, rows) * cv
    for a, ff in (("PR5", f0), ("PR5F", None)):
        o = H / s / "d570" / f"{a}_{k}" / f"{key}_{a}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if o.exists():
            continue
        if ff is None:
            v = f0 > 0
            lf = np.where(v, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)
            y = lf.copy()
            for t in range(1, len(lf)):
                if v[t] and v[t - 1]:
                    y[t] = A15 * y[t - 1] + (1 - A15) * lf[t]
            ff = np.where(v, 200.0 * np.exp(y), 0.0)
        sf.write(o, synth_causal(ff, e, ap_from_per(pr["per"], ff)), 48000, subtype="PCM_16")
    return 1


def knob(key: str) -> dict:
    pr = parts("dev2", 7, key)
    rows, c0, f0 = pr["PR1"]
    ap = ap_from_per(pr["per"], f0)
    cv = curve1() * curve3("dev2", int(key[7:10]))
    ys = {a: synth_causal(f0, env4(c0, rows, a) * cv, ap) for a in (1.0, 0.9, 1.1)}
    sp = {a: analyze(y.astype(np.float64))[1] for a, y in ys.items()}
    return {"key": key, **{f"render_{a}": env_shift(sp[1.0], sp[a], f0 > 0) for a in (0.9, 1.1)}}


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["cal", "render", "knob", "rough"])
    a = ap_.parse_args()
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    npr = int(os.environ.get("LVC_NPROC", "12"))
    if a.stage == "cal":
        cal()
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
    sysn = lambda k: {"G": G[k][0], "PR4C": f"PR4C_{k}", "PR5": f"PR5_{k}", "PR5F": f"PR5F_{k}"}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d664/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep = {}
    for arm in ("PR4C", "PR5", "PR5F"):
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
