"""D659: 表の引きのパワーの較正の曲線。学習の側の話者(開発の組に依らない)で、本人の学習の表(rvoc_tables_real)を本人の事後(C1)で引いた形と実の行を、
描画と同じ一本の物差し(c0 を対数の平均・正規化なし・発声の区間の長時間のパワー)で比べた帯ごとの比 → 1/6 oct で平滑 → 固定の曲線。PR2 = PR1(D653)× 曲線(α の後・絶対の周波数)。
規則は results/conv_p0/d659/prereg.yaml。
    uv run python d659_table_cal.py --stage cal | check | render | knob | rough | dev
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

from d631_world_swap import C, FEATS, G, H, ROOT, SETS, analyze, keys, measure
from d634_artic_render import DCT, FC, FQ, synth
from d638_artic_gate_g import env_shift
from d639_knob_hf import ap_from_per
from d651_copy_hf import best8, hf_tilt
from d653_causal_render import env_full, parts
from d655_deficit import env_stats
from d657_effort_tilt import fit_job

OUT = ROOT / "results/conv_p0/d659"
D24 = DCT[1:25]


def raw_env(c0: np.ndarray, rows: np.ndarray) -> np.ndarray:
    lm = c0[:, None] * DCT[0][None] + (rows * 10.0) @ D24
    return np.exp(2.0 * np.stack([np.interp(FQ, FC, r) for r in lm]))


def act_mask(e: np.ndarray) -> np.ndarray:
    lv = 10 * np.log10(e.sum(1) + 1e-30)
    return lv > lv.max() - 40


def c1_post(x: np.ndarray, Tn: int) -> np.ndarray:
    import train_c1 as T1
    import c1_content as CC
    if "net" not in C1C:
        st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
        n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to("cuda")
        n.load_state_dict(st1["net"]); n.eval().requires_grad_(False)
        C1C.update(net=n, mfc=CC.MelFront().to("cuda"))
    xc = torch.from_numpy(np.concatenate([np.zeros(T1.PRE), x]).astype(np.float32))[None].to("cuda")
    with torch.no_grad():
        return torch.softmax(C1C["net"](C1C["mfc"](xc))[..., -Tn:], 1)[0].cpu().numpy().T.astype(np.float64)


C1C: dict = {}


def cal() -> None:
    import train_ppgvoc as P
    held = P.e2e_held()
    rows, _, _ = P.load_rows({h[0] for h in held}, True)
    rows = [r for r in rows if r[3] >= 9.0]
    random.Random(659).shuffle(rows)
    with Pool(12) as pl:
        got = [g for g in pl.map(fit_job, rows[:200]) if g is not None]
    pr, po = np.zeros(len(FQ)), np.zeros(len(FQ))
    for spk, x, f0, env in got:
        T = np.load(ROOT / "data/rvoc_tables_real" / f"{spk}.npz")["T"].astype(np.float64)
        p = c1_post(x, len(f0))
        er = raw_env(env[0], env[1:25].T / 10.0)
        eo = raw_env(env[0], (p @ T.T) / 10.0)
        m = act_mask(er)
        sr, so = er[m].sum(0), eo[m].sum(0)
        pr += sr / sr.sum()
        po += so / sr.sum()
    db = 10 * np.log10((pr + 1e-30) / (po + 1e-30))
    lf = np.log2(np.maximum(FQ, 20.0))
    sm = np.array([db[np.abs(lf - lf[i]) <= 1 / 12].mean() for i in range(len(FQ))])
    sm = np.clip(sm - np.interp(500.0, FQ, sm), -6.0, 12.0)
    sm[FQ < 60] = sm[np.argmax(FQ >= 60)]
    (OUT / "curve2.json").write_text(json.dumps({"files": len(got), "db": [float(v) for v in sm]}))
    print("cal2 files", len(got), "curve2 dB", {f: round(float(np.interp(f, FQ, sm)), 2) for f in (250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 14000)}, flush=True)


def curve2() -> np.ndarray:
    return 10.0 ** (np.array(json.loads((OUT / "curve2.json").read_text())["db"]) / 10.0)


def check() -> None:
    import nvoc as N
    import pae as PA
    import pyworld as pw
    import scipy.signal as ss
    from d6_unit_cov import load48
    cv = curve2()
    out = []
    for s in SETS:
        for ti in sorted({int(k[7:10]) for k in keys(s)}):
            x = best8(load48(H / s / "d37" / "_work" / f"ref_{ti:03d}.wav").astype(np.float64))
            Tn = len(x) // N.HOP - 2
            seg = Tn * N.HOP
            xa = np.concatenate([np.zeros(PA.A), x[:seg], np.zeros(PA.A)])
            f0, _ = pw.harvest(ss.resample_poly(x[:seg + N.HOP], 1, 3), 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
            f0 = np.pad(f0, (0, max(0, Tn - len(f0))))[:Tn]
            env = PA.envelope(xa, f0).astype(np.float64)
            T = np.fromfile(H / s / "d85_work" / f"en_{ti:03d}" / "enroll.bin", "<f4").reshape(24, -1).astype(np.float64)
            p = c1_post(x[:seg], Tn)
            r = env_stats(env_full(env[0], env[1:25].T / 10.0))
            o = env_stats(env_full(env[0], (p @ T.T) / 10.0))
            oc = env_stats(env_full(env[0], (p @ T.T) / 10.0) * cv)
            out.append((o[0] - r[0], oc[0] - r[0], o[1] - r[1], oc[1] - r[1]))
    o = np.median(np.array(out), 0)
    rep = {"dev_refs_rust_table": {"OWN_minus_REAL_hf": round(float(o[0]), 2), "OWNC_minus_REAL_hf": round(float(o[1]), 2), "OWN_minus_REAL_tilt": round(float(o[2]), 3), "OWNC_minus_REAL_tilt": round(float(o[3]), 3)}}
    (OUT / "check.json").write_text(json.dumps(rep, indent=1))
    print("check", rep, flush=True)


def render(it: tuple) -> int:
    s, k, key = it
    o = H / s / "d570" / f"PR2_{k}" / f"{key}_PR2_{k}.wav"
    o.parent.mkdir(parents=True, exist_ok=True)
    if not o.exists():
        pr = parts(s, k, key)
        rows, c0, f0 = pr["PR1"]
        sf.write(o, synth(f0, env_full(c0, rows) * curve2(), ap_from_per(pr["per"], f0)), 48000, subtype="PCM_16")
    return 1


def knob(key: str) -> dict:
    pr = parts("dev2", 7, key)
    rows, c0, f0 = pr["PR1"]
    ap = ap_from_per(pr["per"], f0)
    cv = curve2()
    ys = {a: synth(f0, env_full(c0, rows, a) * cv, ap) for a in (1.0, 0.9, 1.1)}
    sp = {a: analyze(y.astype(np.float64))[1] for a, y in ys.items()}
    return {"key": key, **{f"render_{a}": env_shift(sp[1.0], sp[a], f0 > 0) for a in (0.9, 1.1)}}


def dev_job(it: tuple) -> dict:
    s, k, key = it
    from d6_unit_cov import load48
    out = {"k": k}
    for a in ("PR1", "PR2"):
        x, _ = sf.read(H / s / "d570" / f"{a}_{k}" / f"{key}_{a}_{k}.wav")
        out[a] = hf_tilt(np.asarray(x, np.float64))
    out["REF"] = hf_tilt(best8(load48(H / s / "d37" / "_work" / f"ref_{int(key[7:10]):03d}.wav").astype(np.float64)))
    return out


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["cal", "check", "render", "knob", "rough", "dev"])
    a = ap_.parse_args()
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    npr = int(os.environ.get("LVC_NPROC", "12"))
    if a.stage == "cal":
        cal()
        return 0
    if a.stage == "check":
        check()
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
            rep[k] = {x: {"d_hf": round(float(np.median([r[x][0] - r["REF"][0] for r in rk])), 2), "d_tilt": round(float(np.median([r[x][1] - r["REF"][1] for r in rk])), 3)} for x in ("PR1", "PR2")}
            print("target-relative", k, json.dumps(rep[k]), flush=True)
        (OUT / "target_rel.json").write_text(json.dumps(rep, indent=1))
        return 0
    sysn = lambda k: {"G": G[k][0], "PR1": f"PR1_{k}", "PR2": f"PR2_{k}"}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d653/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep = {}
    for arm in ("PR1", "PR2"):
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
