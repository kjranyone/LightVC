"""D652: 構音の描画の定数の較正。学習の側の実女声の録音(開発の組に依らない)で、写し合成 W24P(D651 と同じ道)と元の録音の長時間のパワーの比を周波数ごとに測り(1/6 oct で平滑)、
描画の包絡に絶対の周波数で掛ける(α の後 = つまみに中立)。EN3HR = EN3H(D649)+ 較正。規則は results/conv_p0/d652/prereg.yaml。
    uv run python d652_render_cal.py --stage cal | render | knob | rough
"""
from __future__ import annotations

import argparse
import json
import os
import random
from multiprocessing import Pool

import numpy as np
import pyworld as pw
import scipy.signal as ss
import soundfile as sf
import torch

import d637_envnet_eval as E7
import nvoc as N
import pae as PA
from d631_world_swap import C, FEATS, G, H, ROOT, SETS, analyze, keys, measure
from d634_artic_render import FQ, PRE, env_from_rows, synth, vdir
from d635_split import norm
from d636_traj_gain import onepole
from d638_artic_gate_g import env_shift
from d639_knob_hf import ap_from_per
from d648_envnet3_eval import base

OUT = ROOT / "results/conv_p0/d652"
E7.NET = ROOT / "results/diag_envnet3/last.pt"


def spec_act(x: np.ndarray) -> np.ndarray:
    x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
    n = len(x) // N.HOP
    e = 10 * np.log10((x[: n * N.HOP].reshape(n, N.HOP) ** 2).mean(1) + 1e-12)
    act = e > e.max() - 40
    _, _, Z = ss.stft(x, N.SR, nperseg=2048, noverlap=2048 - N.HOP)
    return (np.abs(Z[:, : len(act)]) ** 2)[:, act[: Z.shape[1]]].mean(1)


def cal_job(row: tuple) -> tuple:
    import train_rvoc as TR
    torch.set_num_threads(1)
    wav, _, sr, dur = row[:4]
    x = TR.read_span(wav, sr, int(0.5 * N.SR), 8 * N.SR).astype(np.float64)
    if len(x) < 8 * N.SR or np.sqrt((x ** 2).mean()) < 1e-3:
        return None
    f0, t = pw.harvest(ss.resample_poly(x, 1, 3), 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, N.SR)
    T = len(f0)
    xa = np.concatenate([np.zeros(PA.A), x, np.zeros(PA.A + N.HOP * 2)])
    env = PA.envelope(xa, f0)
    per = PA.periodicity(torch.from_numpy(xa)[None].float(), torch.from_numpy(f0)[None].float())[0].numpy().T.astype(np.float64)
    e24 = norm(env_from_rows(env[1:25].T.astype(np.float64) / 10.0), sp.mean(1, keepdims=True))
    y = pw.synthesize(f0, e24, ap_from_per(per[:T], f0), N.SR, 5.0)
    return spec_act(x), spec_act(np.asarray(y))


def cal() -> None:
    import train_ppgvoc as P
    held = P.e2e_held()
    rows, _, _ = P.load_rows({h[0] for h in held}, True)
    rows = [r for r in rows if r[3] >= 9.0]
    random.Random(652).shuffle(rows)
    with Pool(12) as pl:
        res = [r for r in pl.map(cal_job, rows[:160]) if r is not None]
    po, pw_ = np.sum([r[0] for r in res], 0), np.sum([r[1] for r in res], 0)
    db = 10 * np.log10((po + 1e-20) / (pw_ + 1e-20))
    lf = np.log2(np.maximum(FQ, 20.0))
    sm = np.array([db[np.abs(lf - lf[i]) <= 1 / 12].mean() for i in range(len(FQ))])
    sm = np.clip(sm, -6.0, 12.0)
    sm[FQ < 60] = sm[np.argmax(FQ >= 60)]
    (OUT / "curve.json").write_text(json.dumps({"files": len(res), "db": [float(v) for v in sm]}))
    pick = {f: round(float(np.interp(f, FQ, sm)), 2) for f in (250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 14000)}
    print("cal files", len(res), "curve dB at Hz", pick, flush=True)


def curve() -> np.ndarray:
    return 10.0 ** (np.array(json.loads((OUT / "curve.json").read_text())["db"]) / 10.0)


def rows_en3h(s: str, k: int, key: str) -> np.ndarray:
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float32)
    vc = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T))[None]
    with torch.no_grad():
        y = E7.net()(t(rc[:, 31:231]), t(rc[:, 0:1]), t(rc[:, 25:26]), t(rc[:, 26:27]), t(rc[:, 1:25]), t(rc[:, 231:295]))[0].numpy().T.astype(np.float64)
    tab = rc[:, 1:25].astype(np.float64)
    d = y - tab
    return (tab + d - onepole(d, float(np.exp(-0.005 / 2.0))) + (vc[:, 1:25] - rc[:, 295:]))[PRE:PRE + 1600]


def render(it: tuple) -> int:
    s, k, key = it
    torch.set_num_threads(1)
    o = H / s / "d570" / f"EN3HR_{k}" / f"{key}_EN3HR_{k}.wav"
    o.parent.mkdir(parents=True, exist_ok=True)
    if not o.exists():
        _, f0, pin, ap = base(s, k, key)
        sf.write(o, synth(f0, norm(env_from_rows(rows_en3h(s, k, key)) * curve(), pin), ap), 48000, subtype="PCM_16")
    return 1


def knob(key: str) -> dict:
    torch.set_num_threads(1)
    _, f0, pin, ap = base("dev2", 7, key)
    r = rows_en3h("dev2", 7, key)
    cv = curve()
    ys = {a: synth(f0, norm(env_from_rows(r, a) * cv, pin), ap) for a in (1.0, 0.9, 1.1)}
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
    sysn = lambda k: {"G": G[k][0], "EN3H": f"EN3H_{k}", "EN3HR": f"EN3HR_{k}"}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d649/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("EN3H", "EN3HR"):
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
