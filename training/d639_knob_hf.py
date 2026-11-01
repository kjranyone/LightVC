"""D639: (A) 声道長のつまみの測り方の較正 = 錨と同じ作りの包絡(入力の CheapTrick を α_G で伸ばしたもの)をさらに 0.9 / 1.1 倍に伸縮して同じ合成・同じ測りを当てる(dev2 k7 の 12 本)。
(B) 構音の描画の高域不足(関門 G のこもり・hf)の切り分け: ENC = 内容の網の c0n の入力を学習の発声の中央値へずらす(定数)・ENP = 非周期性を条件の 4 帯の周期性(製品の因果な源)から作る。
規則は results/conv_p0/d639/prereg.yaml。
    uv run python d639_knob_hf.py --stage c0stat
    uv run python d639_knob_hf.py --stage knob
    uv run python d639_knob_hf.py --stage render
    uv run python d639_knob_hf.py --stage rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import torch

from d631_world_swap import C, FEATS, G, H, ROOT, SETS, analyze, keys, measure
from d632_f0env_swap import fit, lag, shift, warp
from d634_artic_render import FQ, PRE, env_from_rows, synth, vdir
from d635_split import norm
from d637_envnet_eval import DC, net, rdir
from d638_artic_gate_g import env_shift

OUT = ROOT / "results/conv_p0/d639"
ARMS = ("ENC", "ENP")
HBC = np.array([np.sqrt(300 * 1000), np.sqrt(1000 * 2500), np.sqrt(2500 * 5000), np.sqrt(5000 * 10000)])


def c0stat() -> dict:
    import train_envnet as TE
    import train_ppgvoc as P
    import train_rvoc as TR
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    TR.SEG = 33600 * 4
    TR.T = TR.SEG // N.HOP
    held = P.e2e_held()
    rows, spks, _ = P.load_rows({h[0] for h in held}, True)
    spk_idx = {s: i for i, s in enumerate(spks)}
    sm = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    front = TR.Front("pae", TR.ckpt_env_smooth(sm, None)).to("cuda")
    dl = iter(torch.utils.data.DataLoader(TR.DS(rows, rows, 639, spk_idx=spk_idx, ctx=T1.PRE, sexpert=0.5), batch_size=8, num_workers=8))
    vals = []
    for _ in range(30):
        bt = [t.to("cuda") for t in next(dl)]
        xa, f0, env = bt[:3]
        xp = bt[5] if len(bt) > 5 else xa
        with torch.no_grad():
            ct = front(xp, f0, env)
        c = ct[:, 0]
        vals.append(c[c > 0.05].cpu().numpy())
    tr = float(np.median(np.concatenate(vals)))
    inf = []
    for s in SETS:
        for k in C:
            for key in keys(s):
                rc = np.fromfile(rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, DC + 24)[PRE:PRE + 1600, 0].astype(np.float32)
                inf.append(rc[rc > 0.05])
    iv = float(np.median(np.concatenate(inf)))
    rep = {"train_speech_c0n_median": round(tr, 4), "infer_speech_c0n_median": round(iv, 4), "delta": round(tr - iv, 4), "train_batches": 30}
    (OUT / "c0stat.json").write_text(json.dumps(rep, indent=1))
    print("c0stat", rep, flush=True)
    return rep


def knob(key: str) -> dict:
    s, k = "dev2", 7
    from d6_unit_cov import load48
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    f0s, sps, aps = analyze(xs)
    spg, apg = warp(sps, G[k][1]), warp(aps, G[k][1])
    f0g = f0s * 2 ** (k / 12)
    vo = f0g > 0
    ys = {a: synth(f0g, spg if a == 1.0 else warp(spg, a), apg) for a in (1.0, 0.9, 1.1)}
    sp = {a: analyze(y.astype(np.float64))[1] for a, y in ys.items()}
    rec = {"key": key}
    for a in (0.9, 1.1):
        rec[f"render_{a}"] = env_shift(sp[1.0], sp[a], vo)
        rec[f"env_{a}"] = env_shift(spg, warp(spg, a), vo)
    return rec


def ap_from_per(per: np.ndarray, f0: np.ndarray) -> np.ndarray:
    hnr = 30.0 * np.clip(per, 0.0, 1.5)
    a2 = 1.0 / (1.0 + 10.0 ** (hnr / 10.0))
    lf, lc = np.log(np.maximum(FQ, 1.0)), np.log(HBC)
    ap = np.sqrt(np.stack([np.interp(lf, lc, r) for r in a2]))
    ap[f0 <= 0] = 1.0
    return np.clip(ap, 1e-3, 1.0)


def rows_c0(s: str, k: int, key: str, delta: float) -> tuple:
    rc = np.fromfile(rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, DC + 24).astype(np.float32)
    vc = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T))[None]
    c0 = np.where(rc[:, 0:1] > 0.05, rc[:, 0:1] + delta, rc[:, 0:1]).astype(np.float32)
    with torch.no_grad():
        y = net()(t(rc[:, 31:231]), t(c0), t(rc[:, 25:26]), t(rc[:, 26:27]), t(rc[:, 1:25]), t(rc[:, 231:DC]))[0].numpy().T
        y0 = net()(t(rc[:, 31:231]), t(rc[:, 0:1]), t(rc[:, 25:26]), t(rc[:, 26:27]), t(rc[:, 1:25]), t(rc[:, 231:DC]))[0].numpy().T
    post = vc[:, 1:25] - rc[:, DC:]
    sl = slice(PRE, PRE + 1600)
    return (y + post)[sl], (y0 + post)[sl], rc[sl, 27:31]


def render(it: tuple) -> dict:
    s, k, key = it
    delta = json.loads((OUT / "c0stat.json").read_text())["delta"]
    d = H / s / "d570"
    from d6_unit_cov import load48
    v = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)[PRE:PRE + 1600].astype(np.float64)
    xc = load48(d / C[k] / f"{key}_{C[k]}.wav").astype(np.float64)
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    lg = lag(xc, xs)
    _, sps, aps = (shift(m, lg) for m in analyze(xs))
    n = len(v)
    apg = fit(warp(aps, G[k][1]), n)
    pin = fit(sps, n).mean(1, keepdims=True)
    f0 = np.where(v[:, 26] > 0.5, 200.0 * np.exp(v[:, 25]), 0.0)
    rcc, renr, per = rows_c0(s, k, key, delta)
    app = ap_from_per(per.astype(np.float64), f0)
    rec = {"key": f"{s}|{k}|{key}", "ap_hi_med_D4C": float(np.median(apg[f0 > 0][:, FQ >= 5000])), "ap_hi_med_per": float(np.median(app[f0 > 0][:, FQ >= 5000])),
           "per_med": [float(x) for x in np.median(per[f0 > 0], 0)], "c1_ENC_minus_ENR": float(np.mean(rcc[f0 > 0, 0] - renr[f0 > 0, 0]))}
    for a, rows, ap in (("ENC", rcc, apg), ("ENP", renr, app)):
        o = d / f"{a}_{k}" / f"{key}_{a}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            sf.write(o, synth(f0, norm(env_from_rows(rows.astype(np.float64)), pin), ap), 48000, subtype="PCM_16")
    return rec


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["c0stat", "knob", "render", "rough"])
    a = ap_.parse_args()
    npr = int(os.environ.get("LVC_NPROC", "12"))
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    if a.stage == "c0stat":
        c0stat()
        return 0
    if a.stage == "knob":
        with Pool(12) as pl:
            recs = pl.map(knob, keys("dev2")[:12])
        enr = json.loads((ROOT / "results/conv_p0/d638/knob.json").read_text())
        rep = {"items": recs, "cal_median_render": {x: round(float(np.median([r[f"render_{x}"] for r in recs])), 3) for x in (0.9, 1.1)},
               "cal_median_env": {x: round(float(np.median([r[f"env_{x}"] for r in recs])), 3) for x in (0.9, 1.1)},
               "enr_median_render": enr["median_render"], "enr_median_env": enr["median_env"]}
        rep["knob_cal_ok"] = bool(all(abs(rep["enr_median_render"][str(x)] - rep["cal_median_render"][x]) <= 0.015 for x in (0.9, 1.1)))
        (OUT / "knob_cal.json").write_text(json.dumps(rep, indent=1))
        print("knob_cal", {k_: v_ for k_, v_ in rep.items() if k_ != "items"}, flush=True)
        return 0
    if a.stage == "render":
        with Pool(npr) as pl:
            recs = pl.map(render, its, chunksize=2)
        (OUT / "render_rec.json").write_text(json.dumps(recs, indent=0))
        print("rendered", len(recs), "ap(>5 kHz・有声)の中央値 D4C", round(float(np.median([r["ap_hi_med_D4C"] for r in recs])), 3), "周期性から", round(float(np.median([r["ap_hi_med_per"] for r in recs])), 3),
              "周期性の中央値", np.round(np.median([r["per_med"] for r in recs], 0), 3).tolist(), "c1 ENC − ENR", round(float(np.median([r["c1_ENC_minus_ENR"] for r in recs])), 4), flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0]} | {x: f"{x}_{k}" for x in ARMS + ("ENR",)}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d637/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("ENR",) + ARMS:
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
