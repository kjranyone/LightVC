"""D641: 内容の網の対数の回帰の偏り(条件付き中央値 → パワーの平均を低く見積もる)を、学習の分布で測った帯ごとの固定の補正で直す(ENPB = ENP + 補正)。開発の組 dev2 / dev3。
--stage bias: 学習と同じ前処理の 40 バッチで、有声のフレームの包絡の形(c0 なし)のパワーの平均の比 → 帯ごとの補正 → 行の補正 24。表の引きの偏りも記録。
--stage render / knob / rough。規則は results/conv_p0/d641/prereg.yaml。
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
from d634_artic_render import DCT, PRE, env_from_rows, synth, vdir
from d635_split import norm
from d637_envnet_eval import env_rows
from d638_artic_gate_g import env_shift
from d639_knob_hf import ap_from_per

OUT = ROOT / "results/conv_p0/d641"
D24 = DCT[1:25]


def bias() -> dict:
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_ppgvoc as P
    import train_rvoc as TR
    from train_envnet import WARM, EnvNet
    import c1_content as CC
    dev = "cuda"
    torch.manual_seed(641)
    TR.SEG = 33600 * 4
    TR.T = TR.SEG // N.HOP
    SEG, T = TR.SEG, TR.T
    held = P.e2e_held()
    rows, spks, _ = P.load_rows({h[0] for h in held}, True)
    spk_idx = {s: i for i, s in enumerate(spks)}
    tabT = torch.from_numpy(np.stack([np.load(ROOT / "data/rvoc_tables_real" / f"{s}.npz")["T"] for s in spks])).float().to(dev)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"]); c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    P.cvin_setup(str(ROOT / "results/c2_1/last.pt"), dev)
    sm = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    front = TR.Front("pae", TR.ckpt_env_smooth(sm, None)).to(dev)
    st = torch.load(ROOT / "results/diag_envnet1/last.pt", map_location="cpu", weights_only=False)
    net = EnvNet(st["cfg"]["ch"], tuple(st["cfg"]["dils"]), st["cfg"]["d_c2"]).to(dev).eval()
    net.load_state_dict(st["net"])
    dl = iter(torch.utils.data.DataLoader(TR.DS(rows, rows, 641, spk_idx=spk_idx, ctx=T1.PRE, sexpert=0.5), batch_size=8, num_workers=8))
    D = torch.from_numpy(D24).float().to(dev)
    acc = {x: torch.zeros(128, device=dev, dtype=torch.float64) for x in ("t", "p", "b")}
    nfr = 0
    for _ in range(40):
        bt = [t.to(dev) for t in next(dl)]
        xa, f0, env, xl, si = bt[:5]
        xp = bt[5] if len(bt) > 5 else xa
        env_o = bt[6] if len(bt) > 6 else env
        with torch.no_grad():
            ct = front(xp, f0, env)
            ct_o = ct if env_o is env else front(xa, f0, env_o)
            xc = torch.cat([xl, xp[:, PA.A:PA.A + SEG]], 1)
            p = P.ppg_form(c1n(mfc(xc))[..., -T:], "soft")
            c2 = P.cvin_frames(xc, T)
            tab = torch.einsum("bdk,bkt->bdt", tabT[si], p) / 10
            y = net(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], tab, c2)
            m = (ct_o[:, 26, WARM:] > 0.5)
            for x, r in (("t", ct_o[:, 1:25]), ("p", y), ("b", tab)):
                lm = torch.einsum("bdt,dm->btm", r[..., WARM:] * 10, D)
                acc[x] += torch.exp(2 * lm[m].double()).sum(0)
            nfr += int(m.sum())
    lp = {x: torch.log(v / nfr).cpu().numpy() for x, v in acc.items()}
    off = 0.5 * (lp["t"] - lp["p"])
    off -= off.mean()
    offb = 0.5 * (lp["t"] - lp["b"])
    offb -= offb.mean()
    import librosa
    fc = librosa.mel_frequencies(130, fmin=0.0, fmax=24000.0)[1:-1]
    hi = fc >= 4000
    rep = {"frames": nfr, "drows": [float(v) for v in (off @ D24.T) / 10.0], "off_band_net": [float(v) for v in off], "off_band_tab": [float(v) for v in offb],
           "net_hi_minus_lo_db": round(float(20 / np.log(10) * (off[hi].mean() - off[~hi].mean())), 2), "tab_hi_minus_lo_db": round(float(20 / np.log(10) * (offb[hi].mean() - offb[~hi].mean())), 2)}
    (OUT / "bias.json").write_text(json.dumps(rep, indent=1))
    print("bias", {k: v for k, v in rep.items() if k in ("frames", "net_hi_minus_lo_db", "tab_hi_minus_lo_db")}, "drows c1..c6", np.round(rep["drows"][:6], 4).tolist(), flush=True)
    return rep


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
    rc = np.fromfile(H / s / "d637_rcond" / f"k{k}" / f"{key}.rcond16", "<f2").reshape(-1, 319)[PRE:PRE + 1600].astype(np.float64)
    ap = ap_from_per(rc[:, 27:31], f0)
    dr = np.array(json.loads((OUT / "bias.json").read_text())["drows"])
    rows = env_rows(s, k, key)["ENR"].astype(np.float64) + dr
    return rows, f0, pin, ap


def render(it: tuple) -> int:
    s, k, key = it
    o = H / s / "d570" / f"ENPB_{k}" / f"{key}_ENPB_{k}.wav"
    o.parent.mkdir(parents=True, exist_ok=True)
    if not o.exists():
        rows, f0, pin, ap = base(s, k, key)
        sf.write(o, synth(f0, norm(env_from_rows(rows), pin), ap), 48000, subtype="PCM_16")
    return 1


def knob(key: str) -> dict:
    rows, f0, pin, ap = base("dev2", 7, key)
    vo = f0 > 0
    ys = {a: synth(f0, norm(env_from_rows(rows, a), pin), ap) for a in (1.0, 0.9, 1.1)}
    sp = {a: analyze(y.astype(np.float64))[1] for a, y in ys.items()}
    return {"key": key, **{f"render_{a}": env_shift(sp[1.0], sp[a], vo) for a in (0.9, 1.1)}, "alpha1_identity": bool(np.array_equal(env_from_rows(rows, 1.0), env_from_rows(rows)))}


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["bias", "render", "knob", "rough"])
    a = ap_.parse_args()
    npr = int(os.environ.get("LVC_NPROC", "12"))
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    if a.stage == "bias":
        bias()
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
        rep = {"items": recs, "median_render": med, "cals_render": cals, "alpha1_identity": all(r["alpha1_identity"] for r in recs)}
        rep["knob_ok_cals"] = bool(rep["alpha1_identity"] and all(abs(med[x] - cals[str(x)]) <= 0.015 for x in (0.9, 1.1)))
        (OUT / "knob.json").write_text(json.dumps(rep, indent=1))
        print("knob", {k_: v_ for k_, v_ in rep.items() if k_ != "items"}, flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0], "ENP": f"ENP_{k}", "ENPB": f"ENPB_{k}"}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d639/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("ENP", "ENPB"):
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
