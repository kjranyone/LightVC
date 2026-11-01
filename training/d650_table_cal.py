"""D650: 目標の表の登録の時の較正(目標ごとの定数)。表 = 単位ごとの対数の平均の形 → 予測しにくい高域の揺れの分だけパワーの長時間の形が暗く出る。
目標の参照の録音で、本人の実の包絡の行(CheapTrick)と表の引き(Σ p T / 10・p = 参照の C1)の有声のフレームの形のパワーの平均(フレームのパワーで重み)の比 → 帯ごとの補正 → 行の補正 24(目標ごとの定数)。
EN3HC = EN3H(D649)の有声のフレームの行 + 補正・TABC = 表の引き + 遅い揺らぎ + R の ΔS + 補正(記録)。規則は results/conv_p0/d650/prereg.yaml。
    uv run python d650_table_cal.py --stage cal | render | rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import torch

import d637_envnet_eval as E7
from d631_world_swap import C, FEATS, G, H, ROOT, SETS, keys, measure
from d634_artic_render import DCT, FQ, PRE, env_from_rows, synth, vdir
from d635_split import norm
from d636_traj_gain import onepole
from d648_envnet3_eval import base

OUT = ROOT / "results/conv_p0/d650"
E7.NET = ROOT / "results/diag_envnet3/last.pt"
D24 = DCT[1:25]
HI, LO = (FQ >= 4000) & (FQ < 10000), (FQ >= 100) & (FQ < 4000)


def cal() -> dict:
    import nvoc as N
    import pae as PA
    import pyworld as pw
    import train_c1 as T1
    import c1_content as CC
    from d6_unit_cov import load48
    dev = "cuda"
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"]); c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    rep: dict = {}
    for s in SETS:
        tis = sorted({int(k[7:10]) for k in keys(s)})
        for ti in tis:
            x = load48(H / s / "d37" / "_work" / f"ref_{ti:03d}.wav").astype(np.float64)[: 30 * N.SR]
            T = len(x) // N.HOP - 2
            seg = T * N.HOP
            xa = np.concatenate([np.zeros(PA.A), x[:seg], np.zeros(PA.A)])
            import scipy.signal as ss
            f0, _ = pw.harvest(ss.resample_poly(x[:seg + N.HOP], 1, 3), 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
            f0 = np.pad(f0, (0, max(0, T - len(f0))))[:T]
            env = PA.envelope(xa, f0)
            tt = (PA.A + np.arange(T) * N.HOP) / N.SR
            sp = pw.cheaptrick(xa, f0, tt, N.SR, fft_size=PA.NFFT)
            xc = torch.from_numpy(np.concatenate([np.zeros(T1.PRE), x[:seg]]).astype(np.float32))[None].to(dev)
            with torch.no_grad():
                p = torch.softmax(c1n(mfc(xc))[..., -T:], 1)[0].cpu().numpy()
            tb = np.fromfile(H / s / "d85_work" / f"en_{ti:03d}" / "enroll.bin", "<f4").reshape(24, -1)
            rows_t = (tb @ p).T / 10.0
            rows_r = env[1:25].T / 10.0
            v = f0 > 0
            w = sp.mean(1)[v]
            pw_r = (w[:, None] * np.exp(2 * (rows_r[v] * 10) @ D24)).sum(0)
            pw_t = (w[:, None] * np.exp(2 * (rows_t[v] * 10) @ D24)).sum(0)
            off = 0.5 * np.log(pw_r / pw_t)
            off -= off.mean()
            import librosa
            fc = librosa.mel_frequencies(130, fmin=0.0, fmax=24000.0)[1:-1]
            hi = fc >= 4000
            rep[f"{s}|{ti}"] = {"drows": [float(q) for q in (off @ D24.T) / 10.0], "hi_minus_lo_db": round(float(20 / np.log(10) * (off[hi].mean() - off[~hi].mean())), 3), "voiced_frames": int(v.sum())}
    (OUT / "cal.json").write_text(json.dumps(rep, indent=1))
    hl = [r["hi_minus_lo_db"] for r in rep.values()]
    print("cal targets", len(rep), "hi − lo dB(実 − 表)median", round(float(np.median(hl)), 2), "p10 / p90", round(float(np.percentile(hl, 10)), 2), round(float(np.percentile(hl, 90)), 2), flush=True)
    return rep


def render(it: tuple) -> dict:
    s, k, key = it
    torch.set_num_threads(1)
    cj = json.loads((OUT / "cal.json").read_text())
    dr = np.array(cj[f"{s}|{int(key[7:10])}"]["drows"])
    _, f0, pin, ap = base(s, k, key)
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float32)
    vc = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T))[None]
    with torch.no_grad():
        y = E7.net()(t(rc[:, 31:231]), t(rc[:, 0:1]), t(rc[:, 25:26]), t(rc[:, 26:27]), t(rc[:, 1:25]), t(rc[:, 231:295]))[0].numpy().T.astype(np.float64)
    tab = rc[:, 1:25].astype(np.float64)
    post = (vc[:, 1:25] - rc[:, 295:]).astype(np.float64)
    d = y - tab
    sl = slice(PRE, PRE + 1600)
    vo = (f0 > 0)[:, None]
    rows = {"EN3HC": (tab + d - onepole(d, float(np.exp(-0.005 / 2.0))) + post)[sl] + vo * dr, "TABC": (tab + post)[sl] + vo * dr}
    m = (rc[sl, 0] > 0.05) & (f0 > 0)
    rec = {"key": f"{s}|{k}|{key}"}
    for a, r in rows.items():
        e = norm(env_from_rows(r), pin)
        rec[a] = float(10 * np.log10(e[m][:, HI].sum() / e[m][:, LO].sum()))
        o = H / s / "d570" / f"{a}_{k}" / f"{key}_{a}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            sf.write(o, synth(f0, e, ap), 48000, subtype="PCM_16")
    return rec


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["cal", "render", "rough"])
    a = ap_.parse_args()
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    npr = int(os.environ.get("LVC_NPROC", "12"))
    if a.stage == "cal":
        cal()
        return 0
    if a.stage == "render":
        with Pool(npr) as pl:
            recs = pl.map(render, its, chunksize=2)
        (OUT / "render_rec.json").write_text(json.dumps(recs, indent=0))
        for k in C:
            rk = [r for r in recs if r["key"].split("|")[1] == str(k)]
            print("voiced hi/lo dB", k, {x: round(float(np.median([r[x] for r in rk])), 2) for x in ("EN3HC", "TABC")}, flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0], "EN3H": f"EN3H_{k}", "EN3HC": f"EN3HC_{k}"}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text()) | json.loads((ROOT / "results/conv_p0/d649/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("EN3H", "EN3HC"):
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
