"""段 P79 step a(results/snv_p79/prereg.yaml・学習なし・ASR なし): 崩れる源(写しの読みの一致 R の低い源)は、録音の鎖の印と、発声・前段の印のどちらと並ぶか。
写しの R と CER は results/snv_p78/reread.json の行を当てる。R_fix = 源と写しの包絡を一定の f0(源の有声の f0 の中央値)で作り直した写しの R。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=4 uv run --no-sync python snv_diag/p79_sig.py [sys=snv_p30_c] [sets=cv,x:d_clean] [out=results/snv_p79/sig.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i13_bgm_dev_probe as I13
import p18_align as AL
import p78_reread as R78
import snv_p0_front as P
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV
SR = P.SR
CHAIN = ("floor_gap", "floor_spread", "floor_frac", "silence", "agc_slope", "crest", "clip", "hf4", "hf8", "lf100", "bw99", "holes")
PHON = ("f0_med", "low_f0", "oct_jump", "flips", "jitter", "per_mean", "per_hi", "vfrac")


def chain_sig(x: np.ndarray) -> dict:
    n, h = int(0.02 * SR), int(0.01 * SR)
    fr = librosa.util.frame(x, frame_length=n, hop_length=h)
    e = 10 * np.log10((fr ** 2).mean(0) + 1e-12)
    L = np.percentile(e, 95)
    fl, sp = e < L - 30, e >= L - 20
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=h, win_length=n, center=False)) ** 2
    m = min(S.shape[1], len(e))
    S, sp2 = S[:, :m], sp[:m]
    f = np.fft.rfftfreq(2048, 1 / SR)
    ps = S[:, sp2].sum(1) + 1e-20
    cum = np.cumsum(ps) / ps.sum()
    ed = np.geomspace(300, 8000, 65)
    B = 10 * np.log10(np.stack([S[(f >= a) & (f < b)].sum(0) for a, b in zip(ed[:-1], ed[1:])]) + 1e-14)[:, sp2]
    t = np.flatnonzero(sp) * h / SR
    r = lambda lo, hi: float(10 * np.log10(ps[(f >= lo) & (f < hi)].sum() / ps.sum() + 1e-12))
    return {"floor_gap": float(L - np.median(e[fl])) if fl.any() else 60.0, "floor_spread": float(np.percentile(e[fl], 90) - np.percentile(e[fl], 10)) if fl.sum() > 5 else 0.0,
            "floor_frac": float(fl.mean()), "silence": float((e < -90).mean()), "agc_slope": float(abs(np.polyfit(t, e[sp], 1)[0])) if sp.sum() > 5 else 0.0,
            "crest": float(20 * np.log10(np.abs(x).max() / (np.sqrt((fr[:, sp] ** 2).mean()) + 1e-9) + 1e-9)), "clip": float((np.abs(x) >= 0.985 * np.abs(x).max()).mean()),
            "hf4": r(4000, SR), "hf8": r(8000, SR), "lf100": r(0, 100), "bw99": float(f[np.searchsorted(cum, 0.99)]),
            "holes": float((B < np.median(B, 1, keepdims=True) - 25).mean())}


def phon_sig(u: dict) -> dict:
    f0, v = u["f0"].astype(np.float64), u["v"] > 0.5
    fv = f0[v]
    pair = v[1:] & v[:-1]
    d = np.abs(np.log2(np.maximum(f0[1:], 1.0) / np.maximum(f0[:-1], 1.0)))[pair]
    per = u["y"][25:29][:, v]
    return {"f0_med": float(np.median(fv)) if len(fv) else 0.0, "low_f0": float((fv < 80).mean()) if len(fv) else 0.0, "oct_jump": float((d > 0.6).mean()) if len(d) else 0.0,
            "flips": float(np.abs(np.diff(v.astype(np.int8))).sum() / (len(v) * P.H / SR)), "jitter": float(np.median(d) * 1200) if len(d) else 0.0,
            "per_mean": float(per.mean()) if per.size else 0.0, "per_hi": float(per[-1].mean()) if per.size else 0.0, "vfrac": float(v.mean())}


def refix(u: dict, x: np.ndarray, fc: float) -> dict:
    Tn = len(u["v"])
    env = P.envelope(x[: Tn * P.H].astype(np.float64), np.full(Tn, fc))
    y = u["y"].copy()
    y[:25] = T.to_y(env, np.zeros((4, Tn), np.float32), np.ones(Tn, np.float32))[:25]
    return u | {"y": y}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag, names = opt.get("sys", "snv_p30_c"), opt.get("sets", "cv,x:d_clean").split(",")
    out = ROOT / opt.get("out", "results/snv_p79/sig.json")
    old = {(r["set"], r["src"]): r for r in json.loads((ROOT / "results/snv_p78/reread.json").read_text())["rows"]}
    sy = EV.System(tag, EV.load_gen())
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    sets = {nm: [x for x in its if x["valid"] and (nm, x["src"]) in old][: int(opt.get("nmax", 10 ** 6))] for nm, its in AL.build_sets(names, 64, split).items()}
    rows = []
    for nm, its in sets.items():
        for it in its:
            o, enr, u = old[(nm, it["src"])], it["enr"][:3], it["u"]
            path = str(T.FEAT / f"cv/{it['src']}.wav") if nm == "cv" else it["path"]
            x = librosa.load(path, sr=SR, mono=True)[0].astype(np.float64)
            fv = u["f0"][u["v"] > 0.5]
            fc = float(np.median(fv)) if len(fv) else 120.0
            wc = np.asarray(sy.render(u, torch.from_numpy(u["y"])[None].to(DEV), 0), np.float64)
            uc = I13.utt_of(wc)
            st = sy.op_stat(enr)
            a, va = R78.feat(sy, refix(u, x, fc), st)
            b, vb = R78.feat(sy, refix(uc, wc, fc), st)
            n = min(len(va), len(vb))
            m = (va[:n] & vb[:n]).float()
            a0, _ = R78.feat(sy, u, st)
            b0, _ = R78.feat(sy, uc, st)
            rows.append({"set": nm, "spk": it["spk"], "src": it["src"], "R": o["copy"]["R_wav"], "cer_x": float(np.mean(o["cer_x"])), "cer_self": o["cer_self"],
                         "R_def": round(max(R78.cosm(a0, b0, m, lag) for lag in range(-4, 5)), 4), "R_fix": round(max(R78.cosm(a, b, m, lag) for lag in range(-4, 5)), 4),
                         "vagree": float((va[:n] == vb[:n]).float().mean())} | {k: round(v, 5) for k, v in (chain_sig(x) | phon_sig(u)).items()})
        print(nm, len([r for r in rows if r["set"] == nm]), flush=True)
    keys = CHAIN + PHON + ("R_fix", "vagree")
    res: dict = {"prereg": "results/snv_p79/prereg.yaml(step_a)", "sys": tag, "sets": {}}
    for nm in sets:
        rs = [r for r in rows if r["set"] == nm]
        res["sets"][nm] = {"源": len(rs), "中央値": {k: round(float(np.median([r[k] for r in rs])), 4) for k in keys + ("R", "R_def")}}
    cv = [r for r in rows if r["set"] == "cv"]
    if len(cv) > 10:
        Rv, C, spk = np.array([r["R"] for r in cv]), np.array([r["cer_x"] for r in cv]), np.array([r["spk"] for r in cv])
        us = sorted(set(spk))
        g = np.random.default_rng(0)
        idx = [np.concatenate([np.flatnonzero(spk == us[j]) for j in g.integers(0, len(us), len(us))]) for _ in range(2000)]
        sp = lambda a, b: float(np.corrcoef(R78.rank(a), R78.rank(b))[0, 1])
        o = np.argsort(Rv)
        k3 = len(o) // 3
        ter = lambda z: [round(float(z[ix].mean()), 4) for ix in (o[:k3], o[k3:2 * k3], o[2 * k3:])]
        tab = {}
        for k in keys:
            z = np.array([r[k] for r in cv])
            bs = [sp(z[ix], Rv[ix]) for ix in idx]
            tab[k] = {"写しの R との順位相関": round(sp(z, Rv), 3), "区間": [round(float(np.nanpercentile(bs, q)), 3) for q in (2.5, 97.5)], "CER との順位相関": round(sp(z, C), 3), "R の下 1/3・中・上 1/3 の平均": ter(z)}
        Rd, Rf = np.array([r["R_def"] for r in cv]), np.array([r["R_fix"] for r in cv])
        res["cv"] = {"量": tab, "R(登録の値)の 3 層": ter(Rv), "R_def の 3 層": ter(Rd), "R_fix の 3 層": ter(Rf), "R_fix − R_def の 3 層": ter(Rf - Rd), "CER の 3 層": ter(C),
                     "R_fix と CER の順位相関": round(sp(Rf, C), 3), "R_def と CER の順位相関": round(sp(Rd, C), 3)}
    out.write_text(json.dumps(res | {"rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
