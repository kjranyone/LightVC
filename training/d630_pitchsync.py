"""D630: D629 の倍音の山谷とちらつきを、ピッチに合わせた窓(4 周期の Hann・周期ごとのフレーム)で測り直す。規則 E3 = D629 の E2 + 声域への頑健さ(本物の女声で |Spearman ρ(統計, f0)| ≤ 0.3)。
規則は results/conv_p0/d630/prereg.yaml。
    uv run python d630_pitchsync.py
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import scipy.signal as ss
from scipy.stats import spearmanr

from d629_smooth import CAND, H, HOP, ROOT, S, SR, load

OUT = ROOT / "results/conv_p0/d630"
HI = {"phc_lo": True, "phc_hi": True, "pflutter": False}


def spec(x: np.ndarray, c: int, T: float, nper: float) -> tuple:
    N = int(round(nper * T))
    a = c - N // 2
    if a < 0 or a + N > len(x):
        return None, None
    nfft = 1 << int(np.ceil(np.log2(8 * N)))
    m = 20 * np.log10(np.abs(np.fft.rfft(x[a:a + N] * np.hanning(N), nfft)) + 1e-9)
    return m, SR / nfft


def feats(x: np.ndarray) -> dict:
    f0, _ = pw.harvest(ss.resample_poly(x, 1, 3), 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
    nf = min(len(f0), len(x) // HOP)
    f0 = f0[:nf]
    v = f0 > 0
    st = v.copy()
    for s in range(1, 5):
        st[s:] &= v[:-s]
        st[:-s] &= v[s:]
    st[:4] = False
    st[-4:] = False
    idx = np.flatnonzero(st)
    r: dict = {"steady_s": float(len(idx) * 0.005), "f0med": float(np.median(f0[v])) if v.any() else float("nan")}
    if len(idx) < 40:
        return r | {k: float("nan") for k in HI}
    lo_c, hi_c = [], []
    for i in idx:
        f = f0[i]
        m, bw = spec(x, i * HOP, SR / f, 4.0)
        if m is None:
            continue
        hs = np.arange(1, int(8000 // f) + 1)
        w = max(1, int(np.ceil(0.05 * f / bw)))
        pc, vc = np.round(hs * f / bw).astype(int), np.round((hs + 0.5) * f / bw).astype(int)
        ok = vc + w < len(m)
        hs, pc, vc = hs[ok], pc[ok], vc[ok]
        pk = np.array([m[p - w:p + w + 1].max() for p in pc])
        vl = np.array([m[q - w:q + w + 1].mean() for q in vc])
        cf = hs * f
        lo, hi = (cf >= 500) & (cf < 3000), (cf >= 3000) & (cf < 8000)
        if lo.any():
            lo_c.append(float((pk[lo] - vl[lo]).mean()))
        if hi.any():
            hi_c.append(float((pk[hi] - vl[hi]).mean()))
    r["phc_lo"] = float(np.median(lo_c)) if lo_c else float("nan")
    r["phc_hi"] = float(np.median(hi_c)) if hi_c else float("nan")
    d2 = []
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for run in runs:
        if len(run) < 4:
            continue
        pos, end = float(run[0] * HOP), run[-1] * HOP
        A = []
        while pos <= end:
            fi = int(round(pos / HOP))
            f = f0[fi]
            m, bw = spec(x, int(round(pos)), SR / f, 2.0)
            if m is None:
                break
            hs = np.arange(1, min(20, int(8000 // f)) + 1)
            w = max(1, int(np.ceil(0.1 * f / bw)))
            pc = np.round(hs * f / bw).astype(int)
            a = np.full(20, np.nan)
            a[: len(hs)] = [m[p - w:p + w + 1].max() for p in pc]
            a[a < np.nanmax(a) - 30] = np.nan
            A.append(a)
            pos += SR / f
        if len(A) >= 3:
            A = np.array(A)
            d2.append(np.abs(A[2:] - 2 * A[1:-1] + A[:-2]).ravel())
    d2 = np.concatenate(d2) if d2 else np.array([np.nan])
    r["pflutter"] = float(np.nanmedian(d2)) if np.isfinite(d2).any() else float("nan")
    return r


def job(it: tuple) -> tuple:
    name, path, mid = it
    try:
        return name, feats(load(Path(path), mid))
    except Exception as ex:
        return name, {"error": repr(ex)}


def main() -> int:
    items = []
    pairs = sorted({f.name[:10] for f in (S / "gate" / "GU5").glob("p*_GU5.wav")})
    for k, c in CAND.items():
        for p in pairs:
            items.append((f"cand{k}|{p}", str(S / "gate" / c / f"{p}_{c}.wav"), False))
            items.append((f"GU{k}|{p}", str(S / "gate" / f"GU{k}" / f"{p}_GU{k}.wav"), False))
    for p in pairs:
        items.append((f"SRC|{p}", str(S / "d56" / f"{p}_SRC.wav"), False))
    cache = json.loads((S / "d109_cache.json").read_text())
    for kk in cache:
        if kk.startswith("2|pool|"):
            _, _, spk, fn = kk.split("|")
            items.append((f"real|{spk}", str(ROOT / "female-dataset" / spk / fn), True))
    for f in sorted((H / "mfv3" / "gate" / "CV").glob("p*_CV.wav")):
        p = f.name[:10]
        items.append((f"v3CV|{p}", str(f), False))
        items.append((f"v3SRC|{p}", str(H / "mfv3" / "d56" / f"{p}_SRC.wav"), False))
    print("items", len(items), flush=True)
    with Pool(int(os.environ.get("LVC_NPROC", "12"))) as pl:
        res = dict(pl.imap_unordered(job, items, chunksize=4))
    err = {k: v["error"] for k, v in res.items() if "error" in v}
    print("errors", len(err), list(err.items())[:3], flush=True)
    (OUT / "feat.json").write_text(json.dumps(res, indent=0))

    def better(f: str, a: float, b: float) -> bool:
        return a > b if HI[f] else a < b

    def val(n: str, f: str) -> float:
        return res.get(n, {}).get(f, float("nan"))

    def auc(f: str, good: np.ndarray, bad: np.ndarray) -> float:
        g = good[np.isfinite(good)]
        return float(np.mean([(np.sum(g > b) + 0.5 * np.sum(g == b)) / len(g) if HI[f] else (np.sum(g < b) + 0.5 * np.sum(g == b)) / len(g) for b in bad[np.isfinite(bad)]]))

    tally = json.loads((ROOT / "results/earbattery/mf_v5_ear/tally.json").read_text())["trials"]
    rk = [k for k in res if k.startswith("real|") and "error" not in res[k]]
    rep: dict = {"n_real": len(rk), "features": {}}
    for f in HI:
        rv = np.array([res[k][f] for k in rk])
        rf = np.array([res[k]["f0med"] for k in rk])
        m = np.isfinite(rv) & np.isfinite(rf)
        rho = float(spearmanr(rv[m], rf[m]).correlation)
        r = {"ear": {}, "gate": {}, "real_rho_f0": round(rho, 3)}
        npos = 0
        for t, tv in tally.items():
            c, g = val(f"cand{tv['k']}|{tv['pair']}", f), val(f"GU{tv['k']}|{tv['pair']}", f)
            ok = better(f, g, c)
            npos += ok
            r["ear"][t] = {"k": tv["k"], "cand": round(c, 4), "GU": round(g, 4), "src": round(val(f"SRC|{tv['pair']}", f), 4), "GU_better": bool(ok), "cand_ok": tv["cand_ok"], "GU_ok": tv["GU_ok"]}
        r["ear_GU_better"] = int(npos)
        okall = npos >= 11 and abs(rho) <= 0.3
        srcs = np.array([val(f"SRC|{p}", f) for p in pairs])
        r["record_AUC_real_vs_SRC"] = round(auc(f, rv, srcs), 3)
        for k in CAND:
            cs = np.array([val(f"cand{k}|{p}", f) for p in pairs])
            gs = np.array([val(f"GU{k}|{p}", f) for p in pairs])
            m1, m2 = np.isfinite(cs) & np.isfinite(gs), np.isfinite(cs) & np.isfinite(srcs)
            fg = float(np.mean([better(f, g, c) for c, g in zip(cs[m1], gs[m1])]))
            fs = float(np.mean([better(f, s, c) for c, s in zip(cs[m2], srcs[m2])]))
            a = auc(f, rv, cs)
            r["gate"][k] = {"GU_better": round(fg, 3), "SRC_better": round(fs, 3), "AUC_real_vs_cand": round(a, 3), "AUC_real_vs_GU": round(auc(f, rv, gs), 3),
                            "median": {"cand": round(float(np.nanmedian(cs)), 4), "GU": round(float(np.nanmedian(gs)), 4), "SRC": round(float(np.nanmedian(srcs)), 4), "real": round(float(np.nanmedian(rv)), 4)}}
            okall &= fg >= 0.75 and fs >= 0.75 and a >= 0.75
        v3 = sorted(k_.split("|")[1] for k_ in res if k_.startswith("v3CV|"))
        a3 = [(val(f"v3CV|{p}", f), val(f"v3SRC|{p}", f)) for p in v3]
        a3 = [(c, s) for c, s in a3 if np.isfinite(c) and np.isfinite(s)]
        r["record_v3_SRC_better_than_P57S0"] = round(float(np.mean([better(f, s, c) for c, s in a3])), 3) if a3 else None
        r["rule_E3"] = bool(okall)
        rep["features"][f] = r
        g = r["gate"]
        print(f"{f:9s} ear {npos:2d}/12 rho {rho:+.2f} | " + " | ".join(f"k{k} GU>{g[k]['GU_better']:.2f} SRC>{g[k]['SRC_better']:.2f} AUCc {g[k]['AUC_real_vs_cand']:.2f} AUCg {g[k]['AUC_real_vs_GU']:.2f} "
              f"[c {g[k]['median']['cand']:.3f} g {g[k]['median']['GU']:.3f} s {g[k]['median']['SRC']:.3f} r {g[k]['median']['real']:.3f}]" for k in CAND)
              + f" | AUC real/SRC {r['record_AUC_real_vs_SRC']} | v3 {r['record_v3_SRC_better_than_P57S0']} | E3 {r['rule_E3']}", flush=True)
    rep["rule_E3_features"] = [f for f, r in rep["features"].items() if r["rule_E3"]]
    rep["fork"] = "some_E3" if rep["rule_E3_features"] else "none_E3"
    (OUT / "result.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("rule_E3", rep["rule_E3_features"], "fork", rep["fork"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
