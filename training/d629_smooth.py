"""D629: つややかさの物差しの探索。オーナーの所見(2026-10-28)『ザラザラ、グリッチなど、つややかさの逆の要素が強くて聞くに耐えない』を直接測る統計を、向きを先に決めて作り、
耳 mf_v5_ear(錨 GU* が 12 / 12)と既知のラベル(入力・本物の女声)で較正する。規則は results/conv_p0/d629/prereg.yaml。
    uv run python d629_smooth.py
"""
from __future__ import annotations

import json
import os
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import scipy.signal as ss
import soundfile as sf
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
S = H / "mfv5"
OUT = ROOT / "results/conv_p0/d629"
SR, HOP, SEC = 48000, 240, 8.0
CAND = {5: "RUST5Q3", 7: "RUST7Q3", 12: "RUST12"}
HI = {"pcc_full": True, "pcc_hi": True, "hc_lo": True, "hc_hi": True, "flutter": False, "glitch": False, "flux_spike": False}
SOS_HB = ss.butter(4, [2000, 8000], "bandpass", fs=SR, output="sos")
SOS_HP = ss.butter(4, 3000, "highpass", fs=SR, output="sos")


def load(p: Path, mid: bool = False) -> np.ndarray:
    x, sr = sf.read(p, dtype="float64", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        fr = Fraction(SR, sr)
        x = ss.resample_poly(x, fr.numerator, fr.denominator)
    n = int(SEC * SR)
    if mid and len(x) > n:
        st = (len(x) - n) // 2
        x = x[st:st + n]
    x = x[:n]
    return x / (np.sqrt((x ** 2).mean()) + 1e-12) * 0.05


def melfb(n_fft: int, nb: int = 64, fmin: float = 50.0, fmax: float = 16000.0) -> np.ndarray:
    m = lambda f: 2595 * np.log10(1 + f / 700)
    im = lambda z: 700 * (10 ** (z / 2595) - 1)
    pts = im(np.linspace(m(fmin), m(fmax), nb + 2))
    fr = np.fft.rfftfreq(n_fft, 1 / SR)
    fb = np.zeros((nb, len(fr)))
    for b in range(nb):
        l, c, r = pts[b], pts[b + 1], pts[b + 2]
        fb[b] = np.clip(np.minimum((fr - l) / (c - l), (r - fr) / (r - c)), 0, None)
    return fb


FB = melfb(1024)


def feats(x: np.ndarray) -> dict:
    x16 = ss.resample_poly(x, 1, 3)
    f0, _ = pw.harvest(x16, 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
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
    r: dict = {"steady_s": float(len(idx) * 0.005)}
    if len(idx) < 40:
        return r | {k: float("nan") for k in HI}
    hb = ss.sosfiltfilt(SOS_HB, x)

    def pcc(y: np.ndarray) -> float:
        out = []
        for i in idx:
            n = i * HOP
            T = int(round(SR / f0[i]))
            L = max(1, int(0.03 * T))
            if n - T < 0 or n + T + L > len(y):
                continue
            a = y[n - T:n]
            B = np.lib.stride_tricks.sliding_window_view(y[n - L:n + L + T], T)
            c = (B @ a) / (np.sqrt((B * B).sum(1) * (a @ a)) + 1e-20)
            out.append(c.max())
        return float(np.median(out)) if out else float("nan")

    r["pcc_full"] = pcc(x)
    r["pcc_hi"] = pcc(hb)

    n_fft = 2048
    _, _, Z = ss.stft(x, SR, window="hann", nperseg=n_fft, noverlap=n_fft - HOP, boundary="even", padded=True)
    M = 20 * np.log10(np.abs(Z) + 1e-9).T
    bw = SR / n_fft
    hlo, hhi, amps = [], [], {}
    for i in idx:
        if i >= len(M):
            continue
        f = f0[i]
        m = M[i]
        nh = int(8000 // f)
        hs = np.arange(1, nh + 1)
        pk = np.array([m[int(np.floor((h * f - 0.1 * f) / bw)):int(np.ceil((h * f + 0.1 * f) / bw)) + 1].max() for h in hs])
        vl = np.array([m[int(np.floor(((h + 0.5) * f - 0.1 * f) / bw)):int(np.ceil(((h + 0.5) * f + 0.1 * f) / bw)) + 1].mean() for h in hs])
        cf = hs * f
        lo, hi = (cf >= 500) & (cf < 3000), (cf >= 3000) & (cf < 8000)
        if lo.any():
            hlo.append(float((pk[lo] - vl[lo]).mean()))
        if hi.any():
            hhi.append(float((pk[hi] - vl[hi]).mean()))
        a = np.full(20, np.nan)
        k = min(20, nh)
        a[:k] = pk[:k]
        a[a < pk.max() - 40] = np.nan
        amps[i] = a
    r["hc_lo"] = float(np.median(hlo)) if hlo else float("nan")
    r["hc_hi"] = float(np.median(hhi)) if hhi else float("nan")
    d2 = [np.abs(amps[i + 1] - 2 * amps[i] + amps[i - 1]) for i in idx if (i - 1) in amps and (i + 1) in amps]
    d2 = np.concatenate(d2) if d2 else np.array([np.nan])
    r["flutter"] = float(np.nanmedian(d2)) if np.isfinite(d2).any() else float("nan")

    hp = ss.sosfiltfilt(SOS_HP, x)
    e = np.convolve(hp ** 2, np.ones(48) / 48, "same")[::24]
    med = median_filter(e, size=81, mode="nearest")
    rat = 10 * np.log10((e + 1e-20) / (med + 1e-20))
    pks, _ = ss.find_peaks(rat, height=10.0, distance=10)
    g = 0
    for p in pks:
        fi = int(round(p * 24 / HOP))
        if fi >= nf or not st[fi]:
            continue
        Tq = SR / f0[fi] / 24
        w = max(1, int(round(0.05 * Tq)))
        nb = []
        for c in (p - Tq, p + Tq):
            c = int(round(c))
            if 0 <= c - w and c + w < len(e):
                nb.append(e[c - w:c + w + 1].max())
        if nb and e[p] > 10 * max(nb):
            g += 1
    r["glitch"] = g / r["steady_s"]

    _, _, Z2 = ss.stft(x, SR, window="hann", nperseg=1024, noverlap=1024 - 120, boundary="even", padded=True)
    L = 10 * np.log10(FB @ (np.abs(Z2) ** 2) + 1e-12).T
    L = np.maximum(L, L.max() - 80)
    fl = np.r_[0.0, np.abs(np.diff(L, axis=0)).mean(1)]
    fidx = np.flatnonzero(np.repeat(st, 2)[: len(fl)])
    r["_flux"] = fl[fidx].astype(np.float32).tolist()
    return r


def job(it: tuple) -> tuple:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
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
    realflux = np.concatenate([np.array(v["_flux"]) for k, v in res.items() if k.startswith("real|") and "_flux" in v])
    thr = float(np.percentile(realflux, 99))
    for v in res.values():
        if "_flux" in v:
            fl = np.array(v.pop("_flux"))
            v["flux_spike"] = float((fl > thr).mean()) if len(fl) else float("nan")
    (OUT / "feat.json").write_text(json.dumps({"flux_thr_db": thr, "feat": res}, indent=0))

    def better(f: str, a: float, b: float) -> bool:
        return a > b if HI[f] else a < b

    def val(n: str, f: str) -> float:
        return res.get(n, {}).get(f, float("nan"))

    tally = json.loads((ROOT / "results/earbattery/mf_v5_ear/tally.json").read_text())["trials"]
    real = {f: np.array([v[f] for k, v in res.items() if k.startswith("real|") and f in v and np.isfinite(v[f])]) for f in HI}
    rep = {"flux_thr_db": thr, "n_real": int(len(real["pcc_full"])), "features": {}}
    for f in HI:
        r = {"ear": {}, "gate": {}}
        npos = 0
        for t, tv in tally.items():
            c, g = val(f"cand{tv['k']}|{tv['pair']}", f), val(f"GU{tv['k']}|{tv['pair']}", f)
            ok = better(f, g, c)
            npos += ok
            r["ear"][t] = {"k": tv["k"], "cand": round(c, 4), "GU": round(g, 4), "src": round(val(f"SRC|{tv['pair']}", f), 4), "GU_better": bool(ok),
                           "cand_ok": tv["cand_ok"], "GU_ok": tv["GU_ok"]}
        r["ear_GU_better"] = int(npos)
        okall = npos >= 11
        for k in CAND:
            cs = np.array([val(f"cand{k}|{p}", f) for p in pairs])
            gs = np.array([val(f"GU{k}|{p}", f) for p in pairs])
            sr_ = np.array([val(f"SRC|{p}", f) for p in pairs])
            m1, m2 = np.isfinite(cs) & np.isfinite(gs), np.isfinite(cs) & np.isfinite(sr_)
            fg = float(np.mean([better(f, g, c) for c, g in zip(cs[m1], gs[m1])]))
            fs = float(np.mean([better(f, s, c) for c, s in zip(cs[m2], sr_[m2])]))
            cc = cs[np.isfinite(cs)]
            rv = real[f]
            auc = float(np.mean([(np.sum(rv > c) + 0.5 * np.sum(rv == c)) / len(rv) for c in cc])) if HI[f] else float(np.mean([(np.sum(rv < c) + 0.5 * np.sum(rv == c)) / len(rv) for c in cc]))
            r["gate"][k] = {"GU_better": round(fg, 3), "SRC_better": round(fs, 3), "AUC_real_vs_cand": round(auc, 3),
                            "median": {"cand": round(float(np.nanmedian(cs)), 4), "GU": round(float(np.nanmedian(gs)), 4), "SRC": round(float(np.nanmedian(sr_)), 4), "real": round(float(np.median(rv)), 4)}}
            okall &= fg >= 0.75 and fs >= 0.75 and auc >= 0.75
        v3 = sorted(k_.split("|")[1] for k_ in res if k_.startswith("v3CV|"))
        a3 = [(val(f"v3CV|{p}", f), val(f"v3SRC|{p}", f)) for p in v3]
        a3 = [(c, s) for c, s in a3 if np.isfinite(c) and np.isfinite(s)]
        r["record_v3_SRC_better_than_P57S0"] = round(float(np.mean([better(f, s, c) for c, s in a3])), 3) if a3 else None
        r["rule_E2"] = bool(okall)
        rep["features"][f] = r
        g = r["gate"]
        print(f"{f:10s} ear {npos:2d}/12 | " + " | ".join(f"k{k} GU>{g[k]['GU_better']:.2f} SRC>{g[k]['SRC_better']:.2f} AUC {g[k]['AUC_real_vs_cand']:.2f} "
              f"[c {g[k]['median']['cand']:.3f} g {g[k]['median']['GU']:.3f} s {g[k]['median']['SRC']:.3f} r {g[k]['median']['real']:.3f}]" for k in CAND)
              + f" | v3 {r['record_v3_SRC_better_than_P57S0']} | E2 {r['rule_E2']}", flush=True)
    rep["rule_E2_features"] = [f for f, r in rep["features"].items() if r["rule_E2"]]
    rep["fork"] = "some_E2" if rep["rule_E2_features"] else "none_E2"
    (OUT / "result.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("rule_E2", rep["rule_E2_features"], "fork", rep["fork"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
