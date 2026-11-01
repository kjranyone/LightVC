"""D607(推論だけ): 候補の低い k の基音の弱さ(H1 − H2)は、同じ高さの本物の女声の分布の外か・F1 の倍読みと結びつくか・男声の入力の声質を受け継ぐか。
H1・H2 = D606 と同じ 2 周期の Hann 窓の最小二乗。r = 候補は入力の harvest × 2^(k/12)(遅れ合わせ)・本物の録音は自分の harvest。
    uv run python d607_h1h2.py --out ../results/conv_p0/d607/h1h2.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import NPROC, H, harvest
from d598_arm_cmp import prep_
from d605_onset import align
from d606_phase import harm_c

FB = [(150, 180), (180, 210), (210, 250), (250, 300), (300, 400)]


def job(it: tuple) -> tuple:
    from d6_unit_cov import load48
    nm, s, key, ti, k, path = it
    x = prep_(load48(path)).astype(np.float64)
    if nm.startswith("OURS") and s != "devff2":
        hs = harvest(prep_(load48(H / s / "d56" / f"{key}_SRC.wav")))
        r = align(hs * 2 ** (k / 12), harvest(x))
    elif nm == "FF_OURS0Y58":
        r = align(harvest(prep_(load48(H / s / "d56" / f"{key}_SRC.wav"))), harvest(x))
    else:
        r = harvest(x)
    n = min(len(r), len(x) // 240)
    r = r[:n]
    hc = harm_c(x, r)
    d12 = 20 * np.log10(np.abs(hc[:, 0]) + 1e-12) - 20 * np.log10(np.abs(hc[:, 1]) + 1e-12)
    return it, x.astype(np.float32), r, d12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    its = []
    for s in ("dev2", "dev3"):
        tset = set()
        for f in sorted((H / s / "d570" / "OURS5RY58G").glob("p*_t*.wav")):
            key, ti = re.match(r"(p\d+_t(\d+))_", f.name).groups()
            ti = int(ti)
            for nm, k in (("OURS5RY58G", 5), ("OURS12RY58G", 12), ("OURS5Y59", 5), ("OURS5Y58", 5)):
                its.append((nm, s, key, ti, k, H / s / "d570" / nm / f"{key}_{nm}.wav"))
            its.append(("SRC", s, key, ti, 0, H / s / "d56" / f"{key}_SRC.wav"))
            c = H / s / "cen_reg" / f"cen_{ti:03d}.wav"
            if ti not in tset and c.exists():
                tset.add(ti)
                its.append(("CEN", s, f"t{ti:03d}", ti, 0, c))
    for f in sorted((H / "devff2" / "d570" / "OURS0Y58").glob("p*_t*.wav")):
        key = re.match(r"(p\d+_t\d+)_", f.name).group(1)
        its.append(("FF_SRC", "devff2", key, 1000 + len(its), 0, H / "devff2" / "d56" / f"{key}_SRC.wav"))
        its.append(("FF_OURS0Y58", "devff2", key, 1000 + len(its), 0, f))
    with Pool(NPROC) as pl:
        res = pl.map(job, its, chunksize=2)
    print("fits done", len(res), flush=True)
    import torch
    import f0est as FE
    ROOT = Path(__file__).resolve().parent.parent
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    lhz = np.log(FE.bin_hz().numpy())

    def f1track(x: np.ndarray) -> np.ndarray:
        n = len(x) // 240
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * 240])[None].to(dev)], 1), n))
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vo), pr.shape[1])
        kk = pr[:, :m].argmax(0)
        ii = np.clip(kk[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        w = np.take_along_axis(pr[:, :m], ii, 0)
        c = w.sum(0)
        return np.where(vo[:m] & (c >= 0.5), np.exp((w * lhz[ii]).sum(0) / np.maximum(c, 1e-12)), 0.0)

    F: dict = {}
    for (nm, s, key, ti, k, _), x, r, d12 in res:
        fv = f1track(x)
        n = min(len(fv), len(r), len(d12))
        ok = (r[:n] > 0) & np.isfinite(d12[:n])
        up = (fv[:n] > 0) & ok & (12 * np.log2(np.where(fv[:n] > 0, fv[:n], 1.0) / np.where(r[:n] > 0, r[:n], 1.0)) > 9)
        F.setdefault(nm, {})[(s, key)] = {"t": f"{s}|{ti}", "r": r[:n][ok], "d": d12[:n][ok], "up": up[ok], "f1v": (fv[:n] > 0)[ok]}
    rng = np.random.default_rng(607)
    rep: dict = {"f0_bins": FB, "by_bin": {}, "rules": {}}
    for nm, R in F.items():
        rep["by_bin"][nm] = {}
        for lo, hi in FB:
            d = np.concatenate([v["d"][(v["r"] >= lo) & (v["r"] < hi)] for v in R.values()])
            rep["by_bin"][nm][f"{lo}-{hi}"] = {"n": int(len(d)), "median": round(float(np.median(d)), 2) if len(d) >= 50 else None}
        print(nm, {b: v["median"] for b, v in rep["by_bin"][nm].items()}, {b: v["n"] for b, v in rep["by_bin"][nm].items()}, flush=True)

    def gboot(vals: np.ndarray, tg: np.ndarray, fn, B: int = 2000) -> list:
        us = sorted(set(tg))
        idx = {u: np.flatnonzero(tg == u) for u in us}
        bs = [fn(np.concatenate([idx[u] for u in rng.choice(us, len(us))])) for _ in range(B)]
        return [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]

    C = F["OURS5RY58G"]
    keys = sorted(C)
    tg = np.array([C[k]["t"] for k in keys])
    U = [(C[k]["up"][C[k]["f1v"] & (C[k]["d"] < 0)].sum(), (C[k]["f1v"] & (C[k]["d"] < 0)).sum(), C[k]["up"][C[k]["f1v"] & (C[k]["d"] >= 6)].sum(), (C[k]["f1v"] & (C[k]["d"] >= 6)).sum()) for k in keys]
    U = np.array(U, float)

    def w1(ii: np.ndarray) -> float:
        s_ = U[ii].sum(0)
        return (s_[0] / max(s_[1], 1)) / max(s_[2] / max(s_[3], 1), 1e-5)
    s_ = U.sum(0)
    rep["rules"]["W1"] = {"up_rate_d12_lt0": round(float(s_[0] / s_[1]), 4), "up_rate_d12_ge6": round(float(s_[2] / s_[3]), 4), "ratio": round(float(w1(np.arange(len(keys)))), 2), "ci95": gboot(U, tg, w1)}
    print("W1", rep["rules"]["W1"], flush=True)

    def fmed(v: dict) -> float:
        m = (v["r"] >= 150) & (v["r"] < 210)
        return float(np.median(v["d"][m])) if m.sum() >= 30 else np.nan
    cm = np.array([fmed(C[k]) for k in keys])
    real = [(v["t"], fmed(v)) for nm in ("CEN", "FF_SRC") for v in F[nm].values()]
    rm = np.array([z[1] for z in real])
    rt = np.array([z[0] for z in real])
    okc, okr = np.isfinite(cm), np.isfinite(rm)
    diffs = []
    for _ in range(2000):
        a_ = rng.choice(np.flatnonzero(okc), okc.sum())
        b_ = rng.choice(np.flatnonzero(okr), okr.sum())
        diffs.append(cm[a_].mean() - rm[b_].mean())
    rep["rules"]["W2"] = {"cand_files": int(okc.sum()), "real_files": int(okr.sum()), "cand_mean_of_file_medians": round(float(np.nanmean(cm)), 2), "real_mean_of_file_medians": round(float(np.nanmean(rm)), 2),
                          "diff": round(float(np.nanmean(cm) - np.nanmean(rm)), 2), "ci95": [round(float(np.percentile(diffs, 2.5)), 2), round(float(np.percentile(diffs, 97.5)), 2)]}
    print("W2", rep["rules"]["W2"], flush=True)
    from scipy.stats import spearmanr
    sm = np.array([float(np.median(F["SRC"][k]["d"])) for k in keys])
    cmall = np.array([float(np.median(C[k]["d"])) for k in keys])
    rep["rules"]["W3"] = {"rho": round(float(spearmanr(cmall, sm).correlation), 3), "ci95": gboot(np.arange(len(keys)), tg, lambda ii: float(spearmanr(cmall[ii], sm[ii]).correlation))}
    print("W3", rep["rules"]["W3"], flush=True)
    for nm in ("OURS5Y58", "OURS5Y59"):
        rep["rules"][f"record_{nm}_mean_file_median"] = round(float(np.mean([np.median(v["d"]) for v in F[nm].values()])), 2)
    print({k: v for k, v in rep["rules"].items() if k.startswith("record")}, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
