"""D601(推論だけ): 候補の倍音の間の谷の浅さ(psd_disc.psgram の valley)は本物の録音の分布の外か・F1 の倍読みと結びつくか。全ての版で f0 = その音自身の harvest(同じ取り方)。
    uv run python d601_valley.py --out ../results/conv_p0/d601/valley.json
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

from d592_refkc import NPROC, H, SR, boot, harvest

ROOT = Path(__file__).resolve().parent.parent
CAND = ["OURS5RY58G", "OURS12RY58G", "OURS5RY60G"]


def prep(x: np.ndarray) -> np.ndarray:
    x = x[: int(8 * SR)].astype(np.float32)
    return x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05


def job(it: tuple) -> tuple:
    from d6_unit_cov import load48
    grp, s, key, ti, path = it
    x = prep(load48(path))
    return grp, s, key, ti, x, harvest(x)


def items() -> list:
    out = []
    for s in ("dev2", "dev3"):
        S = H / s
        seen_t = set()
        for f in sorted((S / "d570" / "OURS5RY58G").glob("p*_t*.wav")):
            m = re.match(r"(p\d+_t(\d+))_", f.name)
            key, ti = m.group(1), int(m.group(2))
            out.append(("SRC", s, key, ti, S / "d56" / f"{key}_SRC.wav"))
            c = S / "cen_reg" / f"cen_{ti:03d}.wav"
            if ti not in seen_t and c.exists() and (S / "d570" / "REFKC5" / f"{key}_REFKC5.wav").exists():
                seen_t.add(ti)
                out.append(("CEN", s, f"t{ti:03d}", ti, c))
            for v in CAND + ["REFKC5", "REFKC12"]:
                p = S / "d570" / v / f"{key}_{v}.wav"
                if p.exists():
                    out.append((v, s, key, ti, p))
    S = H / "devff2"
    for f in sorted((S / "d570" / "OURS0Y58").glob("p*_t*.wav")):
        key = re.match(r"(p\d+_t\d+)_", f.name).group(1)
        out.append(("FF_SRC", "devff2", key, -1, S / "d56" / f"{key}_SRC.wav"))
        out.append(("FF_OURS0Y58", "devff2", key, -1, f))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    its = items()
    with Pool(NPROC) as pl:
        res = list(pl.imap(job, its, chunksize=4))
    print("harvest done", len(res), flush=True)
    import torch
    import f0est as FE
    import nvoc as N
    import psd_disc as PS
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    taps = PS.lpf_taps().to(dev)
    lhz = np.log(FE.bin_hz().numpy())

    def f1track(x: np.ndarray) -> np.ndarray:
        n = len(x) // N.HOP
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * N.HOP])[None].to(dev)], 1), n))
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vo), pr.shape[1])
        k = pr[:, :m].argmax(0)
        ii = np.clip(k[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        w = np.take_along_axis(pr[:, :m], ii, 0)
        c = w.sum(0)
        return np.where(vo[:m] & (c >= 0.5), np.exp((w * lhz[ii]).sum(0) / np.maximum(c, 1e-12)), 0.0)

    rows: dict = {}
    for grp, s, key, ti, x, h in res:
        f0 = h[1:]
        T = min(len(f0), len(x) // N.HOP)
        with torch.no_grad():
            z, ok = PS.psgram(torch.from_numpy(x[: T * N.HOP])[None].to(dev), torch.from_numpy(f0[:T].astype(np.float32))[None].to(dev), 0, None, taps=taps)
        st = PS.z_stats(z, ok)
        if st["nf"] < 20:
            continue
        if grp in CAND or grp == "FF_OURS0Y58":
            fv = f1track(x)
            n = min(len(fv), len(h))
            b = (fv[:n] > 0) & (h[:n] > 0)
            st["up"] = float((12 * np.log2(fv[:n][b] / h[:n][b]) > 9).sum() / max(b.sum(), 1))
        rows.setdefault(grp, {})[(s, key)] = {"t": ti, **st}
    rep: dict = {"sets": {}, "pairs": {}}
    for g, R in rows.items():
        vv = np.array([r["valley"] for r in R.values()])
        rep["sets"][g] = {"n": len(R), "valley_median": round(float(np.median(vv)), 4), "valley_p15": round(float(np.percentile(vv, 15)), 4), "valley_p85": round(float(np.percentile(vv, 85)), 4),
                          "oe_z_median": round(float(np.median([r["oe_z"] for r in R.values()])), 4)} | ({"up_mean": round(float(np.mean([r["up"] for r in R.values()])), 4)} if "up" in next(iter(R.values())) else {})
        print(g, rep["sets"][g], flush=True)
    pool = np.array([r["valley"] for g in ("CEN", "FF_SRC") for r in rows[g].values()])
    p15, p85 = float(np.percentile(pool, 15)), float(np.percentile(pool, 85))
    rep["real_female_pool"] = {"n": len(pool), "p15": round(p15, 4), "p85": round(p85, 4), "median": round(float(np.median(pool)), 4)}
    for g in CAND + ["REFKC5", "REFKC12", "FF_OURS0Y58", "SRC"]:
        vv = np.array([r["valley"] for r in rows[g].values()])
        rep["sets"][g]["out_frac_above_p85"] = round(float((vv > p85).mean()), 3)
    print("pool", rep["real_female_pool"], flush=True)
    rng = np.random.default_rng(601)
    for g in CAND + ["REFKC5"]:
        ks = [k for k in rows[g] if (k[0], f"t{rows[g][k]['t']:03d}") in rows["CEN"]]
        d = np.array([rows[g][k]["valley"] - rows["CEN"][(k[0], f"t{rows[g][k]['t']:03d}")]["valley"] for k in ks])
        tg = np.array([f"{k[0]}|{rows[g][k]['t']}" for k in ks])
        rep["pairs"][f"{g}-CEN:valley"] = {"n": len(ks), "diff": round(float(d.mean()), 4), "ci95": boot(d, tg, rng)}
        print(f"{g}-CEN:valley", rep["pairs"][f"{g}-CEN:valley"], flush=True)
    ks = sorted(set(rows["FF_OURS0Y58"]) & set(rows["FF_SRC"]))
    d = np.array([rows["FF_OURS0Y58"][k]["valley"] - rows["FF_SRC"][k]["valley"] for k in ks])
    rep["pairs"]["FF_OURS0Y58-FF_SRC:valley"] = {"n": len(ks), "diff": round(float(d.mean()), 4), "ci95": boot(d, np.array([k[1] for k in ks]), rng)}
    print("FF_OURS0Y58-FF_SRC:valley", rep["pairs"]["FF_OURS0Y58-FF_SRC:valley"], flush=True)
    from scipy.stats import spearmanr
    cv = [(r["valley"], r["up"], f"{k[0]}|{r['t']}") for g in ("OURS5RY58G", "OURS5RY60G") for k, r in rows[g].items()]
    vx, ux, tg = np.array([c[0] for c in cv]), np.array([c[1] for c in cv]), np.array([c[2] for c in cv])
    rho = float(spearmanr(vx, ux).correlation)
    us = sorted(set(tg))
    idx = {u: np.flatnonzero(tg == u) for u in us}
    bs = []
    for _ in range(2000):
        ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        bs.append(float(spearmanr(vx[ii], ux[ii]).correlation))
    rep["V2_spearman_valley_up"] = {"n": len(cv), "rho": round(rho, 3), "ci95": [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]}
    print("V2", rep["V2_spearman_valley_up"], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
