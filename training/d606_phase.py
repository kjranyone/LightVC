"""D606(推論だけ): 倍音の相対位相 θ = φ2 − 2φ1(H1・H2 の当てはめの位相・[−π, π))と F1 の倍読みの結びつき。D605 と同じ 2 周期の Hann 窓の最小二乗・基準の高さ r。
版: 候補 OURS5RY58G / OURS12RY58G・REFKC5(WORLD)・devff2 の実女声の入力 FF_SRC(r = 自分の harvest)と女声 → 女声の製品 FF_OURS0Y58(r = 入力の harvest)。
    uv run python d606_phase.py --out ../results/conv_p0/d606/phase.json
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

from d592_refkc import NPROC, H, SR, harvest
from d598_arm_cmp import prep_
from d605_onset import align

NB = 8


def harm_c(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    out = np.full((len(r), 3), np.nan + 0j)
    for i in np.nonzero(r > 0)[0]:
        h = int(round(SR / r[i]))
        c = i * 240
        if c - h < 0 or c + h >= len(x):
            continue
        n = np.arange(-h, h + 1)
        w = 0.5 + 0.5 * np.cos(np.pi * n / (h + 1))
        ph = 2 * np.pi * r[i] * n / SR
        B = np.stack([np.ones_like(ph)] + [f(k * ph) for k in (1, 2, 3) for f in (np.cos, np.sin)], 1)
        co, *_ = np.linalg.lstsq(B * w[:, None], x[c - h:c + h + 1] * w, rcond=None)
        out[i] = co[1::2] - 1j * co[2::2]
    return out


def job(it: tuple) -> tuple:
    from d6_unit_cov import load48
    nm, s, key, ti, k, path = it
    S = H / s
    x = prep_(load48(path)).astype(np.float64)
    if nm == "FF_SRC":
        r = harvest(x)
    else:
        hs = harvest(prep_(load48(S / "d56" / f"{key}_SRC.wav")))
        if nm.startswith("REFKC"):
            fc = harvest(load48(S / "cen_reg" / f"cen_{ti:03d}.wav").astype(np.float64)[: 8 * SR])
            r = fc * float(np.median(hs[hs > 0])) * 2 ** (k / 12) / float(np.median(fc[fc > 0]))
        else:
            r = align(hs * 2 ** (k / 12), harvest(x))
    n = min(len(r), len(x) // 240)
    r = r[:n]
    return it, x, r, harm_c(x, r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    its = []
    for s in ("dev2", "dev3"):
        for nm, k in (("OURS5RY58G", 5), ("OURS12RY58G", 12), ("REFKC5", 5)):
            for f in sorted((H / s / "d570" / nm).glob("p*_t*.wav")):
                key, ti = re.match(r"(p\d+_t(\d+))_", f.name).groups()
                if (H / s / "d570" / "REFKC5" / f"{key}_REFKC5.wav").exists():
                    its.append((nm, s, key, int(ti), k, f))
    for f in sorted((H / "devff2" / "d570" / "OURS0Y58").glob("p*_t*.wav")):
        key = re.match(r"(p\d+_t\d+)_", f.name).group(1)
        its.append(("FF_SRC", "devff2", key, -1, 0, H / "devff2" / "d56" / f"{key}_SRC.wav"))
        its.append(("FF_OURS0Y58", "devff2", key, -1, 0, f))
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
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * 240].astype(np.float32))[None].to(dev)], 1), n))
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vo), pr.shape[1])
        kk = pr[:, :m].argmax(0)
        ii = np.clip(kk[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        w = np.take_along_axis(pr[:, :m], ii, 0)
        c = w.sum(0)
        return np.where(vo[:m] & (c >= 0.5), np.exp((w * lhz[ii]).sum(0) / np.maximum(c, 1e-12)), 0.0)

    edges = np.linspace(-np.pi, np.pi, NB + 1)
    acc: dict = {}
    for (nm, s, key, ti, k, _), x, r, hc in res:
        fv = f1track(x)
        n = min(len(fv), len(r))
        ok = (fv[:n] > 0) & np.isfinite(hc[:n, 0]) & (np.abs(hc[:n, 0]) > 0) & (np.abs(hc[:n, 1]) > 0)
        th = np.angle(hc[:n, 1] * np.conj(hc[:n, 0]) ** 2)
        up = 12 * np.log2(np.where(ok, fv[:n], 1.0) / np.where(r[:n] > 0, r[:n], 1.0)) > 9
        d12 = 20 * np.log10(np.abs(hc[:n, 0]) + 1e-12) - 20 * np.log10(np.abs(hc[:n, 1]) + 1e-12)
        b = np.clip(np.digitize(th, edges) - 1, 0, NB - 1)
        A = acc.setdefault(nm, {"cnt": np.zeros(NB), "up": np.zeros(NB), "files": [], "d12": [[] for _ in range(NB)]})
        fc_, fu_ = np.zeros(NB), np.zeros(NB)
        for j in range(NB):
            m = ok & (b == j)
            fc_[j], fu_[j] = m.sum(), (m & up).sum()
            A["d12"][j].extend(d12[m][::5].tolist())
        A["cnt"] += fc_
        A["up"] += fu_
        A["files"].append((f"{s}|{ti}", fc_, fu_))
    rng = np.random.default_rng(606)
    rep: dict = {"bin_edges": [round(float(e), 3) for e in edges], "versions": {}}
    for nm, A in acc.items():
        rate = A["up"] / np.maximum(A["cnt"], 1)
        frac = A["cnt"] / max(A["cnt"].sum(), 1)
        tg = np.array([f[0] for f in A["files"]])
        us = sorted(set(tg))
        idx = {u: np.flatnonzero(tg == u) for u in us}
        FC = np.array([f[1] for f in A["files"]])
        FU = np.array([f[2] for f in A["files"]])
        jw, jb = int(np.argmax(rate)), int(np.argmin(np.where(A["cnt"] > 100, rate, np.inf)))
        bs = []
        for _ in range(2000):
            ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
            c_, u_ = FC[ii].sum(0), FU[ii].sum(0)
            rr = u_ / np.maximum(c_, 1)
            bs.append(rr[jw] / max(rr[jb], 1e-4))
        rep["versions"][nm] = {"n_frames": int(A["cnt"].sum()), "frac_by_bin": [round(float(v), 4) for v in frac], "up_rate_by_bin": [round(float(v), 4) for v in rate],
                               "worst_bin": jw, "best_bin": jb, "ratio_worst_best": round(float(rate[jw] / max(rate[jb], 1e-4)), 2),
                               "ratio_ci95": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
                               "h1_minus_h2_median_by_bin": [round(float(np.median(v)), 2) if v else None for v in A["d12"]]}
        print(nm, json.dumps(rep["versions"][nm]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
