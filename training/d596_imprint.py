"""D596(推論だけ): 候補の奇数倍音の不足(OE)が目標の声域の倍音の縞の写り込みか。ΔOE(候補 − REFKC)と P(q)(q = 目標の f0 / 出力の f0)の相関。
    uv run python d596_imprint.py --sets dev2 dev3 --out ../results/conv_p0/d596/imprint.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import H, SR, harvest

ROOT = Path(__file__).resolve().parent.parent
NFFT = 2048


def pq(q: float) -> float:
    return float(np.mean([np.cos(2 * np.pi * n / q) for n in (2, 4, 6, 8)]) - np.mean([np.cos(2 * np.pi * n / q) for n in (3, 5, 7)]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import f0est as FE
    import nvoc as N
    from d6_unit_cov import load48
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    lhz = np.log(FE.bin_hz().numpy())
    HOP = N.HOP

    def f0track(x: np.ndarray) -> np.ndarray:
        n = len(x) // HOP
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * HOP])[None].to(dev)], 1), n))
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vo), pr.shape[1])
        k = pr[:, :m].argmax(0)
        ii = np.clip(k[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        w = np.take_along_axis(pr[:, :m], ii, 0)
        c = w.sum(0)
        f = np.exp((w * lhz[ii]).sum(0) / np.maximum(c, 1e-12))
        return np.where(vo[:m] & (c >= 0.5), f, 0.0)

    def prep(x: np.ndarray) -> np.ndarray:
        x = x[: int(8 * SR)].astype(np.float32)
        return x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05

    def shift(r0: np.ndarray, lag: int) -> np.ndarray:
        r = np.roll(r0, lag)
        if lag > 0:
            r[:lag] = 0
        elif lag < 0:
            r[lag:] = 0
        return r

    def align(r0: np.ndarray, h: np.ndarray) -> np.ndarray:
        best = (0, -1.0)
        for lag in range(-10, 11):
            r = shift(r0, lag)
            n = min(len(r), len(h))
            b = (r[:n] > 0) & (h[:n] > 0)
            if b.sum() < 50:
                continue
            ag = float((np.abs(12 * np.log2(h[:n][b] / r[:n][b])) < 1).mean())
            if ag > best[1]:
                best = (lag, ag)
        return shift(r0, best[0])

    def oe(x: np.ndarray, r: np.ndarray) -> float:
        fv = f0track(x)
        _, _, Z = ss.stft(x.astype(np.float64), SR, nperseg=NFFT, noverlap=NFFT - 240)
        A = np.abs(Z)
        n = min(len(fv), len(r), A.shape[1])
        bw = SR / NFFT
        vals = []
        for t in np.nonzero((fv[:n] > 0) & (r[:n] > 0))[0]:
            def amp(f: float) -> float:
                i = int(round(f / bw))
                return float(A[max(i - 1, 0): i + 2, t].max()) if i + 1 < A.shape[0] else 0.0
            vals.append(sum(amp(k * r[t]) for k in (3, 5, 7)) / max(sum(amp(k * r[t]) for k in (2, 4, 6, 8)), 1e-12))
        return float(np.median(vals)) if vals else float("nan")

    rows: dict = {}
    for s in a.sets:
        S = H / s
        ft = {m["ti"]: m["f0_ref"] for m in json.loads((S / "cen_reg" / "meta.json").read_text()) if m["ok"]}
        src: dict = {}
        for k in (5, 7, 12):
            for nm in (f"OURS{k}RY58G", f"REFKC{k}"):
                for f in sorted((S / "d570" / nm).glob("p*_t*.wav")):
                    m = re.match(r"(p\d+_t(\d+))_", f.name)
                    key, ti = m.group(1), int(m.group(2))
                    if ti not in ft:
                        continue
                    if key not in src:
                        src[key] = harvest(prep(load48(S / "d56" / f"{key}_SRC.wav")))
                    x = prep(load48(f))
                    if nm.startswith("OURS"):
                        r = align(src[key] * 2 ** (k / 12), harvest(x))
                    else:
                        fc = harvest(load48(S / "cen_reg" / f"cen_{ti:03d}.wav").astype(np.float64)[: 8 * SR])
                        fs = src[key]
                        r = fc * float(np.median(fs[fs > 0])) * 2 ** (k / 12) / float(np.median(fc[fc > 0]))
                    fo = float(np.median(r[r > 0]))
                    rows.setdefault((s, key, k), {})["tg"] = f"{s}|{ti}"
                    rows[(s, key, k)][nm[:4]] = {"oe": oe(x, r), "q": ft[ti] / fo}
                print(s, nm, flush=True)
    rng = np.random.default_rng(596)
    rep: dict = {"per_k": {}}

    def corr(sel: list, fld: str) -> dict:
        xs = np.array([pq(rows[kk][fld]["q"]) for kk in sel])
        ys = np.array([rows[kk]["OURS"]["oe"] - rows[kk]["REFK"]["oe"] for kk in sel]) if fld == "OURS" else np.array([rows[kk]["REFK"]["oe"] for kk in sel])
        tg = np.array([rows[kk]["tg"] for kk in sel])
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys, tg = xs[ok], ys[ok], tg[ok]
        ut = np.unique(tg)
        bs = []
        for _ in range(4000):
            idx = np.concatenate([np.nonzero(tg == t)[0] for t in rng.choice(ut, len(ut))])
            bs.append(spearmanr(xs[idx], ys[idx])[0])
        return {"n": int(len(xs)), "rho": round(float(spearmanr(xs, ys)[0]), 3), "ci95": [round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]}

    both = [kk for kk, v in rows.items() if "OURS" in v and "REFK" in v]
    for k in (5, 7, 12):
        sel = [kk for kk in both if kk[2] == k]
        rep["per_k"][k] = {"dOE_vs_Pq": corr(sel, "OURS"), "REFKC_OE_vs_Pq": corr(sel, "REFK"), "q_range": [round(min(rows[kk]["OURS"]["q"] for kk in sel), 2), round(max(rows[kk]["OURS"]["q"] for kk in sel), 2)]}
        print(k, rep["per_k"][k], flush=True)
    sel = [kk for kk in both if kk[2] in (5, 7)]
    rep["main_k5k7"] = corr(sel, "OURS")
    rep["control_k5k7"] = corr(sel, "REFK")
    bins = {}
    for kk in sel:
        q = rows[kk]["OURS"]["q"]
        b = "q<1.75" if q < 1.75 else "1.75-2.25" if q < 2.25 else "2.25-2.75" if q < 2.75 else "2.75-3.5" if q < 3.5 else "q>=3.5"
        bins.setdefault(b, []).append(rows[kk]["OURS"]["oe"] - rows[kk]["REFK"]["oe"])
    rep["dOE_by_q"] = {b: {"n": len(v), "mean": round(float(np.nanmean(v)), 4)} for b, v in sorted(bins.items())}
    print("main", rep["main_k5k7"], "control", rep["control_k5k7"], rep["dOE_by_q"], flush=True)
    rep["rows"] = [{"set": kk[0], "key": kk[1], "k": kk[2], **{f"{f}_{x}": rows[kk][f][x] for f in ("OURS", "REFK") for x in ("oe", "q")}} for kk in both]
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
