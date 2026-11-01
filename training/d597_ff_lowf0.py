"""D597(推論だけ): (A) R の OE への寄与(dev2 / dev3 の OURS{5,7}Y58 対 OURS{5,7}RY58G)・(B) devff2 の k = 0 描画 OURS0Y58 で f0 の帯ごとの ΔOE(出力 − 入力)と F1 倍読みの超過。
    uv run python d597_ff_lowf0.py --out ../results/conv_p0/d597/ff_lowf0.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import H, SR, boot, harvest

ROOT = Path(__file__).resolve().parent.parent
NFFT = 2048
BINS = ((0, 210), (210, 260), (260, 330), (330, 5000))


def main() -> int:
    ap = argparse.ArgumentParser()
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

    def frames(x: np.ndarray, r: np.ndarray) -> tuple:
        fv = f0track(x)
        _, _, Z = ss.stft(x.astype(np.float64), SR, nperseg=NFFT, noverlap=NFFT - 240)
        A = np.abs(Z)
        n = min(len(fv), len(r), A.shape[1])
        fv, r = fv[:n], r[:n]
        bw = SR / NFFT
        oe = np.full(n, np.nan)
        vb = (fv > 0) & (r > 0)
        for t in np.nonzero(vb)[0]:
            def amp(f: float) -> float:
                i = int(round(f / bw))
                return float(A[max(i - 1, 0): i + 2, t].max()) if i + 1 < A.shape[0] else 0.0
            oe[t] = sum(amp(k * r[t]) for k in (3, 5, 7)) / max(sum(amp(k * r[t]) for k in (2, 4, 6, 8)), 1e-12)
        up = vb & (12 * np.log2(np.where(vb, fv, 1.0) / np.where(r > 0, r, 1.0)) > 9)
        return r, vb, oe, up

    rng = np.random.default_rng(597)
    rep: dict = {"A_R": {}, "B_FF": {}}
    for k in (5, 7):
        rows: dict = {}
        for s in ("dev2", "dev3"):
            S = H / s
            for nm in (f"OURS{k}Y58", f"OURS{k}RY58G"):
                for f in sorted((S / "d570" / nm).glob("p*_t*.wav")):
                    m = re.match(r"(p\d+_t(\d+))_", f.name)
                    x = prep(load48(f))
                    hs = harvest(prep(load48(S / "d56" / f"{m.group(1)}_SRC.wav")))
                    r = align(hs * 2 ** (k / 12), harvest(x))
                    _, vb, oe, _ = frames(x, r)
                    rows.setdefault((s, m.group(1)), {"tg": f"{s}|{int(m.group(2))}"})[nm] = float(np.nanmedian(oe[vb])) if vb.sum() else float("nan")
        ks = [kk for kk, v in rows.items() if f"OURS{k}Y58" in v and f"OURS{k}RY58G" in v]
        d = np.array([rows[kk][f"OURS{k}RY58G"] - rows[kk][f"OURS{k}Y58"] for kk in ks])
        tg = np.array([rows[kk]["tg"] for kk in ks])
        ok = np.isfinite(d)
        rep["A_R"][k] = {"n": int(ok.sum()), "RY58G_minus_Y58": round(float(d[ok].mean()), 4), "ci95": boot(d[ok], tg[ok], rng)}
        print("A_R", k, rep["A_R"][k], flush=True)
    S = H / "devff2"
    fb: dict = {}
    for f in sorted((S / "d570" / "OURS0Y58").glob("p*_t*.wav")):
        key = re.match(r"(p\d+_t\d+)_", f.name).group(1)
        xi = prep(load48(S / "d56" / f"{key}_SRC.wav"))
        hs = harvest(xi)
        xo = prep(load48(f))
        ro, vbo, oeo, upo = frames(xo, align(hs, harvest(xo)))
        ri, vbi, oei, upi = frames(xi, hs)
        for lo, hi in BINS:
            bo = vbo & (ro >= lo) & (ro < hi)
            bi = vbi & (ri >= lo) & (ri < hi)
            if bo.sum() >= 20 and bi.sum() >= 20:
                fb.setdefault(f"{lo}-{hi}", {})[key] = {"dOE": float(np.nanmedian(oeo[bo]) - np.nanmedian(oei[bi])), "oe_out": float(np.nanmedian(oeo[bo])), "oe_in": float(np.nanmedian(oei[bi])),
                                                        "up_ex": float(upo[bo].mean() - upi[bi].mean()), "up_out": float(upo[bo].mean()), "up_in": float(upi[bi].mean()), "n": int(bo.sum())}
        print("ff", key, flush=True)
    for b, v in fb.items():
        R = list(v.values())
        rep["B_FF"][b] = {"files": len(R), "frames": int(sum(r["n"] for r in R)), "dOE_mean": round(float(np.mean([r["dOE"] for r in R])), 4),
                          "oe_out": round(float(np.mean([r["oe_out"] for r in R])), 4), "oe_in": round(float(np.mean([r["oe_in"] for r in R])), 4),
                          "up_out": round(float(np.mean([r["up_out"] for r in R])), 4), "up_in": round(float(np.mean([r["up_in"] for r in R])), 4),
                          "dOE_ci95": boot(np.array([r["dOE"] for r in R]), np.array(list(v.keys())), rng)}
        print("B", b, rep["B_FF"][b], flush=True)
    lo, hi = fb.get("0-210", {}), fb.get("330-5000", {})
    files = sorted(set(lo) | set(hi))
    bs_oe, bs_up = [], []
    for _ in range(4000):
        pick = rng.choice(files, len(files))
        L = [lo[f] for f in pick if f in lo]
        Hh = [hi[f] for f in pick if f in hi]
        if L and Hh:
            bs_oe.append(np.mean([r["dOE"] for r in L]) - np.mean([r["dOE"] for r in Hh]))
            bs_up.append(np.mean([r["up_ex"] for r in L]) - np.mean([r["up_ex"] for r in Hh]))
    pt_oe = float(np.mean([r["dOE"] for r in lo.values()]) - np.mean([r["dOE"] for r in hi.values()])) if lo and hi else float("nan")
    pt_up = float(np.mean([r["up_ex"] for r in lo.values()]) - np.mean([r["up_ex"] for r in hi.values()])) if lo and hi else float("nan")
    rep["B_low_vs_high"] = {"files_low": len(lo), "files_high": len(hi), "dOE_low_minus_high": round(pt_oe, 4), "ci95": [round(float(np.percentile(bs_oe, 2.5)), 4), round(float(np.percentile(bs_oe, 97.5)), 4)],
                            "upex_low_minus_high": round(pt_up, 4), "upex_ci95": [round(float(np.percentile(bs_up, 2.5)), 4), round(float(np.percentile(bs_up, 97.5)), 4)]}
    print("B_low_vs_high", rep["B_low_vs_high"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
