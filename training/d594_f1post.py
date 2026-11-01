"""D594(推論だけ): 低い k の候補で F1 が倍を読む性質 = 事後のあいまいさ(P2/P1)・奇数倍音の不足(OE)・有声の端への集中。基準の高さ r は D593 rev1 と同じ。
    uv run python d594_f1post.py --sets dev2 dev3 --out ../results/conv_p0/d594/f1post.json
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

    def f0post(x: np.ndarray) -> tuple:
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
        return np.where(vo[:m] & (c >= 0.5), f, 0.0), pr[:, :m]

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

    def stats(x: np.ndarray, r: np.ndarray) -> dict:
        fv, pr = f0post(x)
        _, _, Z = ss.stft(x.astype(np.float64), SR, nperseg=NFFT, noverlap=NFFT - 240)
        A = np.abs(Z)
        n = min(len(fv), len(r), A.shape[1])
        fv, r, pr = fv[:n], r[:n], pr[:, :n]
        bw = SR / NFFT
        vb = (fv > 0) & (r > 0)
        e = np.where(vb, 12 * np.log2(np.where(vb, fv, 1.0) / np.where(vb, r, 1.0)), 0.0)
        up = vb & (e > 9)
        st = 12 * lhz[:, None] / np.log(2)
        amb, oe = [], np.full(n, np.nan)
        for t in np.nonzero(vb)[0]:
            s0 = 12 * np.log2(r[t])
            p1 = pr[np.abs(st[:, 0] - s0) <= 1, t].sum()
            p2 = pr[np.abs(st[:, 0] - s0 - 12) <= 1, t].sum()
            amb.append(p2 / max(p1, 1e-9) > 0.3)

            def amp(f: float) -> float:
                i = int(round(f / bw))
                return float(A[max(i - 1, 0): i + 2, t].max()) if i + 1 < A.shape[0] else 0.0

            od = sum(amp(k * r[t]) for k in (3, 5, 7))
            ev = sum(amp(k * r[t]) for k in (2, 4, 6, 8))
            oe[t] = od / max(ev, 1e-12)
        vo = fv > 0
        bnd = np.zeros(n, bool)
        ch = np.nonzero(vo[1:] != vo[:-1])[0]
        for c in ch:
            bnd[max(0, c - 2): c + 4] = True
        return {"amb": float(np.mean(amb)) if amb else float("nan"), "oe": float(np.nanmedian(oe[vb])) if vb.sum() else float("nan"),
                "oe_up": float(np.nanmedian(oe[up])) if up.sum() >= 5 else float("nan"), "oe_other": float(np.nanmedian(oe[vb & ~up])) if (vb & ~up).sum() else float("nan"),
                "n_up": int(up.sum()), "up_edge": int((up & bnd).sum()), "nv": int(vb.sum()), "v_edge": int((vb & bnd).sum())}

    rows: dict = {}
    for s in a.sets:
        S = H / s
        srcf: dict = {}
        for k in (5, 7, 12):
            for nm in (f"OURS{k}RY58G", f"REFKC{k}"):
                rows.setdefault(nm, {})
                for f in sorted((S / "d570" / nm).glob("p*_t*.wav")):
                    m = re.match(r"(p\d+_t(\d+))_", f.name)
                    key, ti = m.group(1), int(m.group(2))
                    x = prep(load48(f))
                    if key not in srcf:
                        srcf[key] = harvest(prep(load48(S / "d56" / f"{key}_SRC.wav")))
                    if nm.startswith("OURS"):
                        r = align(srcf[key] * 2 ** (k / 12), harvest(x))
                    else:
                        fc = harvest(load48(S / "cen_reg" / f"cen_{ti:03d}.wav").astype(np.float64)[: 8 * SR])
                        fs = srcf[key]
                        r = fc * float(np.median(fs[fs > 0])) * 2 ** (k / 12) / float(np.median(fc[fc > 0]))
                    rows[nm][(s, key)] = {"t": ti, **stats(x, r)}
                print(s, nm, len(rows[nm]), flush=True)
    rep: dict = {"sets": {}, "pairs": {}}
    for nm, R0 in rows.items():
        R = list(R0.values())
        nu, ue, nv, ve = (sum(r[c] for r in R) for c in ("n_up", "up_edge", "nv", "v_edge"))
        rep["sets"][nm] = {"n": len(R), "amb_median": round(float(np.nanmedian([r["amb"] for r in R])), 4), "amb_mean": round(float(np.nanmean([r["amb"] for r in R])), 4),
                           "oe_median": round(float(np.nanmedian([r["oe"] for r in R])), 4), "up_frames": nu, "up_edge_share": round(ue / max(nu, 1), 3), "voiced_edge_share": round(ve / max(nv, 1), 3)}
        print(nm, rep["sets"][nm], flush=True)
    c = [r for k in (5, 7) for r in rows[f"OURS{k}RY58G"].values()]
    nu, ue, nv, ve = (sum(r[q] for r in c) for q in ("n_up", "up_edge", "nv", "v_edge"))
    rep["P_edge"] = {"up_edge_share": round(ue / max(nu, 1), 3), "base_share": round(ve / max(nv, 1), 3), "ratio": round((ue / max(nu, 1)) / max(ve / max(nv, 1), 1e-9), 2)}
    print("P_edge", rep["P_edge"], flush=True)
    rng = np.random.default_rng(594)
    for k in (5, 7, 12):
        A_, B_ = f"OURS{k}RY58G", f"REFKC{k}"
        ks = sorted(set(rows[A_]) & set(rows[B_]))
        tg = np.array([f"{key[0]}|{rows[A_][key]['t']}" for key in ks])
        for stt in ("amb", "oe"):
            d = np.array([rows[A_][key][stt] - rows[B_][key][stt] for key in ks])
            okm = np.isfinite(d)
            rep["pairs"][f"{A_}-{B_}:{stt}"] = {"n": int(okm.sum()), "diff": round(float(d[okm].mean()), 4), "ci95": boot(d[okm], tg[okm], rng)}
            print(f"{A_}-{B_}:{stt}", rep["pairs"][f"{A_}-{B_}:{stt}"], flush=True)
    for k in (5, 7):
        A_ = f"OURS{k}RY58G"
        ks = [key for key in rows[A_] if np.isfinite(rows[A_][key]["oe_up"])]
        if len(ks) < 5:
            continue
        d = np.array([rows[A_][key]["oe_up"] - rows[A_][key]["oe_other"] for key in ks])
        tg = np.array([f"{key[0]}|{rows[A_][key]['t']}" for key in ks])
        rep["pairs"][f"{A_}:oe_up-oe_other"] = {"n": len(ks), "diff": round(float(d.mean()), 4), "ci95": boot(d, tg, rng)}
        print(f"{A_}:oe_up-oe_other", rep["pairs"][f"{A_}:oe_up-oe_other"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
