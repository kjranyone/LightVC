"""D593 rev1(推論だけ): 低い k の候補の出力で F1 がオクターブを読み違う性質 = 半分の高さの成分(SHR_r)か基音の弱さ(H1H2_r)か。
基準の高さ r = 候補は入力 SRC の harvest × 2^(k/12)(ファイルごとの遅れ ±10 フレーム)・REFKC は描画に使った輪郭。F1・harvest の読みを r と照合する。
    uv run python d593_subharm.py --sets dev2 dev3 --out ../results/conv_p0/d593/subharm.json
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


def spec_stats(x: np.ndarray, fv: np.ndarray, h: np.ndarray, r: np.ndarray) -> dict:
    _, _, Z = ss.stft(x, SR, nperseg=NFFT, noverlap=NFFT - 240)
    A = np.abs(Z)
    n = min(len(fv), len(h), len(r), A.shape[1])
    fv, h, r = fv[:n], h[:n], r[:n]
    bw = SR / NFFT
    out = {}
    for nm, rd in (("F1", fv), ("h", h)):
        b = (rd > 0) & (r > 0)
        e = 12 * np.log2(rd[b] / r[b])
        out[f"{nm}_down"] = int((e < -9).sum())
        out[f"{nm}_up"] = int((e > 9).sum())
        out[f"{nm}_nv"] = int(b.sum())
    out["agree_h"] = float((np.abs(12 * np.log2(h[(h > 0) & (r > 0)] / r[(h > 0) & (r > 0)])) < 1).mean()) if ((h > 0) & (r > 0)).sum() else float("nan")
    shr, h12 = [], []
    for t in np.nonzero((r > 0) & ((fv > 0) | (h > 0)))[0]:
        def amp(f: float) -> float:
            i = int(round(f / bw))
            return float(A[max(i - 1, 0): i + 2, t].max()) if i + 1 < A.shape[0] else 0.0

        hh = [amp(k * r[t]) for k in range(1, 9)]
        sb = [amp((k - 0.5) * r[t]) for k in range(1, 9)]
        shr.append(sum(sb) / max(sum(hh), 1e-12))
        h12.append(20 * np.log10((hh[0] + 1e-12) / (hh[1] + 1e-12)))
    out["shr"] = float(np.median(shr)) if shr else float("nan")
    out["h1h2"] = float(np.median(h12)) if h12 else float("nan")
    return out


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

    def align(r0: np.ndarray, h: np.ndarray) -> np.ndarray:
        best = (0, -1.0)
        for lag in range(-10, 11):
            r = np.roll(r0, lag)
            if lag > 0:
                r[:lag] = 0
            elif lag < 0:
                r[lag:] = 0
            n = min(len(r), len(h))
            b = (r[:n] > 0) & (h[:n] > 0)
            if b.sum() < 50:
                continue
            ag = float((np.abs(12 * np.log2(h[:n][b] / r[:n][b])) < 1).mean())
            if ag > best[1]:
                best = (lag, ag)
        r = np.roll(r0, best[0])
        if best[0] > 0:
            r[: best[0]] = 0
        elif best[0] < 0:
            r[best[0]:] = 0
        return r

    t = np.arange(8 * SR) / SR
    f0 = 190.0
    K = int(15000 // f0)

    def synth(sub: float, h1: float) -> np.ndarray:
        y = sum((h1 if k == 1 else 1.0) / k * np.cos(2 * np.pi * k * f0 * t) for k in range(1, K + 1))
        y = y + sum(sub / k * np.cos(2 * np.pi * (k - 0.5) * f0 * t) for k in range(1, K + 1))
        return (y / np.abs(y).max() * 0.5).astype(np.float32)

    s0 = {}
    for nm, sb, h1 in (("clean", 0.0, 1.0), ("doubling", 0.25, 1.0), ("noh1", 0.0, 0.1)):
        x = prep(synth(sb, h1))
        fv, h = f0track(x), harvest(x)
        s0[nm] = spec_stats(x.astype(np.float64), fv, h, np.full(len(h), f0))
    ok = s0["doubling"]["shr"] >= s0["clean"]["shr"] + 0.10 and s0["noh1"]["h1h2"] <= s0["clean"]["h1h2"] - 10 and s0["noh1"]["shr"] <= s0["clean"]["shr"] + 0.05
    print("S0", {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in s0.items()}, "PASS" if ok else "FAIL", flush=True)
    rep: dict = {"S0": s0, "S0_pass": bool(ok)}
    if not ok:
        Path(a.out).write_text(json.dumps(rep, indent=1))
        return 2
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
                    fv, h = f0track(x), harvest(x)
                    if nm.startswith("OURS"):
                        if key not in srcf:
                            srcf[key] = harvest(prep(load48(S / "d56" / f"{key}_SRC.wav")))
                        r = align(srcf[key] * 2 ** (k / 12), h)
                    else:
                        if key not in srcf:
                            srcf[key] = harvest(prep(load48(S / "d56" / f"{key}_SRC.wav")))
                        fc = harvest(load48(S / "cen_reg" / f"cen_{ti:03d}.wav").astype(np.float64)[: 8 * SR])
                        fs = srcf[key]
                        r = fc * float(np.median(fs[fs > 0])) * 2 ** (k / 12) / float(np.median(fc[fc > 0]))
                    rows[nm][(s, key)] = {"t": ti, **spec_stats(x.astype(np.float64), fv, h, r)}
                print(s, nm, len(rows[nm]), flush=True)
    ag12 = np.array([v["agree_h"] for v in rows["OURS12RY58G"].values()])
    al_ok = float((ag12 >= 0.6).mean()) >= 0.9
    rep["align_check"] = {"frac_files_agree_ge_0.6": round(float((ag12 >= 0.6).mean()), 3), "median_agree": round(float(np.median(ag12)), 3), "pass": al_ok}
    print("align", rep["align_check"], flush=True)
    if not al_ok:
        Path(a.out).write_text(json.dumps(rep, indent=1))
        return 3
    rep["sets"] = {}
    for nm, R0 in rows.items():
        R = list(R0.values())
        tot = {c: sum(r[c] for r in R) for c in ("F1_down", "F1_up", "F1_nv", "h_down", "h_up", "h_nv")}
        rep["sets"][nm] = {"n": len(R), "shr_median": round(float(np.nanmedian([r["shr"] for r in R])), 4), "h1h2_median": round(float(np.nanmedian([r["h1h2"] for r in R])), 2),
                           "F1_down_frac_of_voiced": round(tot["F1_down"] / max(tot["F1_nv"], 1), 4), "F1_up_frac_of_voiced": round(tot["F1_up"] / max(tot["F1_nv"], 1), 4),
                           "h_down_frac_of_voiced": round(tot["h_down"] / max(tot["h_nv"], 1), 4), "h_up_frac_of_voiced": round(tot["h_up"] / max(tot["h_nv"], 1), 4),
                           "agree_h_median": round(float(np.nanmedian([r["agree_h"] for r in R])), 3)}
        print(nm, rep["sets"][nm], flush=True)
    c = [r for k in (5, 7) for r in rows[f"OURS{k}RY58G"].values()]
    dn, upn = sum(r["F1_down"] for r in c), sum(r["F1_up"] for r in c)
    rep["D1"] = {"F1_oct_frames": dn + upn, "F1_down_frac": round(dn / max(dn + upn, 1), 3), "F1_up_frac": round(upn / max(dn + upn, 1), 3)}
    print("D1", rep["D1"], flush=True)
    rng = np.random.default_rng(593)
    rep["pairs"] = {}
    for k in (5, 7, 12):
        A, B = f"OURS{k}RY58G", f"REFKC{k}"
        ks = sorted(set(rows[A]) & set(rows[B]))
        tg = np.array([f"{key[0]}|{rows[A][key]['t']}" for key in ks])
        for st in ("shr", "h1h2", "h_down"):
            if st == "h_down":
                d = np.array([rows[A][key]["h_down"] / max(rows[A][key]["h_nv"], 1) - rows[B][key]["h_down"] / max(rows[B][key]["h_nv"], 1) for key in ks])
            else:
                d = np.array([rows[A][key][st] - rows[B][key][st] for key in ks])
            okm = np.isfinite(d)
            rep["pairs"][f"{A}-{B}:{st}"] = {"n": int(okm.sum()), "diff": round(float(d[okm].mean()), 4), "ci95": boot(d[okm], tg[okm], rng)}
            print(f"{A}-{B}:{st}", rep["pairs"][f"{A}-{B}:{st}"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
