"""D592(推論だけ): gate_mf_v4 の G k5 / k7 の切り分け。
  --stage render: REFKC<k> = 目標本人の別の録音(<組>/cen_reg/cen_TTT.wav の先頭 8 s)を WORLD で再合成し、高さを操作者の SRC 8 s の f0 の中央値 × 2^(k/12) へ一律に移す(d584 の REFK と同じ作り・元の録音だけ違う)。
  --stage jump: d109 と同じ F1 + 確信度 0.5 の jump の事象を harvest と照合して分類し、ltas_d を帯ごとに分解、対の差 候補 − REFKC を目標でまとめたブートストラップで。
    uv run python d592_refkc.py --stage render --sets dev2 dev3
    uv run python d592_refkc.py --stage jump --sets dev2 dev3 --out ../results/conv_p0/d592/jump_ltas.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import scipy.signal as ss
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
SR = 48000
NPROC = int(os.environ.get("LVC_NPROC", "12"))
KS = (0, 5, 7, 12)


def harvest(x: np.ndarray) -> np.ndarray:
    f0, _ = pw.harvest(ss.resample_poly(x.astype(np.float64), 1, 3), 16000, f0_floor=50.0, f0_ceil=1100.0, frame_period=5.0)
    return f0


def one(job: tuple) -> list:
    from d6_unit_cov import load48
    S, key, ti = job
    od = S / "d570"
    xs = load48(S / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * SR]
    xt = load48(S / "cen_reg" / f"cen_{ti:03d}.wav").astype(np.float64)[: 8 * SR]
    f0 = harvest(xt)
    t = np.arange(len(f0)) * 0.005
    sp = pw.cheaptrick(xt, f0, t, SR)
    ap = pw.d4c(xt, f0, t, SR)
    mt = float(np.median(f0[f0 > 0]))
    fs = harvest(xs)
    mo = float(np.median(fs[fs > 0]))
    chk = []
    for k in KS:
        nm = f"REFKC{k}"
        o = od / nm / f"{key}_{nm}.wav"
        want = mo * 2 ** (k / 12)
        if not o.exists():
            y = pw.synthesize(f0 * want / mt, sp, ap, SR, 5.0)
            o.parent.mkdir(parents=True, exist_ok=True)
            sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), SR, subtype="PCM_16")
        fo = harvest(load48(o).astype(np.float64))
        n = min(len(fo), len(f0))
        a, b = fo[:n], f0[:n] * want / mt
        both = (a > 0) & (b > 0)
        chk.append((k, float(np.median(12 * np.log2(a[both] / b[both])))))
    return chk


def render(sets: list) -> int:
    jobs = []
    for s in sets:
        S = H / s
        ok = {m["ti"] for m in json.loads((S / "cen_reg" / "meta.json").read_text()) if m["ok"]}
        for r in json.loads((S / "devset.json").read_text())["pairs"]:
            key = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if r["ti"] in ok and (S / "d570" / "OURS7RY58G" / f"{key}_OURS7RY58G.wav").exists():
                jobs.append((S, key, r["ti"]))
    with Pool(NPROC) as pl:
        res = [c for r in pl.imap_unordered(one, jobs) for c in r]
    out = {}
    for k in KS:
        d = np.array([v for kk, v in res if kk == k])
        out[k] = {"n": int(len(d)), "within_0.5st": round(float((np.abs(d) <= 0.5).mean()), 3), "median_dev_st": round(float(np.median(d)), 3)}
        print(f"REFKC{k}", out[k], flush=True)
    good = all(v["within_0.5st"] >= 0.95 for v in out.values())
    print("self-check", "PASS" if good else "FAIL", "jobs", len(jobs), flush=True)
    return 0 if good else 2


def boot(d: np.ndarray, tg: np.ndarray, rng) -> list:
    ut = np.unique(tg)
    by = {t: d[tg == t] for t in ut}
    bs = [np.concatenate([by[t] for t in rng.choice(ut, len(ut))]).mean() for _ in range(4000)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def jump(sets: list, out: str) -> int:
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
    fps = SR / HOP
    cf = 1000.0 * 2.0 ** (np.arange(-10, 11) / 3.0)
    cf = cf[(cf >= 100) & (cf <= 10000)]

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

    def active(x: np.ndarray) -> np.ndarray:
        n = len(x) // HOP
        e = 10 * np.log10((x[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-12)
        return e > e.max() - 40

    def ltas(x: np.ndarray, act: np.ndarray) -> np.ndarray:
        fq, _, Z = ss.stft(x, SR, nperseg=2048, noverlap=2048 - HOP)
        P = np.abs(Z[:, : len(act)]) ** 2
        P = P[:, act[: P.shape[1]]].mean(1)
        o = np.array([P[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() for c in cf])
        d = 10 * np.log10(o + 1e-20)
        return d - d.mean()

    refl: dict = {}

    def refL(S: Path, t: int) -> np.ndarray:
        if (S, t) not in refl:
            r = load48(S / "d37/_work" / f"ref_{t:03d}.wav")[: int(30 * SR)].astype(np.float32)
            refl[(S, t)] = ltas(r, active(r))
        return refl[(S, t)]

    names = [f"OURS{k}RY58G" for k in (5, 7, 12)] + [f"REFKC{k}" for k in KS] + [f"REFK{k}" for k in (5, 7, 12)] + ["REFW", "GU5_a112", "GU7_a112", "GU12_a120"]
    rows: dict = {nm: {} for nm in names}
    for s in sets:
        S = H / s
        for nm in names:
            for f in sorted((S / "d570" / nm).glob("p*_t*.wav")):
                m = re.match(r"(p\d+_t(\d+))_", f.name)
                t = int(m.group(2))
                x = load48(f)[: int(8 * SR)].astype(np.float32)
                x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
                fv = f0track(x)
                act = active(x)[: len(fv)]
                fv = fv[: len(act)]
                v = fv > 0
                st = 12 * np.log2(np.where(v, fv, 1.0))
                both = v[1:] & v[:-1]
                dst = np.abs(np.diff(st))
                ev = np.nonzero(both & (dst > 7.2))[0]
                h = harvest(x)
                h = np.r_[h, np.zeros(max(0, len(fv) + 4 - len(h)))]
                cls = []
                for e in ev:
                    hb, ha = h[max(0, e - 2): e + 1], h[e + 1: e + 4]
                    if (hb > 0).sum() == 0 or (ha > 0).sum() == 0:
                        cls.append("h_unv")
                        continue
                    dh = abs(12 * np.log2(np.median(ha[ha > 0]) / np.median(hb[hb > 0])))
                    cls.append("tracker" if dh < 2 else "real" if dh > 7.2 else "mid")
                hv = h[: len(fv)]
                bv = v & (hv > 0)
                octf = float((np.abs(12 * np.log2(fv[bv] / hv[bv])) > 9).mean()) if bv.sum() else float("nan")
                db = ltas(x, act) - refL(S, t)
                rows[nm][(s, m.group(1))] = {"t": t, "jump": float(len(ev) / max(v.sum() / fps, 1e-3)), "cls": cls, "oct_frac": octf, "d_b": db.tolist(),
                                            "ltas_d": float(np.sqrt((db ** 2).mean())), "ltas_lo": float(np.sqrt((db[cf < 500] ** 2).mean())), "ltas_hi": float(np.sqrt((db[cf >= 500] ** 2).mean()))}
            print(s, nm, len(rows[nm]), flush=True)
    rng = np.random.default_rng(592)
    rep: dict = {"bands_hz": [round(float(c), 1) for c in cf], "sets": {}, "pairs": {}}
    for nm in names:
        R = list(rows[nm].values())
        if not R:
            continue
        cls = [c for r in R for c in r["cls"]]
        D = np.array([r["d_b"] for r in R])
        sq = (D ** 2).mean(0)
        rep["sets"][nm] = {"n": len(R), "jump_median": round(float(np.median([r["jump"] for r in R])), 3), "jump_mean": round(float(np.mean([r["jump"] for r in R])), 3),
                           "events": len(cls), "class_frac": {c: round(cls.count(c) / max(len(cls), 1), 3) for c in ("tracker", "real", "mid", "h_unv")},
                           "oct_frac_median": round(float(np.nanmedian([r["oct_frac"] for r in R])), 4), "oct_frac_mean": round(float(np.nanmean([r["oct_frac"] for r in R])), 4),
                           "ltas_d_median": round(float(np.median([r["ltas_d"] for r in R])), 3), "ltas_lo_median": round(float(np.median([r["ltas_lo"] for r in R])), 3),
                           "ltas_hi_median": round(float(np.median([r["ltas_hi"] for r in R])), 3), "lo_share_of_sq": round(float(sq[cf < 500].sum() / sq.sum()), 3),
                           "mean_d_b": [round(float(v), 2) for v in D.mean(0)]}
        print(nm, {k: v for k, v in rep["sets"][nm].items() if k != "mean_d_b"}, flush=True)
    for k in (5, 7, 12):
        A, B = f"OURS{k}RY58G", f"REFKC{k}"
        ks = sorted(set(rows[A]) & set(rows[B]))
        tg = np.array([f"{key[0]}|{rows[A][key]['t']}" for key in ks])
        for stat in ("ltas_d", "ltas_lo", "ltas_hi", "jump", "oct_frac"):
            d = np.array([rows[A][key][stat] - rows[B][key][stat] for key in ks])
            ok = np.isfinite(d)
            rep["pairs"][f"{A}-{B}:{stat}"] = {"n": int(ok.sum()), "diff": round(float(d[ok].mean()), 4), "ci95": boot(d[ok], tg[ok], rng)}
            print(f"{A}-{B}:{stat}", rep["pairs"][f"{A}-{B}:{stat}"], flush=True)
    Path(out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["render", "jump"], required=True)
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    return render(a.sets) if a.stage == "render" else jump(a.sets, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
