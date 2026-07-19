"""Hunt for objective metrics that separate world > ltv the way ears do.

Ear verdict (2026-07-12): ltv audibly worse than world vs GT, but
contrast_ratio (1-5k peak-median) is blind (0.977 vs 0.980). This script runs
a battery of candidate discriminators on {gt, world, ltv} x golden utts and
ranks them by sign-consistency (does world beat ltv on all utts?) and gap.
Winners get registered into the E0 gate.

Usage: cd training && uv run python e0_discriminator_hunt.py
Output: results/e0_discriminator_hunt.json + stdout table
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw

SR = 44100
HOP = 512
EPS = 1e-8
ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results/e0_oracle_ltv"
BANDS = [("b0_1k", 0, 1000), ("b1_4k", 1000, 4000),
         ("b4_8k", 4000, 8000), ("b8_16k", 8000, 16000)]


def load(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR)
    return y


def stft_mag(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    return s, np.fft.rfftfreq(2048, 1.0 / SR)


def frame_ctx(gt: np.ndarray) -> dict:
    xd = gt.astype(np.float64)
    f0, t = pw.harvest(xd, SR, frame_period=1000.0 * HOP / SR)
    f0 = pw.stonemask(xd, f0, t, SR)
    n = min(len(f0), len(gt) // HOP)
    rms = np.sqrt((gt[:n * HOP].reshape(n, HOP) ** 2).mean(-1))
    act = rms > 0.05 * np.percentile(rms, 99)
    return {"f0": f0[:n], "act": act, "n": n}


def band_env(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    s, f = stft_mag(y)
    return np.sqrt((s[(f >= lo) & (f < hi)] ** 2).sum(0) + EPS)


def m_band_traj_dist(y: np.ndarray, gt: np.ndarray, ctx: dict) -> dict:
    out = {}
    for name, lo, hi in BANDS:
        ey = np.log(band_env(y, lo, hi) + EPS)
        eg = np.log(band_env(gt, lo, hi) + EPS)
        nf = min(len(ey), len(eg), ctx["n"])
        a = ctx["act"][:nf]
        out[f"trajdist_{name}"] = float(np.abs(ey[:nf][a] - eg[:nf][a]).mean())
    return out


def m_hf_contrast(y: np.ndarray, gt: np.ndarray, ctx: dict) -> dict:
    out = {}
    for name, lo, hi in [("c5_10k", 5000, 10000), ("c10_16k", 10000, 16000)]:
        def c(x):
            s, f = stft_mag(x)
            b = np.log(s[(f >= lo) & (f < hi)] + EPS)
            nf = min(b.shape[1], ctx["n"])
            bb = b[:, :nf][:, ctx["act"][:nf]]
            return (bb.max(0) - np.median(bb, 0)).mean() if bb.shape[1] > 2 else np.nan
        out[name] = float(c(y) / (c(gt) + EPS))
    return out


def _subenv(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos_y = librosa.stft(y, n_fft=2048, hop_length=64)
    f = np.fft.rfftfreq(2048, 1.0 / SR)
    return np.sqrt((np.abs(sos_y[(f >= lo) & (f < hi)]) ** 2).sum(0) + EPS)


def m_f0_am_depth(y: np.ndarray, gt: np.ndarray, ctx: dict) -> dict:
    env = _subenv(y, 2000, 12000)
    fs_env = SR / 64.0
    f0m = np.median(ctx["f0"][ctx["f0"] > 1]) if (ctx["f0"] > 1).any() else 0.0
    if f0m <= 0:
        return {"f0_am_depth": np.nan}
    n8 = min(len(env), int(fs_env * 8))
    e = np.log(env[:n8])
    e = e - e.mean()
    spec = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    fm = np.fft.rfftfreq(len(e), 1.0 / fs_env)
    band = (fm > f0m * 0.8) & (fm < f0m * 1.25)
    ref = (fm > 30) & (fm < 300)
    if band.sum() == 0 or ref.sum() == 0:
        return {"f0_am_depth": np.nan}
    val = spec[band].max() / (np.median(spec[ref]) + EPS)
    env_g = _subenv(gt, 2000, 12000)
    eg = np.log(env_g[:n8])
    eg = eg - eg.mean()
    sg = np.abs(np.fft.rfft(eg * np.hanning(len(eg))))
    vg = sg[band].max() / (np.median(sg[ref]) + EPS)
    return {"f0_am_depth": float(val), "f0_am_depth_gt": float(vg),
            "f0_am_ratio": float(val / (vg + EPS))}


def m_crest(y: np.ndarray, gt: np.ndarray, ctx: dict) -> dict:
    def crest(x):
        n = min(len(x) // HOP, ctx["n"])
        seg = x[:n * HOP].reshape(n, HOP)
        v = ctx["act"][:n] & (ctx["f0"][:n] > 1)
        s = seg[v]
        if len(s) < 3:
            return np.nan
        return float(np.median(s.max(-1) / (np.sqrt((s ** 2).mean(-1)) + EPS)))
    return {"crest_ratio": float(crest(y) / (crest(gt) + EPS))}


def m_cpp(y: np.ndarray, gt: np.ndarray, ctx: dict) -> dict:
    def cpp(x):
        n = min(len(x) // HOP, ctx["n"])
        vals = []
        for i in np.where(ctx["act"][:n] & (ctx["f0"][:n] > 1))[0]:
            seg = x[i * HOP:i * HOP + 2048]
            if len(seg) < 2048:
                continue
            sp = np.log(np.abs(np.fft.rfft(seg * np.hanning(2048))) + EPS)
            ceps = np.abs(np.fft.rfft(sp - sp.mean()))
            q = np.arange(len(ceps))
            lo, hi = int(2048 * ctx["f0"][i] / SR * 0.8), int(2048 * ctx["f0"][i] / SR * 1.3)
            lo = max(lo, 2)
            if hi <= lo:
                continue
            vals.append(float(np.log(ceps[lo:hi].max() + EPS) - np.log(np.median(ceps[2:]) + EPS)))
        return np.mean(vals) if vals else np.nan
    return {"cpp_delta": float(cpp(y) - cpp(gt))}


def m_mod_spec_dist(y: np.ndarray, gt: np.ndarray, ctx: dict) -> dict:
    def mspec(x, lo, hi):
        env = np.log(_subenv(x, lo, hi))
        fs_env = SR / 64.0
        n8 = min(len(env), int(fs_env * 8))
        e = env[:n8] - env[:n8].mean()
        spec = np.abs(np.fft.rfft(e * np.hanning(len(e)))) + EPS
        fm = np.fft.rfftfreq(len(e), 1.0 / fs_env)
        sel = (fm >= 2) & (fm <= 400)
        s = np.log(spec[sel])
        return s - s.mean()
    out = {}
    for name, lo, hi in [("mod_2_8k", 2000, 8000), ("mod_8_16k", 8000, 16000)]:
        out[f"{name}_dist"] = float(np.abs(mspec(y, lo, hi) - mspec(gt, lo, hi)).mean())
    return out


def m_lsd_bands(y: np.ndarray, gt: np.ndarray, ctx: dict) -> dict:
    sy, f = stft_mag(y)
    sg, _ = stft_mag(gt)
    nf = min(sy.shape[1], sg.shape[1], ctx["n"])
    a = ctx["act"][:nf]
    out = {}
    for name, lo, hi in BANDS:
        sel = (f >= lo) & (f < hi)
        ly = np.log(sy[sel][:, :nf][:, a] + EPS)
        lg = np.log(sg[sel][:, :nf][:, a] + EPS)
        out[f"lsd_{name}"] = float(np.abs(ly - lg).mean())
    return out


def m_flux_dist(y: np.ndarray, gt: np.ndarray, ctx: dict) -> dict:
    def flux(x):
        s, _ = stft_mag(x)
        l = np.log(s + EPS)
        return np.abs(np.diff(l, axis=1)).mean(0)
    fy, fg = flux(y), flux(gt)
    nf = min(len(fy), len(fg), ctx["n"] - 1)
    a = ctx["act"][:nf]
    return {"flux_dist": float(np.abs(fy[:nf][a] - fg[:nf][a]).mean()),
            "flux_ratio": float(fy[:nf][a].mean() / (fg[:nf][a].mean() + EPS))}


def m_am_rough(y: np.ndarray, gt: np.ndarray, ctx: dict) -> dict:
    def rough(x, lo, hi):
        env = np.log(_subenv(x, lo, hi))
        fs_env = SR / 64.0
        e = env - env.mean()
        spec = np.abs(np.fft.rfft(e * np.hanning(len(e)))) + EPS
        fm = np.fft.rfftfreq(len(e), 1.0 / fs_env)
        band = (fm >= 20) & (fm <= 80)
        ref = (fm >= 2) & (fm <= 400)
        return float(spec[band].mean() / (spec[ref].mean() + EPS))
    out = {}
    for name, lo, hi in [("rough_1_4k", 1000, 4000), ("rough_4_12k", 4000, 12000)]:
        out[name] = float(rough(y, lo, hi) / (rough(gt, lo, hi) + EPS))
    return out


def m_mod_low(y: np.ndarray, gt: np.ndarray, ctx: dict) -> dict:
    def mspec(x, lo, hi):
        env = np.log(_subenv(x, lo, hi))
        fs_env = SR / 64.0
        n8 = min(len(env), int(fs_env * 8))
        e = env[:n8] - env[:n8].mean()
        spec = np.abs(np.fft.rfft(e * np.hanning(len(e)))) + EPS
        fm = np.fft.rfftfreq(len(e), 1.0 / fs_env)
        sel = (fm >= 2) & (fm <= 400)
        s = np.log(spec[sel])
        return s - s.mean()
    out = {}
    for name, lo, hi in [("mod_0_2k", 100, 2000), ("mod_2_5k", 2000, 5000)]:
        out[f"{name}_dist"] = float(np.abs(mspec(y, lo, hi) - mspec(gt, lo, hi)).mean())
    return out


LSHARP_BUCKETS = [("k1_5", 1, 5), ("k6_15", 6, 15), ("k16_30", 16, 30),
                  ("k31_60", 31, 60)]


def _line_buckets(x: np.ndarray, f0: np.ndarray) -> dict:
    S = np.log(np.abs(librosa.stft(x, n_fft=4096, hop_length=HOP)) + EPS)
    T = min(S.shape[1], len(f0))
    acc = {k: [] for k, _, _ in LSHARP_BUCKETS}
    for i in range(T):
        f = f0[i]
        if f <= 1.0:
            continue
        for name, k0, k1 in LSHARP_BUCKETS:
            ks = np.arange(k0, min(k1 + 1, int(SR / 2 / f)))
            if len(ks) < 2:
                continue
            pb = np.clip((ks * f / SR * 4096).round().astype(int), 0, 2048)
            vb = np.clip(((ks + 0.5) * f / SR * 4096).round().astype(int), 0, 2048)
            acc[name].append(float(S[pb, i].mean() - S[vb, i].mean()))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


def m_line_sharp(y: np.ndarray, gt: np.ndarray, ctx: dict) -> dict:
    by = _line_buckets(y, ctx["f0"])
    bg = _line_buckets(gt, ctx["f0"])
    out = {}
    devs = []
    for k, _, _ in LSHARP_BUCKETS:
        if np.isfinite(by[k]) and np.isfinite(bg[k]):
            out[f"lsharp_{k}"] = round(by[k], 3)
            out[f"lsharp_{k}_gt"] = round(bg[k], 3)
            devs.append(abs(by[k] - bg[k]))
    out["lsharp_dev"] = round(float(np.mean(devs)), 3) if devs else float("nan")
    return out


METRICS = [m_band_traj_dist, m_hf_contrast, m_f0_am_depth, m_crest, m_cpp,
           m_mod_spec_dist, m_lsd_bands, m_flux_dist, m_am_rough, m_mod_low,
           m_line_sharp]


def main() -> None:
    uids = sorted(w.stem[:-3] for w in DIR.glob("*_gt.wav"))
    rows: dict = {}
    for uid in uids:
        gt = load(DIR / f"{uid}_gt.wav")
        ctx = frame_ctx(gt)
        rows[uid] = {}
        for role, fname in [("world", f"{uid}_world.wav"),
                            ("ltv", f"{uid}_ctq30_k1024_nb1025_od_qp4.wav")]:
            y = load(DIR / fname)
            m = {}
            for fn in METRICS:
                m.update(fn(y[:len(gt)], gt[:len(y)], ctx))
            rows[uid][role] = {k: (round(v, 4) if np.isfinite(v) else None)
                               for k, v in m.items()}
    keys = sorted({k for u in rows.values() for k in u["world"]})
    print(f"{'metric':22s} {'world_better':>12} {'mean_world':>11} {'mean_ltv':>9}")
    ranking = []
    lower_better = ("trajdist", "dist", "cpp_delta", "lsd")
    for k in keys:
        wv = [rows[u]["world"][k] for u in uids if rows[u]["world"].get(k) is not None]
        lv = [rows[u]["ltv"][k] for u in uids if rows[u]["ltv"].get(k) is not None]
        if len(wv) != len(lv) or not wv:
            continue
        lb = any(t in k for t in lower_better)
        wins = sum((w < l) if lb else (abs(w - 1) < abs(l - 1)) if ("ratio" in k or k.startswith("c")) else (w > l)
                   for w, l in zip(wv, lv))
        ranking.append((k, wins, len(wv), float(np.mean(wv)), float(np.mean(lv))))
        print(f"{k:22s} {wins:>7}/{len(wv):<4} {np.mean(wv):>11.3f} {np.mean(lv):>9.3f}")
    out = {"per_utt": rows,
           "ranking": [{"metric": k, "world_better": w, "n": n,
                        "mean_world": round(mw, 4), "mean_ltv": round(ml, 4)}
                       for k, w, n, mw, ml in ranking]}
    (ROOT / "results/e0_discriminator_hunt.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"-> {ROOT / 'results/e0_discriminator_hunt.json'}")


if __name__ == "__main__":
    main()
