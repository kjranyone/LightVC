import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy import signal

SR = 48000
HOP = 48
BANDS = {"all": (100, 10000), "lo": (100, 1000), "mid": (1000, 4000), "hi": (4000, 10000)}
MAXLAG = 80


def band_env(w: np.ndarray) -> dict[str, np.ndarray]:
    f, _, s = signal.stft(w, SR, nperseg=240, noverlap=240 - HOP, nfft=512, boundary="zeros", padded=True)
    p = np.abs(s) ** 2
    return {k: 10 * np.log10(p[(f >= a) & (f < b)].sum(0) + 1e-12) for k, (a, b) in BANDS.items()}


def highpass(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(n) / n
    return x - np.convolve(x, k, mode="same")


def xcorr_lag(a: np.ndarray, b: np.ndarray, m: np.ndarray, lo: int = -20, hi: int = MAXLAG) -> tuple[float, float]:
    n = min(len(a), len(b))
    a, b, m = a[:n], b[:n], m[:n]
    idx = np.nonzero(m)[0]
    idx = idx[(idx + hi < n) & (idx + lo >= 0)]
    av = a[idx] - a[idx].mean()
    r = []
    for l in range(lo, hi + 1):
        bv = b[idx + l]
        bv = bv - bv.mean()
        r.append(float((av * bv).sum() / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12)))
    r = np.array(r)
    j = int(r.argmax())
    d = 0.0
    if 0 < j < len(r) - 1:
        den = r[j - 1] - 2 * r[j] + r[j + 1]
        d = 0.5 * (r[j - 1] - r[j + 1]) / den if den != 0 else 0.0
    return lo + j + d, float(r[j])


def env_lags(x: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    ex, ey = band_env(x), band_env(y)
    act = ex["all"] > ex["all"].max() - 35
    out = {}
    for k in BANDS:
        l, r = xcorr_lag(highpass(ex[k], 200), highpass(ey[k], 200), act)
        out[k] = [round(l, 2), round(r, 3)]
    return out


def f0(w: np.ndarray) -> np.ndarray:
    w16 = librosa.resample(w.astype(np.float32), orig_sr=SR, target_sr=16000)
    f, v, _ = librosa.pyin(w16, fmin=60, fmax=700, sr=16000, frame_length=1024, hop_length=32, center=True)
    return np.where(v & np.isfinite(f), np.log(np.nan_to_num(f, nan=1.0)), np.nan)


def f0_lag(fx: np.ndarray, fy: np.ndarray) -> list[float]:
    n = min(len(fx), len(fy))
    fx, fy = fx[:n], fy[:n]
    best = (-1.0, 0)
    rs = []
    lags = list(range(-10, 41))
    for l in lags:
        a = fx[10 : n - 41]
        b = fy[10 + l : n - 41 + l]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 200:
            rs.append(-1.0)
            continue
        da, db = np.diff(a), np.diff(b)
        mm = m[1:] & m[:-1]
        da, db = da[mm], db[mm]
        r = float((da * db).sum() / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-12))
        rs.append(r)
        if r > best[0]:
            best = (r, l)
    j = lags.index(best[1])
    d = 0.0
    if 0 < j < len(rs) - 1:
        den = rs[j - 1] - 2 * rs[j] + rs[j + 1]
        d = 0.5 * (rs[j - 1] - rs[j + 1]) / den if den != 0 else 0.0
    return [round((best[1] + d) * 2.0, 2), round(best[0], 3)]


def onset_lag(x: np.ndarray, y: np.ndarray) -> list[float]:
    ex, ey = band_env(x)["all"], band_env(y)["all"]
    n = min(len(ex), len(ey))
    ex, ey = ex[:n], ey[:n]
    ax = ex > ex.max() - 35
    tx = np.percentile(ex[ax], 90) - 25
    ty = np.percentile(ey[ax], 90) - 25
    up = ex > tx
    d = []
    i = 300
    while i < n - 200:
        if up[i] and not up[i - 300 : i].any():
            seg = ey[max(0, i - 20) : i + 150] > ty
            if seg.any():
                d.append(int(seg.argmax()) - min(20, i))
            i += 300
        else:
            i += 1
    if not d:
        return [float("nan"), 0]
    return [float(np.median(d)), float(np.percentile(d, 25)), float(np.percentile(d, 75)), len(d)]


def stage_lags(x: np.ndarray, trace: Path) -> dict[str, list[float]]:
    rows = list(csv.DictReader(open(trace)))
    sc = np.array([float(r["src_c0"]) for r in rows])
    oc = np.array([float(r["out_c0"]) for r in rows])
    op = np.array([int(r["open"]) for r in rows]) > 0
    ex = band_env(x)["all"]
    t_ms = (np.arange(len(sc)) + 1) * 5.0
    g = np.arange(len(ex), dtype=float)
    su = np.interp(g, t_ms, sc)
    ou = np.interp(g, t_ms, oc)
    act = ex > ex.max() - 35
    a, ra = xcorr_lag(highpass(ex, 200), highpass(su, 200), act)
    b, rb = xcorr_lag(highpass(su, 200), highpass(ou, 200), act, lo=-20, hi=40)
    return {"front_c0_vs_in": [round(a, 2), round(ra, 3)], "model_c0_vs_front_c0": [round(b, 2), round(rb, 3)], "open_frac": [round(float(op.mean()), 3)]}


def run(root: Path, d: Path, tg: Path, tag: str, env: dict[str, str]) -> np.ndarray:
    out = d / f"p96_{tag}.f32"
    if not out.exists():
        e = dict(os.environ)
        e.update(env)
        subprocess.run([str(root / "target/release/examples/snv_trace"), str(d / "in.f32"), str(out), str(d / f"p96_{tag}.csv"), "0", str(d / "op"), str(tg)], check=True, cwd=root, env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.fromfile(out, dtype=np.float32)


def main() -> None:
    root, work, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    res: dict = {"speakers": {}}
    for sp in sys.argv[4:]:
        d = work / sp
        x = np.fromfile(d / "in.f32", dtype=np.float32)
        r: dict = {}
        delayed = np.concatenate([np.zeros(480, np.float32), x])[: len(x)]
        fx = f0(x)
        r["calib_10ms"] = {"env": env_lags(x, delayed), "f0": f0_lag(fx, f0(delayed)), "onset": onset_lag(x, delayed)}
        med = float(np.exp(np.nanmedian(fx)))
        r["src_f0_median_hz"] = round(med, 1)
        r["src_T0_ms"] = round(1000.0 / med, 2)
        for tag, env in [("copy_noesm", {"LVC_COPY": "1", "LVC_NO_GUARD": "1"}), ("copy", {"LVC_COPY": "1", "LVC_NO_GUARD": "1", "LVC_ESM": "1"}), ("conv", {"LVC_ESM": "1"})]:
            y = run(root, d, work / "tg", tag, env)
            n = min(len(x), len(y))
            r[tag] = {"env": env_lags(x[:n], y[:n]), "f0": f0_lag(fx, f0(y[:n])), "onset": onset_lag(x[:n], y[:n]), "stage": stage_lags(x[:n], d / f"p96_{tag}.csv")}
        res["speakers"][sp] = r
        print(sp, json.dumps(r, ensure_ascii=False), flush=True)
    summ: dict = {}
    for tag in ["calib_10ms", "copy_noesm", "copy", "conv"]:
        rows = [res["speakers"][s][tag] for s in res["speakers"]]
        summ[tag] = {f"env_{k}": float(np.median([q["env"][k][0] for q in rows])) for k in BANDS}
        summ[tag]["f0"] = float(np.median([q["f0"][0] for q in rows]))
        summ[tag]["onset"] = float(np.median([q["onset"][0] for q in rows]))
        if tag != "calib_10ms":
            summ[tag]["front_c0_vs_in"] = float(np.median([q["stage"]["front_c0_vs_in"][0] for q in rows]))
            summ[tag]["model_c0_vs_front_c0"] = float(np.median([q["stage"]["model_c0_vs_front_c0"][0] for q in rows]))
    res["median_over_speakers_ms"] = summ
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(dst, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
