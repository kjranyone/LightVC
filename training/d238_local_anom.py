"""D238(学習なし・results/conv_p0/d238_prereg.yaml): 本物の声がまず出さない局所の音のパターン(50 ms)の出現率は、変換の耳の不合格を再現するか。
特徴 = mel 40 帯の log パワー(発話の有音 p95 からの相対 dB)の前後 2 フレーム(200 次元)→ PCA 60 → GMM 32 成分(本物の女声 400 人だけで当てる)。
閾値 τ = 当てに使っていない本物 150 人(D236 の REAL_POOL)の −log p の p99。項 = −log p > τ のフレームの割合(全体 anom・有音 anom_act)。
    uv run python d238_local_anom.py --s <dev の親> --out ../results/conv_p0/d238.json
"""
from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 48000
SEC = 8.0
NFFT, WIN, HOP = 2048, 1200, 480
FAIL_GROUPS = ("FAIL_cand_ab1", "FAIL_prodcand", "FAIL_convg1", "FAIL_conv_ab1", "dev:G03S10ZH", "dev:RRQSDC19LE")
_FB: list = []


def load48(p: str) -> np.ndarray:
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    x, sr = sf.read(p, dtype="float64", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x[: int(SEC * SR)]


def feats(p: str) -> tuple:
    import librosa
    if not _FB:
        _FB.append(librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=40, fmin=50, fmax=16000))
    x = load48(p)
    n = (len(x) - WIN) // HOP + 1
    fr = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP][:n] * np.hanning(WIN)
    P = np.abs(np.fft.rfft(fr, NFFT)) ** 2
    L = 10 * np.log10(_FB[0] @ P.T + 1e-20)
    e = 10 * np.log10(P.sum(1) + 1e-20)
    act = e > e.max() - 40
    ref = np.percentile(e[act], 95)
    L = np.maximum(L - ref, -100.0)
    Lp = np.pad(L, ((0, 0), (2, 2)), mode="edge")
    X = np.concatenate([Lp[:, k:k + L.shape[1]] for k in range(5)], 0).T
    return X.astype(np.float32), act


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    import soundfile as sf
    S = Path(a.s)
    d237 = json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]
    held = [r["id"] for r in d237 if r["grp"] == "REAL_POOL"]
    excl = set((S / "excl.txt").read_text().split()) | set(held)
    for f in list(S.glob("*/devset.json")) + [S / "eval_targets.json"]:
        if f.exists():
            excl |= set(json.loads(f.read_text())["targets"])
    base = ROOT / "female-dataset"
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(238).shuffle(pool)
    fit = []
    for spk in pool:
        ws = sorted((base / spk).glob("*.wav"))
        if len(ws) < 3 or sf.info(str(ws[len(ws) // 2])).duration < 3:
            continue
        fit.append(str(ws[len(ws) // 2]))
        if len(fit) == 400:
            break
    with Pool(a.workers) as pl:
        F = pl.map(feats, fit)
    Xf = np.concatenate([f[0] for f in F])
    rng = np.random.default_rng(0)
    Xf = Xf[rng.permutation(len(Xf))[:200000]]
    Xf = Xf.astype(np.float64)
    pca = PCA(60, random_state=0).fit(Xf)
    gmm = GaussianMixture(32, covariance_type="full", random_state=0, max_iter=200, reg_covar=1.0).fit(pca.transform(Xf))
    print("fit", len(fit), Xf.shape, "converged", gmm.converged_, flush=True)
    paths = {r["id"]: r for r in d237}
    for r in d237:
        r["path"] = str(base / r["id"] / sorted(p.name for p in (base / r["id"]).glob("*.wav"))[len(list((base / r["id"]).glob("*.wav"))) // 2]) if r["grp"] == "REAL_POOL" else None
    eb = ROOT / "results/earbattery"
    for r in d237:
        if r["path"] is None:
            if r["grp"].startswith("dev:"):
                r["path"] = str(S / "dev" / {"KNN": "d135_real", "G03S10ZH": "d114", "RRQSDC19LE": "d139"}[r["form"]] / r["id"].split("/", 1)[1])
            else:
                d, t, L = r["id"].split("/")
                r["path"] = str(eb / d / t / f"{L}.wav")
    order = [r["id"] for r in d237]
    with Pool(a.workers) as pl:
        FF = pl.map(feats, [paths[i]["path"] for i in order])
    nll = {i: -gmm.score_samples(pca.transform(f[0].astype(np.float64))) for i, f in zip(order, FF)}
    acts = {i: f[1] for i, f in zip(order, FF)}
    tau = float(np.percentile(np.concatenate([nll[i] for i in held]), 99))
    rows = []
    for i in order:
        r = paths[i]
        s, ac = nll[i], acts[i]
        rows.append({"grp": r["grp"], "id": i, "form": r.get("form", ""), "anom": float((s > tau).mean()), "anom_act": float((s[ac] > tau).mean()) if ac.sum() else float("nan"),
                     "nll_p99": float(np.percentile(s, 99)), "nll_med": float(np.median(s))})
    rep: dict = {"tau": tau, "fit_n": len(fit)}
    M = ("anom", "anom_act", "nll_p99", "nll_med")
    real = {m: np.array([r[m] for r in rows if r["grp"] == "REAL_POOL"], float) for m in M}
    rep["real_p15_p85"] = {m: [round(float(np.nanpercentile(v, 15)), 4), round(float(np.nanpercentile(v, 85)), 4)] for m, v in real.items()}
    grps = sorted({r["grp"] for r in rows})
    rep["medians"] = {g: {m: round(float(np.nanmedian([r[m] for r in rows if r["grp"] == g])), 4) for m in M} | {"n": sum(r["grp"] == g for r in rows)} for g in grps}
    rep["axis"] = {}
    for m in ("anom", "anom_act"):
        hi = rep["real_p15_p85"][m][1]
        pass_ok = all(rep["medians"][g][m] <= hi for g in ("PASS_rvoc", "PASS_bigvgan"))
        det = {g: rep["medians"][g][m] > hi for g in FAIL_GROUPS}
        rep["axis"][m] = {"pass_ok": pass_ok, "fail_detected": det, "axis_ok": pass_ok and all(det.values())}
    rep["trial1"] = {r["form"]: {m: round(r[m], 4) for m in M} for r in rows if r["id"].startswith("cand_ab1/cand_01/")}
    rep["verdict"] = "LOCAL_EXPLAINS" if any(v["axis_ok"] for v in rep["axis"].values()) else "NONE"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep | {"rows": rows}, indent=1, ensure_ascii=False))
    np.savez_compressed(Path(a.out).with_suffix(".nll.npz"), **{i.replace("/", "__"): nll[i].astype(np.float32) for i in order})
    import pickle
    Path(a.out).with_suffix(".model.pkl").write_bytes(pickle.dumps({"pca": pca, "gmm": gmm, "tau": tau}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
