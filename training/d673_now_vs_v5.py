"""D673(探索・事前登録なし): オーナーの耳『今の候補(構音の描画 RUST{k}R)は v5(生成器 OURS{k}RY58GQ3 / OURS12RY58G)より明確に劣後』を数値に写す候補を並べる。
開発の組 dev2 / dev3 の 60 組 × k 5 / 7 / 12。同じ入力の対(今 / v5 / 錨 GU / 入力 SRC)で、有音フレームの
lvl_sd(10 ms の dB の標準偏差)・flux(log-mel の隣のフレームとの差の平均 = 構音の速さ)・fcon(0.3〜4 kHz の log-mel から直線の傾きを除いた山谷の標準偏差 = フォルマントの鋭さ)・
cpp(ケプストラムのピークの際立ち = 周期性 / ブザー)・trk(入力 SRC との log-mel の帯ごとの時間の相関の平均 = 入力の構音への追従)・hf(4〜10 kHz 対 0.1〜4 kHz の dB)を測り、今 − v5 の符号の割合を出す。
    uv run python d673_now_vs_v5.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
OUT = Path(__file__).resolve().parent.parent / "results/conv_p0/d673"
SR = 48000
PREV = {5: "OURS5RY58GQ3", 7: "OURS7RY58GQ3", 12: "OURS12RY58G"}
GU = {5: "GU5_a112", 7: "GU7_a112", 12: "GU12_a120"}
N, HOP = 2048, 480
FQ = np.fft.rfftfreq(N, 1 / SR)


def melfb(n: int = 64, lo: float = 80.0, hi: float = 10000.0) -> np.ndarray:
    m = lambda f: 2595 * np.log10(1 + f / 700)
    im = lambda x: 700 * (10 ** (x / 2595) - 1)
    pts = im(np.linspace(m(lo), m(hi), n + 2))
    fb = np.zeros((n, len(FQ)))
    for i in range(n):
        a, b, c = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((FQ - a) / (b - a), (c - FQ) / (c - b)), 0, None)
    return fb, pts[1:-1]


FB, FC = melfb()


def spec(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float64", always_2d=True)
    x = x.mean(1)[: 8 * SR]
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.05
    nf = (len(x) - N) // HOP
    fr = np.stack([x[i * HOP:i * HOP + N] for i in range(nf)]) * np.hanning(N)
    return np.abs(np.fft.rfft(fr, axis=1)) ** 2


def feats(P: np.ndarray, ref_lm: np.ndarray | None, act: np.ndarray) -> dict:
    lm = 10 * np.log10(P @ FB.T + 1e-12)
    e = 10 * np.log10(P.sum(1) + 1e-12)
    a = act[: len(e)]
    lma = lm[a]
    band = (FC >= 300) & (FC <= 4000)
    xs = np.log(FC[band])
    A = np.vstack([xs, np.ones_like(xs)]).T
    resid = lma[:, band] - (A @ np.linalg.lstsq(A, lma[:, band].T, rcond=None)[0]).T
    cep = np.fft.irfft(np.log(P[a] + 1e-12), axis=1)
    q = np.arange(cep.shape[1]) / SR
    qm = (q > 1 / 600) & (q < 1 / 80)
    cq = cep[:, qm]
    cpp = float(np.median(cq.max(1) - np.median(cq, 1)))
    lo, hi = (FQ >= 100) & (FQ < 4000), (FQ >= 4000) & (FQ < 10000)
    d = {"lvl_sd": float(np.std(e[a])), "flux": float(np.mean(np.abs(np.diff(lm, axis=0))[a[1:] & a[:-1]])), "fcon": float(np.mean(resid.std(1))),
         "cpp": cpp, "hf": float(10 * np.log10(P[a][:, hi].sum() / P[a][:, lo].sum()))}
    if ref_lm is not None:
        n = min(len(lm), len(ref_lm))
        aa = a[:n]
        x, y = lm[:n][aa], ref_lm[:n][aa]
        x, y = x - x.mean(0), y - y.mean(0)
        d["trk"] = float(np.mean((x * y).sum(0) / (np.sqrt((x ** 2).sum(0) * (y ** 2).sum(0)) + 1e-9)))
    return d


def one(it: tuple) -> tuple:
    s, k, key = it
    D = H / s / "d570"
    Ps = spec(H / s / "d56" / f"{key}_SRC.wav")
    es = 10 * np.log10(Ps.sum(1) + 1e-12)
    out = {}
    paths = {"now": D / f"RUST{k}R" / f"{key}_RUST{k}R.wav", "v5": D / PREV[k] / f"{key}_{PREV[k]}.wav", "gu": D / GU[k] / f"{key}_{GU[k]}.wav"}
    specs = {n: spec(p) for n, p in paths.items()}
    n = min(len(Ps), *(len(v) for v in specs.values()))
    act = es[:n] > es[:n].max() - 35
    lms = 10 * np.log10(Ps[:n] @ FB.T + 1e-12)
    out["src"] = feats(Ps[:n], None, act)
    for nm, P in specs.items():
        out[nm] = feats(P[:n], lms, act)
    return f"{s}|{k}|{key}", out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    its = [(s, k, f.name[:10]) for s in ("dev2", "dev3") for k in (5, 7, 12) for f in sorted((H / s / "d570" / f"RUST{k}R").glob("p*.wav"))]
    with Pool(12) as pl:
        res = dict(pl.imap_unordered(one, its, chunksize=4))
    (OUT / "feats.json").write_text(json.dumps(res))
    F = ["lvl_sd", "flux", "fcon", "cpp", "hf", "trk"]
    rep = {}
    for k in (5, 7, 12):
        rows = [v for kk, v in res.items() if kk.split("|")[1] == str(k)]
        rep[k] = {}
        for f in F:
            g = lambda nm: np.array([r[nm][f] for r in rows if f in r[nm]])
            now, v5, gu = g("now"), g("v5"), g("gu")
            src = g("src") if f != "trk" else None
            rep[k][f] = {"now": round(float(np.median(now)), 3), "v5": round(float(np.median(v5)), 3), "gu": round(float(np.median(gu)), 3),
                         "src": None if src is None else round(float(np.median(src)), 3), "frac_now_gt_v5": round(float(np.mean(now > v5)), 3)}
        print(k, json.dumps(rep[k], ensure_ascii=False), flush=True)
    (OUT / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
