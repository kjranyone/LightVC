"""D675: 目標の参照音声とのメルの分布の距離が、耳の順位(錨 GU > v5 > 今の構音の描画・本人の別録音が最上)を同じ入力の対で再現するか(オーナーの問い 10-29)。
参照 = 目標の登録の録音 d37/_work/ref_TTT.wav(全長)。系 = now(RUST{k}R)/ v5 / gu / src・天井 = 本人の別録音 cen_TTT(8 s の最大音量区間)。
特徴 = log-mel 80 帯(50 Hz〜12 kHz・窓 2048・hop 240・有音のフレーム・音量は有音の平均で正規化)と、その帯の時間差 Δ。
距離 = fd(平均と共分散のフレシェ距離)・w1(帯ごとの 1 次元ワッサースタインの平均 dB)・dw1(Δ の帯ごとのワッサースタインの平均)・env_w1(ケプストラムの 0〜24 で平滑した包絡の w1 = 倍音の縞を除く)。
規則は results/conv_p0/d675/prereg.yaml。
    uv run python d675_mel_dist.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.linalg as sl
import soundfile as sf
from scipy.fft import dct, idct

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
OUT = Path(__file__).resolve().parent.parent / "results/conv_p0/d675"
SR, N, HOP, NM = 48000, 2048, 240, 80
PREV = {5: "OURS5RY58GQ3", 7: "OURS7RY58GQ3", 12: "OURS12RY58G"}
GU = {5: "GU5_a112", 7: "GU7_a112", 12: "GU12_a120"}
FQ = np.fft.rfftfreq(N, 1 / SR)


def _fb() -> np.ndarray:
    m = lambda f: 2595 * np.log10(1 + f / 700)
    im = lambda x: 700 * (10 ** (x / 2595) - 1)
    pts = im(np.linspace(m(50), m(12000), NM + 2))
    return np.stack([np.clip(np.minimum((FQ - pts[i]) / (pts[i + 1] - pts[i]), (pts[i + 2] - FQ) / (pts[i + 2] - pts[i + 1])), 0, None) for i in range(NM)])


FB = _fb()


def logmel(p: Path, sec: float | None = None) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float64", always_2d=True)
    assert sr == SR
    x = x.mean(1)
    if sec is not None:
        w, hop = int(sec * SR), SR // 4
        if len(x) > w:
            o = max(range(0, len(x) - w + 1, hop), key=lambda o_: float((x[o_:o_ + w] ** 2).mean()))
            x = x[o:o + w]
    nf = (len(x) - N) // HOP
    fr = np.lib.stride_tricks.sliding_window_view(x, N)[::HOP][:nf] * np.hanning(N)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    lm = 10 * np.log10(P @ FB.T + 1e-12)
    e = 10 * np.log10(P.sum(1) + 1e-12)
    a = e > e.max() - 35
    lm = lm - lm[a].mean()
    return lm, a


def stats(lm: np.ndarray, a: np.ndarray) -> dict:
    X = lm[a]
    D = np.diff(lm, axis=0)[a[1:] & a[:-1]]
    c = dct(X, type=2, norm="ortho", axis=1)
    c[:, 25:] = 0
    E = idct(c, type=2, norm="ortho", axis=1)
    return {"X": X, "D": D, "E": E}


def w1(a: np.ndarray, b: np.ndarray) -> float:
    q = np.linspace(0.02, 0.98, 49)
    return float(np.mean(np.abs(np.quantile(a, q, axis=0) - np.quantile(b, q, axis=0))))


def fd(a: np.ndarray, b: np.ndarray) -> float:
    m1, m2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    s = sl.sqrtm(c1 @ c2)
    return float(np.sum((m1 - m2) ** 2) + np.trace(c1 + c2 - 2 * np.real(s)))


def dist(s1: dict, s2: dict) -> dict:
    return {"fd": fd(s1["X"], s2["X"]), "w1": w1(s1["X"], s2["X"]), "dw1": w1(np.abs(s1["D"]), np.abs(s2["D"])), "env_w1": w1(s1["E"], s2["E"])}


def one(it: tuple) -> tuple:
    s, k, key = it
    ti = int(key[7:10])
    D = H / s / "d570"
    ref = stats(*logmel(H / s / "d37/_work" / f"ref_{ti:03d}.wav"))
    out = {"real": dist(stats(*logmel(H / s / "d37" / f"cen_{ti:03d}.wav", 8.0)), ref)}
    for nm, p in (("now", D / f"RUST{k}R" / f"{key}_RUST{k}R.wav"), ("v5", D / PREV[k] / f"{key}_{PREV[k]}.wav"), ("gu", D / GU[k] / f"{key}_{GU[k]}.wav"), ("src", H / s / "d56" / f"{key}_SRC.wav")):
        out[nm] = dist(stats(*logmel(p)), ref)
    return f"{s}|{k}|{key}", out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    its = [(s, k, f.name[:10]) for s in ("dev2", "dev3") for k in (5, 7, 12) for f in sorted((H / s / "d570" / f"RUST{k}R").glob("p*.wav"))]
    with Pool(4) as pl:
        res = dict(pl.imap_unordered(one, its, chunksize=4))
    (OUT / "rows.json").write_text(json.dumps(res))
    rep = {}
    for m in ("fd", "w1", "dw1", "env_w1"):
        rep[m] = {}
        for k in (5, 7, 12):
            rows = [v for kk, v in res.items() if kk.split("|")[1] == str(k)]
            g = lambda sy: np.array([r[sy][m] for r in rows])
            now, v5, gu, src, real = g("now"), g("v5"), g("gu"), g("src"), g("real")
            rep[m][k] = {"median": {n: round(float(np.median(v)), 3) for n, v in (("real", real), ("gu", gu), ("v5", v5), ("now", now), ("src", src))},
                         "frac_gu_closer_than_v5": round(float(np.mean(gu < v5)), 3), "frac_v5_closer_than_now": round(float(np.mean(v5 < now)), 3)}
        rep[m]["reproduces"] = bool(all(rep[m][k]["frac_gu_closer_than_v5"] >= 0.8 and rep[m][k]["frac_v5_closer_than_now"] >= 0.8 for k in (5, 7, 12)))
        print(m, json.dumps(rep[m], ensure_ascii=False), flush=True)
    (OUT / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
