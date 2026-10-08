"""A-1 構音学の実データ解析: 男→女の声道の違いを、同じ文を読んだ実音声(VCTK 共通文 001–024)で測る。

問い:
  Q1 声道長の比(一様な周波数伸縮 α: 女性包絡 E_f(f) ≈ 男性包絡 E_m(f/α))はいくつか。文献値(女性の声道は約 15% 短い → α≈1.15–1.2)と合うか
  Q2 伸縮は非一様か(Fant: 咽頭が相対的に短い女性では F1 と F2/F3 の比が母音ごとに違う)。帯域別 α(F1 域/F2 域/F3 域)と母音クラス別の差
  Q3 一様伸縮で説明できない残差はどれだけか(非一様 3 帯域の方が包絡の相関がどれだけ上がるか)

方法: 男女を 1 対ずつ組み、共通文を MFCC(発話内 CMVN・解析専用)の DTW で整列。両方有声の整列フレームで、
因果 LPC(p24・48k・製品と同じ解析)の包絡を対数周波数格子(150–7000Hz)で評価し、各フレームの一次傾き(dB/oct)を除いてから
α を格子探索(相関最大)。母音クラスは男性フレームの MFCC を k-means(k=8)で分けた近似。

    CUDA_VISIBLE_DEVICES= uv run python a1_vtl_warp.py   # results/artic_a1/vtl_warp.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as D
from train_dec2 import load48

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
OUT = ROOT / "results/artic_a1"
FGRID = np.geomspace(150.0, 7000.0, 160)
ALPHAS = np.exp(np.linspace(np.log(0.8), np.log(1.6), 161))
BANDS = {"F1域 250-1000Hz": (250.0, 1000.0), "F2域 800-2800Hz": (800.0, 2800.0), "F3域 2200-4500Hz": (2200.0, 4500.0),
         "全域 300-5000Hz": (300.0, 5000.0)}


def speakers() -> tuple[list[str], list[str]]:
    info = (VC / "speaker-info.txt").read_text().splitlines()[1:]
    m = sorted("p" + r.split()[0] for r in info if len(r.split()) > 2 and r.split()[2] == "M")
    f = sorted("p" + r.split()[0] for r in info if len(r.split()) > 2 and r.split()[2] == "F")
    return m, f


def env_db(lar: np.ndarray) -> np.ndarray:
    a = D.k_to_a(D.lar_to_k(lar))
    A = np.concatenate([np.ones((a.shape[0], 1)), a], 1)
    w = 2 * np.pi * FGRID / D.SR
    E = np.exp(-1j * np.outer(np.arange(A.shape[1]), w))
    H = 1.0 / np.maximum(np.abs(A @ E), 1e-9)
    db = 20 * np.log10(H)
    pre = 20 * np.log10(np.abs(1 - D.MU * np.exp(-1j * w)))
    return db - pre[None, :]


def detilt(db: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    m = (FGRID >= lo) & (FGRID <= hi)
    lf = np.log2(FGRID[m])
    X = np.stack([np.ones_like(lf), lf], 1)
    y = db[:, m]
    coef = np.linalg.lstsq(X, y.T, rcond=None)[0]
    return y - (X @ coef).T, coef[1]


def warp_alpha(Em: np.ndarray, Ef: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """各行で α を探索: corr(Ef(f), Em(f/α)) を f∈[lo,hi] で最大化。返り値 (α [n], 最大相関 [n])。"""
    m = (FGRID >= lo) & (FGRID <= hi)
    lg = np.log(FGRID)
    yf, _ = detilt(Ef, lo, hi)
    yf = (yf - yf.mean(1, keepdims=True)) / (yf.std(1, keepdims=True) + 1e-9)
    best = np.full(len(Em), -2.0)
    arg = np.ones(len(Em))
    for al in ALPHAS:
        src = np.exp(lg[m] - np.log(al))
        idx = np.interp(np.log(src), lg, np.arange(len(FGRID)))
        i0 = np.clip(np.floor(idx).astype(int), 0, len(FGRID) - 2)
        fr = idx - i0
        ym = Em[:, i0] * (1 - fr) + Em[:, i0 + 1] * fr
        X = np.stack([np.ones(m.sum()), np.log2(FGRID[m])], 1)
        coef = np.linalg.lstsq(X, ym.T, rcond=None)[0]
        ym = ym - (X @ coef).T
        ym = (ym - ym.mean(1, keepdims=True)) / (ym.std(1, keepdims=True) + 1e-9)
        c = (ym * yf).mean(1)
        up = c > best
        best[up] = c[up]
        arg[up] = al
    return arg, best


def mfcc(x: np.ndarray) -> np.ndarray:
    import librosa
    y = librosa.resample(x, orig_sr=D.SR, target_sr=16000)
    m = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20, n_fft=512, hop_length=160, win_length=400)
    m = (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)
    return m


def voiced_10ms(x: np.ndarray, n: int) -> np.ndarray:
    import pyworld
    import librosa
    y = librosa.resample(x, orig_sr=D.SR, target_sr=16000)
    f0, t = pyworld.harvest(y, 16000, f0_floor=60, f0_ceil=700, frame_period=10.0)
    v = np.zeros(n, bool)
    k = min(n, len(f0))
    v[:k] = f0[:k] > 0
    f = np.zeros(n)
    f[:k] = f0[:k]
    return v, f


def pair_frames(pm: str, pf: str) -> dict | None:
    import librosa
    xm = load48(str(pm)).astype(np.float64)
    xf = load48(str(pf)).astype(np.float64)
    xm, _ = librosa.effects.trim(xm, top_db=35)
    xf, _ = librosa.effects.trim(xf, top_db=35)
    if len(xm) < D.SR // 2 or len(xf) < D.SR // 2:
        return None
    Mm, Mf = mfcc(xm), mfcc(xf)
    _, wp = librosa.sequence.dtw(X=Mm, Y=Mf, metric="cosine")
    wp = wp[::-1]
    vm, f0m = voiced_10ms(xm, Mm.shape[1])
    vf, f0f = voiced_10ms(xf, Mf.shape[1])
    keep = [(i, j) for i, j in wp if vm[i] and vf[j]]
    if len(keep) < 20:
        return None
    ii = np.array([i for i, _ in keep])
    jj = np.array([j for _, j in keep])
    larm = D.lar_frames(xm, 24)
    larf = D.lar_frames(xf, 24)
    km = np.clip((ii * 160 * 3 + 480) // D.H, 0, len(larm) - 1)
    kf = np.clip((jj * 160 * 3 + 480) // D.H, 0, len(larf) - 1)
    return {"Em": env_db(larm[km]), "Ef": env_db(larf[kf]), "mf": Mm[:, ii].T, "f0m": f0m[ii], "f0f": f0f[jj]}


def one_pair(mf: tuple[str, str]) -> dict | None:
    m, f = mf
    acc = []
    for u in range(1, 25):
        pm = VC / f"wav48/{m}/{m}_{u:03d}.wav"
        pf = VC / f"wav48/{f}/{f}_{u:03d}.wav"
        if not (pm.exists() and pf.exists()):
            continue
        r = pair_frames(pm, pf)
        if r is not None:
            acc.append(r)
    if not acc:
        return None
    cat = {k: np.concatenate([a[k] for a in acc]) for k in acc[0]}
    cat["pair"] = (m, f)
    print("pair", m, f, "frames", len(cat["Em"]), flush=True)
    return cat


def main() -> int:
    from sklearn.cluster import KMeans
    OUT.mkdir(parents=True, exist_ok=True)
    males, females = speakers()
    rng = np.random.default_rng(0)
    mset = males[::2]
    fset = list(rng.permutation(females))
    pairs = [(m, fset[(2 * k) % len(fset)]) for k, m in enumerate(mset)] + \
            [(m, fset[(2 * k + 1) % len(fset)]) for k, m in enumerate(mset)]
    from multiprocessing import Pool
    with Pool(8) as pool:
        rows = [r for r in pool.imap(one_pair, pairs) if r is not None]
    Em = np.concatenate([r["Em"] for r in rows])
    Ef = np.concatenate([r["Ef"] for r in rows])
    mf = np.concatenate([r["mf"] for r in rows])
    f0m = np.concatenate([r["f0m"] for r in rows])
    f0f = np.concatenate([r["f0f"] for r in rows])
    pid = np.concatenate([np.full(len(r["Em"]), i) for i, r in enumerate(rows)])
    rep = {"n_pairs": len(rows), "n_frames": int(len(Em)), "pairs": [list(r["pair"]) for r in rows], "bands": {}}
    al = {}
    cc = {}
    for bn, (lo, hi) in BANDS.items():
        a, c = warp_alpha(Em, Ef, lo, hi)
        al[bn], cc[bn] = a, c
        good = c > 0.5
        rep["bands"][bn] = {"alpha_median": round(float(np.median(a[good])), 4),
                            "alpha_iqr": [round(float(np.percentile(a[good], 25)), 4), round(float(np.percentile(a[good], 75)), 4)],
                            "corr_median": round(float(np.median(c)), 3), "frac_corr_gt_0.5": round(float(good.mean()), 3),
                            "frac_at_grid_edge": round(float(((a <= ALPHAS[1]) | (a >= ALPHAS[-2])).mean()), 3)}
        print("band", bn, rep["bands"][bn], flush=True)
    a1, c1 = warp_alpha(Em, Ef, 300.0, 5000.0)
    yf, _ = detilt(Ef, 300, 5000)
    ym, _ = detilt(Em, 300, 5000)
    zc = lambda y: (y - y.mean(1, keepdims=True)) / (y.std(1, keepdims=True) + 1e-9)
    c_none = (zc(ym) * zc(yf)).mean(1)
    rep["Q3_corr"] = {"no_warp": round(float(np.median(c_none)), 3), "uniform_alpha": round(float(np.median(c1)), 3),
                      "per_band_mean_of_3": round(float(np.median((cc["F1域 250-1000Hz"] + cc["F2域 800-2800Hz"] + cc["F3域 2200-4500Hz"]) / 3)), 3)}
    pair_alpha = []
    for i, r in enumerate(rows):
        s = pid == i
        g = s & (c1 > 0.5)
        if g.sum() > 30:
            pair_alpha.append({"pair": list(r["pair"]), "alpha_median": round(float(np.median(a1[g])), 4),
                               "f0_ratio_st": round(float(12 * np.log2(np.median(f0f[s]) / np.median(f0m[s]))), 2)})
    rep["per_pair"] = pair_alpha
    pa = np.array([p["alpha_median"] for p in pair_alpha])
    rep["pair_alpha_summary"] = {"median": round(float(np.median(pa)), 4), "p10": round(float(np.percentile(pa, 10)), 4),
                                 "p90": round(float(np.percentile(pa, 90)), 4)}
    km = KMeans(8, n_init=4, random_state=0).fit(mf[:, 1:13])
    lab = km.labels_
    cls = {}
    for k in range(8):
        s = lab == k
        prof = Em[s].mean(0)
        pk = [float(FGRID[i]) for i in range(1, len(FGRID) - 1)
              if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and 200 < FGRID[i] < 4000][:3]
        row = {"n": int(s.sum()), "male_env_peaks_hz": [round(v) for v in pk]}
        for bn in ("F1域 250-1000Hz", "F2域 800-2800Hz", "F3域 2200-4500Hz"):
            g = s & (cc[bn] > 0.5)
            row[bn] = round(float(np.median(al[bn][g])), 4) if g.sum() > 20 else None
        cls[f"c{k}"] = row
    rep["by_vowel_cluster"] = cls
    rep["method_note"] = ("VCTK 共通文 001–024・男性 24 人 × 女性 2 人ずつ。DTW=MFCC(発話内CMVN)。包絡=因果 LPC p24(製品と同じ解析)。"
                          "各フレームで一次傾きを除去してから α∈[0.8,1.6] を格子探索(相関最大)。相関 0.5 以下のフレームは α の集計から除外。")
    (OUT / "vtl_warp.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    np.savez_compressed(OUT / "vtl_warp_frames.npz", Em=Em.astype(np.float32), Ef=Ef.astype(np.float32),
                        a_full=a1.astype(np.float32), c_full=c1.astype(np.float32), lab=lab, pid=pid, f0m=f0m, f0f=f0f,
                        **{f"a_b{i}": al[b].astype(np.float32) for i, b in enumerate(BANDS)})
    print(json.dumps({k: rep[k] for k in ("n_pairs", "n_frames", "Q3_corr", "pair_alpha_summary")}, ensure_ascii=False))
    print(json.dumps(cls, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
