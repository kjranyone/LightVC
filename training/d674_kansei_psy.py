"""D674: 官能の物差しの候補(心理音響の標準量 + 変調)が、既にある耳の順位(錨 GU > v5 = 盲検 12 / 12・v5 > 今の構音の描画 = 10-29 の所見・本物の女声が最上)を同じ入力の対で再現するか。
開発の組 dev2 / dev3 の各 30 組(k 5 / 7 / 12 を 10 組ずつ・名前順の先頭から)× 系 now(RUST{k}R)/ v5 / gu / src・本物の女声 = 目標の cen 60 本。
量 = r_ecma(ECMA-418-2 の粗さ)・r_dw(Daniel & Weber の粗さ・有音の中央値)・s_din(DIN 45692 の鋭さ)・m_rough(1〜4 kHz の 4 帯の包絡の変調 30〜150 Hz / 2〜16 Hz の比 dB)・m_200(200 Hz ±3 Hz の線 対 その両脇 dB)。
音量 = 有音の RMS 0.05 FS → 65 dB SPL(d237 と同じ)。mosqito 1.2.1(Apache-2.0・評価専用)。規則は results/conv_p0/d674/prereg.yaml。
    uv run --with mosqito==1.2.1 python d674_kansei_psy.py
"""
from __future__ import annotations

import json
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
OUT = Path(__file__).resolve().parent.parent / "results/conv_p0/d674"
SR = 48000
PA = 20e-6 * 10 ** (65 / 20) / 0.05
PREV = {5: "OURS5RY58GQ3", 7: "OURS7RY58GQ3", 12: "OURS12RY58G"}
GU = {5: "GU5_a112", 7: "GU7_a112", 12: "GU12_a120"}
HIGH_BAD = {"r_ecma": True, "r_dw": True, "s_din": None, "m_rough": True, "m_200": True}


def modfeats(x: np.ndarray, act: np.ndarray) -> dict:
    out_r, out_l = [], []
    for lo, hi in ((1000, 1500), (1500, 2200), (2200, 3200), (3200, 4000)):
        b, a = ss.butter(4, [lo, hi], btype="band", fs=SR)
        env = np.abs(ss.hilbert(ss.lfilter(b, a, x)))
        env = ss.resample_poly(env, 1, 48)
        fr = np.repeat(act, 10)[: len(env)]
        e = env[: len(fr)][fr] if fr.sum() > 2000 else env
        e = e - e.mean()
        f, P = ss.welch(e, 1000, nperseg=1000)
        band = lambda a_, b_: P[(f >= a_) & (f <= b_)].mean()
        out_r.append(10 * np.log10(band(30, 150) / band(2, 16)))
        out_l.append(10 * np.log10(band(197, 203) / (0.5 * (band(185, 194) + band(206, 215)))))
    return {"m_rough": float(np.mean(out_r)), "m_200": float(np.mean(out_l))}


def measure(job: tuple) -> tuple:
    warnings.filterwarnings("ignore")
    from mosqito.sq_metrics import roughness_dw, roughness_ecma, sharpness_din_tv
    tag, path = job
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    x = x.mean(1)[: 8 * SR]
    n = len(x) // 480
    e = 10 * np.log10((x[: n * 480].reshape(n, 480) ** 2).mean(1) + 1e-20)
    act = e > e.max() - 40
    x = x / (np.sqrt((x[: n * 480].reshape(n, 480)[act] ** 2).mean()) + 1e-12) * 0.05
    p = x * PA
    on = lambda t: act[np.clip((np.asarray(t) * 100).astype(int), 0, n - 1)]
    R, _, _, _, _ = roughness_ecma(p, SR)
    Rd, _, _, td = roughness_dw(p, SR, overlap=0.5)
    S, ts = sharpness_din_tv(p, SR, skip=0.2)
    r = {"r_ecma": float(R), "r_dw": float(np.median(Rd[on(td)])) if on(td).sum() else float(np.median(Rd)),
         "s_din": float(np.median(S[on(ts)])) if on(ts).sum() else float(np.median(S))}
    r |= modfeats(x, act)
    return tag, r


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs, pairs = [], []
    for s in ("dev2", "dev3"):
        D = H / s / "d570"
        for k in (5, 7, 12):
            for key in sorted(f.name[:10] for f in (D / f"RUST{k}R").glob("p*.wav"))[:10]:
                tg = f"{s}|{k}|{key}"
                pairs.append(tg)
                jobs += [(f"{tg}|now", D / f"RUST{k}R" / f"{key}_RUST{k}R.wav"), (f"{tg}|v5", D / PREV[k] / f"{key}_{PREV[k]}.wav"),
                         (f"{tg}|gu", D / GU[k] / f"{key}_{GU[k]}.wav"), (f"{tg}|src", H / s / "d56" / f"{key}_SRC.wav")]
        jobs += [(f"real|{s}|{f.name}", f) for f in sorted((H / s / "d37").glob("cen_*.wav"))]
    with Pool(12) as pl:
        res = dict(pl.imap_unordered(measure, jobs, chunksize=2))
    (OUT / "rows.json").write_text(json.dumps(res))
    rep = {}
    real = {m: np.array([v[m] for t, v in res.items() if t.startswith("real|")]) for m in HIGH_BAD}
    for m, hb in HIGH_BAD.items():
        g = lambda sy: np.array([res[f"{p}|{sy}"][m] for p in pairs])
        now, v5, gu, src = g("now"), g("v5"), g("gu"), g("src")
        d = {"median": {"now": round(float(np.median(now)), 4), "v5": round(float(np.median(v5)), 4), "gu": round(float(np.median(gu)), 4), "src": round(float(np.median(src)), 4),
                        "real": round(float(np.median(real[m])), 4), "real_p15_p85": [round(float(np.percentile(real[m], 15)), 4), round(float(np.percentile(real[m], 85)), 4)]}}
        if hb is not None:
            worse = (lambda a, b: a > b) if hb else (lambda a, b: a < b)
            d["frac_gu_better_than_v5"] = round(float(np.mean(worse(v5, gu))), 3)
            d["frac_v5_better_than_now"] = round(float(np.mean(worse(now, v5))), 3)
            d["reproduces"] = bool(d["frac_gu_better_than_v5"] >= 0.8 and d["frac_v5_better_than_now"] >= 0.8)
        rep[m] = d
        print(m, json.dumps(d, ensure_ascii=False), flush=True)
    (OUT / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
