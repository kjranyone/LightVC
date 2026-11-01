"""D676: 官能指数 v0 = 本物の声の帯(オーナー 10-29:『いままで聴かせたものは A/B の比較しようもないくらい全部不合格』→ 正解 = 本物だけが合格・これまでの合成は全部不合格)。
軸(8 s・有音の RMS で正規化): r_ecma(ECMA-418-2 の粗さ = ザラザラ)・cpp(周期性の強さ = ブザー)・d_lo / d_mid / d_hi(|Δlog-mel| の帯の平均 = 動き・ラジオの止まった音)・
hf(4〜10 kHz 対 0.1〜4 kHz = こもり / 明るさ)・fcon(0.3〜4 kHz の山谷 = フォルマントの鋭さ)・lvl_sd(有音の音量の揺れ)・hc_hi / phc_lo / phc_hi / pflutter(倍音の山谷・周期ごとのちらつき = ガビ / ジャギー)。
本物の帯 = female-dataset の実女声(話者ごとに 1 本・刺激の目標の話者を除く)を fit / calib / test に話者で 3 分割。各軸の f0 で条件づけた頑健な z(刺激の f0 の中央値 ±2 半音の fit の本物・40 本に満たなければ広げる)。
指数 = 軸の |z| の最大・閾 τ = calib の本物の 95 パーセンタイル。合成の刺激 = 耳に出した全部(results/earbattery の音声の残るもの)・本物の刺激 = 同じページの目標・参照・本物の入力。規則は results/conv_p0/d676/prereg.yaml。
    uv run --with mosqito==1.2.1 python d676_kansei_band.py --stage feat | judge
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"
OUT = ROOT / "results/conv_p0/d676"
SR = 48000
PA = 20e-6 * 10 ** (65 / 20) / 0.05
AX = ["r_ecma", "cpp", "d_lo", "d_mid", "d_hi", "hf", "fcon", "lvl_sd", "hc_hi", "phc_lo", "phc_hi", "pflutter"]
N, HOP = 2048, 480


def stimuli() -> list:
    rows = []
    add = lambda sess, f, role, sysn: rows.append({"sess": sess, "path": str(f), "role": role, "sys": sysn})
    for sess in ("mf_v5_ear", "mf_v3_ear", "ff_v3_ear", "ff_ab_ear", "mf_pitch_ear", "mf_pitch_ear2", "now_v6_dev"):
        key = {}
        kf = [p for p in (EB / sess).glob("*.json") if "key" in p.name]
        if kf:
            key = json.loads(kf[0].read_text())
        for f in sorted((EB / sess).glob("*.wav")):
            tid, part = f.stem.split("_", 1)
            if part == "tgt":
                add(sess, f, "real", "tgt")
            elif part == "in":
                add(sess, f, "real" if sess.startswith("ff") else "tts", "in")
            else:
                sysn = part
                if sess == "mf_v5_ear":
                    sysn = key[tid][part]
                elif sess == "ff_ab_ear":
                    sysn = key[tid][part]
                add(sess, f, "synth", sysn)
    for sess in ("zsvc_ab", "zsvc_ab2"):
        for f in sorted((EB / sess).glob("*.wav")):
            part = f.stem.split("_", 1)[1]
            add(sess, f, "synth" if part == "conv" else "real", part)
    for f in sorted((EB / "ddsp_ab").glob("*.wav")):
        part = f.stem.split("_", 1)[1]
        add("ddsp_ab", f, "real" if part == "ref" else "synth", part)
    for f in sorted((EB / "ddsp_sim").glob("*.wav")):
        part = f.stem.split("_", 1)[1]
        add("ddsp_sim", f, "real" if part == "ref" else "synth", part)
    for sess in ("ddsp_chorus", "dspvc_p0"):
        mp = json.loads((EB / sess / "_key_聴取後に開く.json").read_text())["map"]
        for x, v in mp.items():
            kind = v.split("|")[0] if sess == "ddsp_chorus" else v.split("|")[1]
            role = {"conv": "synth", "pos_ctrl": "synth", "pitch": "synth", "pitch_vtl": "synth", "anchor": "real", "source": "real"}[kind]
            add(sess, EB / sess / f"{x}.wav", role, v)
    return rows


def pool(exclude: set, n: int = 600) -> list:
    base = ROOT / "female-dataset"
    spk = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in exclude)
    rng = random.Random(676)
    rng.shuffle(spk)
    out = []
    for i, s in enumerate(spk[:n]):
        ws = sorted((base / s).glob("*.wav"))
        if ws:
            out.append({"sess": "POOL", "path": str(ws[len(ws) // 2]), "role": ["fit", "fit", "calib", "test"][i % 4], "sys": s})
    return out


def feat(row: dict) -> dict:
    warnings.filterwarnings("ignore")
    import d629_smooth as D9
    import d630_pitchsync as D0
    from mosqito.sq_metrics import roughness_ecma
    x = D9.load(Path(row["path"]), mid=row["sess"] == "POOL")
    r = dict(row)
    r0, r9 = D0.feats(x), D9.feats(x)
    r |= {"f0med": r0.get("f0med", float("nan")), "phc_lo": r0.get("phc_lo", float("nan")), "phc_hi": r0.get("phc_hi", float("nan")), "pflutter": r0.get("pflutter", float("nan")), "hc_hi": r9.get("hc_hi", float("nan"))}
    r["r_ecma"] = float(roughness_ecma(x * PA, SR)[0])
    nf = (len(x) - N) // HOP
    fr = np.lib.stride_tricks.sliding_window_view(x, N)[::HOP][:nf] * np.hanning(N)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    fq = np.fft.rfftfreq(N, 1 / SR)
    from d673_now_vs_v5 import FB, FC
    fb = np.stack([np.interp(fq, np.fft.rfftfreq(2048, 1 / SR), b) for b in FB])
    lm = 10 * np.log10(P @ fb.T + 1e-12)
    e = 10 * np.log10(P.sum(1) + 1e-12)
    a = e > e.max() - 35
    aa = a[1:] & a[:-1]
    D = np.abs(np.diff(lm, axis=0))[aa]
    r |= {"d_lo": float(D[:, FC < 500].mean()), "d_mid": float(D[:, (FC >= 500) & (FC < 3000)].mean()), "d_hi": float(D[:, FC >= 3000].mean()), "lvl_sd": float(np.std(e[a]))}
    lo, hi = (fq >= 100) & (fq < 4000), (fq >= 4000) & (fq < 10000)
    r["hf"] = float(10 * np.log10(P[a][:, hi].sum() / P[a][:, lo].sum()))
    band = (FC >= 300) & (FC <= 4000)
    xs = np.log(FC[band])
    A = np.vstack([xs, np.ones_like(xs)]).T
    lma = lm[a][:, band]
    r["fcon"] = float(np.mean((lma - (A @ np.linalg.lstsq(A, lma.T, rcond=None)[0]).T).std(1)))
    cep = np.fft.irfft(np.log(P[a] + 1e-12), axis=1)
    q = np.arange(cep.shape[1]) / SR
    cq = cep[:, (q > 1 / 600) & (q < 1 / 80)]
    r["cpp"] = float(np.median(cq.max(1) - np.median(cq, 1)))
    return r


def zrow(r: dict, fit: list) -> dict:
    f0 = r["f0med"]
    zs = {}
    for ax in AX:
        if not np.isfinite(r.get(ax, np.nan)):
            continue
        cand = [(abs(12 * np.log2(p["f0med"] / f0)), p[ax]) for p in fit if np.isfinite(p.get(ax, np.nan)) and np.isfinite(p["f0med"])] if np.isfinite(f0) else []
        w = 2.0
        sel = [v for d, v in cand if d <= w]
        while len(sel) < 40 and w < 24:
            w += 1.0
            sel = [v for d, v in cand if d <= w]
        if len(sel) < 10:
            sel = [p[ax] for p in fit if np.isfinite(p.get(ax, np.nan))]
        med = float(np.median(sel))
        mad = float(np.median(np.abs(np.array(sel) - med))) * 1.4826 + 1e-9
        zs[ax] = (r[ax] - med) / mad
    return zs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["feat", "judge"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "feat":
        st = stimuli()
        tg = set()
        for k in ("mf_v5_ear", "mf_v3_ear", "ff_v3_ear", "ff_ab_ear", "mf_pitch_ear", "mf_pitch_ear2"):
            kf = [p for p in (EB / k).glob("*.json") if "key" in p.name]
            if kf:
                tg |= {v.get("target") for v in json.loads(kf[0].read_text()).values() if isinstance(v, dict) and v.get("target")}
        rows = st + pool(tg)
        print("stimuli", {r_: sum(1 for x in st if x["role"] == r_) for r_ in ("real", "synth", "tts")}, "pool", len(rows) - len(st), flush=True)
        with Pool(12) as pl:
            res = list(pl.imap_unordered(feat, rows, chunksize=2))
        (OUT / "feats.json").write_text(json.dumps(res))
        print("feats", len(res), flush=True)
        return 0
    res = json.loads((OUT / "feats.json").read_text())
    fit = [r for r in res if r["role"] == "fit"]
    for r in res:
        r["z"] = zrow(r, fit)
        r["idx"] = max((abs(v) for v in r["z"].values()), default=float("nan"))
        r["worst"] = max(r["z"], key=lambda k_: abs(r["z"][k_])) if r["z"] else None
    tau = float(np.percentile([r["idx"] for r in res if r["role"] == "calib"], 95))
    rep = {"tau": round(tau, 3)}
    grp = lambda f: [r for r in res if f(r)]
    for nm, g in (("test_pool", grp(lambda r: r["role"] == "test")), ("real_stimuli", grp(lambda r: r["role"] == "real")),
                  ("synth_stimuli", grp(lambda r: r["role"] == "synth")), ("tts_inputs", grp(lambda r: r["role"] == "tts"))):
        fail = [r for r in g if r["idx"] > tau]
        rep[nm] = {"n": len(g), "fail_rate": round(len(fail) / max(1, len(g)), 3),
                   "worst_axis_counts": {k: sum(1 for r in fail if r["worst"] == k) for k in AX if any(r["worst"] == k for r in fail)}}
    rep["synth_by_session"] = {}
    for s in sorted({r["sess"] for r in res if r["role"] == "synth"}):
        g = grp(lambda r: r["role"] == "synth" and r["sess"] == s)
        rep["synth_by_session"][s] = {"n": len(g), "fail_rate": round(float(np.mean([r["idx"] > tau for r in g])), 3), "median_idx": round(float(np.median([r["idx"] for r in g])), 2)}
    rep["synth_axis_exceed_rate"] = {ax: round(float(np.mean([abs(r["z"].get(ax, 0)) > tau for r in grp(lambda r: r["role"] == "synth")])), 3) for ax in AX}
    rep["real_axis_exceed_rate"] = {ax: round(float(np.mean([abs(r["z"].get(ax, 0)) > tau for r in grp(lambda r: r["role"] in ("test", "real"))])), 3) for ax in AX}
    rep["valid"] = bool(rep["test_pool"]["fail_rate"] <= 0.10 and rep["real_stimuli"]["fail_rate"] <= 0.15 and rep["synth_stimuli"]["fail_rate"] >= 0.90)
    (OUT / "judge.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    (OUT / "rows_z.json").write_text(json.dumps([{k: r[k] for k in ("sess", "path", "role", "sys", "f0med", "idx", "worst", "z")} for r in res], ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
