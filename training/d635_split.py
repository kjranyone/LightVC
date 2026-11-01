"""D635: D634 の PT の不足の切り分け(開発の組 dev2 / dev3・WORLD・錨の非周期性)。
明瞭さ: PV = PT の包絡 + 候補 C の f0・有声 / PM = C の音を条件と同じ領域(STFT の log-mel・DCT 1〜24)で表して PT の描画へ / PP = C の CheapTrick を DCT 24 に縮めた形 + PT の f0・有声・大きさ。
hc_hi: GW = 錨の再合成(対照)/ GS = 錨の包絡を DCT 24 に縮める / GM = 錨の音を条件の領域で表して描く。規則は results/conv_p0/d635/prereg.yaml。
    uv run python d635_split.py --stage render
    uv run python d635_split.py --stage rough
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import librosa
import numpy as np
import soundfile as sf

from d631_world_swap import C, FEATS, G, H, ROOT, SETS, analyze, keys, measure
from d632_f0env_swap import fit, lag, shift, warp
from d634_artic_render import DCT, FB, FQ, PRE, env_from_rows, lift, synth, vdir

OUT = ROOT / "results/conv_p0/d635"
ARMS = ("PV", "PM", "PP", "GW", "GS", "GM")
LO, HI_ = (FQ >= 500) & (FQ < 3000), (FQ >= 3000) & (FQ < 8000)


def rows_of(x: np.ndarray, n: int) -> np.ndarray:
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=240, window="hann", center=True))
    lm = np.log(np.maximum(FB @ S, 1e-5)).T
    return fit((lm @ DCT[1:25].T) / 10.0, n)


def norm(sh: np.ndarray, pin: np.ndarray) -> np.ndarray:
    return sh / sh.mean(1, keepdims=True) * pin


def tilt(sp: np.ndarray, f0: np.ndarray) -> float:
    v = f0[: len(sp)] > 0
    t = 10 * np.log10(sp[: len(v)][v][:, HI_].mean(1) / sp[: len(v)][v][:, LO].mean(1))
    return float(np.median(t)) if len(t) else float("nan")


def render(it: tuple) -> dict:
    s, k, key = it
    d = H / s / "d570"
    from d6_unit_cov import load48
    v = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)[PRE:PRE + 1600].astype(np.float64)
    xc = load48(d / C[k] / f"{key}_{C[k]}.wav").astype(np.float64)
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    xg = load48(d / G[k][0] / f"{key}_{G[k][0]}.wav").astype(np.float64)
    lg = lag(xc, xs)
    f0c, spc, _ = analyze(xc)
    f0s0, sps0, aps0 = analyze(xs)
    sps, aps = shift(sps0, lg), shift(aps0, lg)
    n = len(v)
    apg = fit(warp(aps, G[k][1]), n)
    pin = fit(sps, n).mean(1, keepdims=True)
    f0 = np.where(v[:, 26] > 0.5, 200.0 * np.exp(v[:, 25]), 0.0)
    sp_pt = norm(env_from_rows(v[:, 1:25]), pin)
    ng = len(f0s0)
    apg0, spg0 = warp(aps0, G[k][1]), warp(sps0, G[k][1])
    f0g = f0s0 * 2 ** (k / 12)
    pg = spg0.mean(1, keepdims=True)
    sp = {"PV": sp_pt, "PM": norm(env_from_rows(rows_of(xc, n)), pin), "PP": norm(fit(lift(spc), n), pin),
          "GW": spg0, "GS": lift(spg0), "GM": norm(env_from_rows(rows_of(xg, ng)), pg)}
    f0s = {"PV": fit(f0c[:, None], n)[:, 0], "PM": f0, "PP": f0, "GW": f0g, "GS": f0g, "GM": f0g}
    aps_ = {"PV": apg, "PM": apg, "PP": apg, "GW": apg0, "GS": apg0, "GM": apg0}
    for a in ARMS:
        o = d / f"{a}_{k}" / f"{key}_{a}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            sf.write(o, synth(f0s[a], sp[a], aps_[a]), 48000, subtype="PCM_16")
    rec = {"key": f"{s}|{k}|{key}", "lag": lg, "tilt": {a: tilt(sp[a], f0s[a]) for a in ARMS}}
    rec["tilt"]["PT"] = tilt(sp_pt, f0)
    rec["tilt"]["C"] = tilt(spc, f0c)
    return rec


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["render", "rough"])
    a = ap_.parse_args()
    npr = int(os.environ.get("LVC_NPROC", "12"))
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    if a.stage == "render":
        with Pool(npr) as pl:
            recs = pl.map(render, its, chunksize=2)
        (OUT / "render_rec.json").write_text(json.dumps(recs, indent=0))
        for k in C:
            rk = [r for r in recs if r["key"].split("|")[1] == str(k)]
            print("tilt", k, {x: round(float(np.nanmedian([r["tilt"][x] for r in rk])), 2) for x in rk[0]["tilt"]}, flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0]} | {x: f"{x}_{k}" for x in ARMS}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ARMS:
        rep[arm] = {}
        for k in C:
            rep[arm][k] = {}
            nm = sysn(k)
            for f, (_, hi) in FEATS.items():
                xs = np.array([res[str(H / s / "d570" / nm[arm] / f"{key}_{nm[arm]}.wav")][f] for s, kk, key in its if kk == k])
                gs = np.array([res[str(H / s / "d570" / nm["G"] / f"{key}_{nm['G']}.wav")][f] for s, kk, key in its if kk == k])
                m = np.isfinite(xs) & np.isfinite(gs)
                gb = float(np.mean(gs[m] > xs[m])) if hi else float(np.mean(gs[m] < xs[m]))
                rep[arm][k][f] = {"G_better": round(gb, 3), "median_X": round(float(np.median(xs[m])), 4), "median_G": round(float(np.median(gs[m])), 4)}
            print(arm, k, {f: (v["G_better"], v["median_X"], v["median_G"]) for f, v in rep[arm][k].items()}, flush=True)
        rep[arm]["hcpar"] = bool(all(rep[arm][k]["hc_hi"]["G_better"] <= 0.60 for k in C))
    rep["GW_control_ok"] = bool(all(0.40 <= rep["GW"][k]["hc_hi"]["G_better"] <= 0.60 for k in C))
    (OUT / "rough.json").write_text(json.dumps(rep, indent=1))
    print("hcpar", {arm: rep[arm]["hcpar"] for arm in ARMS}, "GW control", rep["GW_control_ok"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
