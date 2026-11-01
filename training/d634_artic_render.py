"""D634: 生成器を使わない構音の描画の試作(オーナー承認 10-28)。製品の Rust の経路を LIGHTVC_DUMP_VCOND つきで鳴らし、生成器に入る実効の条件
[c0・包絡 24・log f0・有声] の包絡(構音目標表 Σ p T + 文脈の補正 + R の ΔS)から WORLD で合成する(PT)。診断 PB = 候補 C の包絡を DCT 0〜24 の log-mel に縮めて戻す。
規則は results/conv_p0/d634/prereg.yaml。
    uv run python d634_artic_render.py --stage rust
    uv run python d634_artic_render.py --stage render
    uv run python d634_artic_render.py --stage rough
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf

from d631_world_swap import C, FEATS, FP, G, H, ROOT, SETS, analyze, keys, measure
from d632_f0env_swap import fit, lag, shift, warp

OUT = ROOT / "results/conv_p0/d634"
BIN = ROOT / "target/release/lightvc-app"
N_MEL, NB = 128, 1025
FC = librosa.mel_frequencies(N_MEL + 2, fmin=0.0, fmax=24000.0)[1:-1]
FQ = np.linspace(0, 24000, NB)
FB = librosa.filters.mel(sr=48000, n_fft=2 * (NB - 1), n_mels=N_MEL, fmin=0.0, fmax=24000.0)
k_ = np.arange(N_MEL)[:, None]
i_ = np.arange(N_MEL)[None]
DCT = np.cos(np.pi / N_MEL * (i_ + 0.5) * k_) * np.sqrt(2.0 / N_MEL)
DCT[0] /= np.sqrt(2.0)
PRE = 2400


def vdir(s: str, k: int) -> Path:
    return H / s / "d634_vcond" / f"k{k}"


def rust(job: tuple) -> str:
    s, k, key = job
    S = H / s
    pi, ti = int(key[1:5]), int(key[7:10])
    vf = vdir(s, k) / f"{key}.vcond"
    o = S / "d570" / f"R634_{k}" / f"{key}_R634_{k}.wav"
    if vf.exists() and o.exists():
        return "skip"
    opw = S / "d56" / "_work" / f"op_{pi:04d}.wav"
    cmd = [str(BIN), "artic", "--enroll", str(S / "d85_work" / f"en_{ti:03d}"), "--operator-enroll", str(S / "d85_work" / f"op_{pi:04d}"),
           "--mix", "1", "--causal-ct", "--per-p75", "--f4", "results/f4_1/export", "--rvoc", "results/diag_ppgvoc58/export", "--c2", "results/c2_1/export",
           "--opslow", "4,0", "--c0-ppg-ref", str(opw), "--semitones", str(k), "--rr", "results/d587_rbar/export", "--rr-mem", str(S / "rr_mem" / f"en_{ti:03d}"),
           "--rr-op-ref", str(opw), "--rr-stride", "2"] + (["--h1eq", "3"] if k in (5, 7) else [])
    vf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "y.wav"
        r = subprocess.run(cmd + ["-i", str(S / "d467" / f"{key}.wav"), "-o", str(tmp)], cwd=ROOT, capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "LIGHTVC_DUMP_VCOND": str(vf), "LVC_SPIN_US": "0", "LVC_NO_PIN": "1"})
        if r.returncode != 0:
            return f"FAIL {s} {k} {key} {r.stderr[-200:]}"
        y, sr = sf.read(tmp, dtype="float32")
    n0 = 12 * sr + 240
    y = y[n0:n0 + 8 * sr]
    o.parent.mkdir(parents=True, exist_ok=True)
    sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), sr)
    return "ok"


def env_from_rows(rows: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    lm = (rows * 10.0) @ DCT[1:25]
    fc = FC * alpha
    return np.exp(2.0 * np.stack([np.interp(FQ, fc, r) for r in lm]))


def lift(sp: np.ndarray) -> np.ndarray:
    lm = np.log(np.maximum(np.sqrt(sp) @ FB.T, 1e-12))
    c = lm @ DCT.T
    c[:, 25:] = 0.0
    sm = c @ DCT
    out = np.exp(2.0 * np.stack([np.interp(FQ, FC, r) for r in sm]))
    return out * (sp.mean(1, keepdims=True) / out.mean(1, keepdims=True))


def synth(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray) -> np.ndarray:
    y = pw.synthesize(np.ascontiguousarray(f0, dtype=np.float64), np.ascontiguousarray(sp), np.ascontiguousarray(ap), 48000, FP)
    return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)


def render(it: tuple) -> dict:
    s, k, key = it
    d = H / s / "d570"
    from d6_unit_cov import load48
    v = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)[PRE:PRE + 1600].astype(np.float64)
    xc = load48(d / C[k] / f"{key}_{C[k]}.wav").astype(np.float64)
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    lg = lag(xc, xs)
    f0c, spc, _ = analyze(xc)
    _, sps, aps = (shift(m, lg) for m in analyze(xs))
    n = len(v)
    apg = fit(warp(aps, G[k][1]), n)
    pin = fit(sps, n).mean(1, keepdims=True)
    f0 = np.where(v[:, 26] > 0.5, 200.0 * np.exp(v[:, 25]), 0.0)
    rec = {"key": f"{s}|{k}|{key}", "lag": lg}
    nf = min(len(f0c), n)
    vv = (f0[:nf] > 0) & (f0c[:nf] > 0)
    rec["f0_ratio_med"] = float(np.median(f0[:nf][vv] / f0c[:nf][vv])) if vv.any() else float("nan")
    sh = env_from_rows(v[:, 1:25])
    sp = sh / sh.mean(1, keepdims=True) * pin
    o = d / f"PT_{k}" / f"{key}_PT_{k}.wav"
    o.parent.mkdir(parents=True, exist_ok=True)
    yt = synth(f0, sp, apg)
    sf.write(o, yt, 48000, subtype="PCM_16")
    if s == "dev2" and k == 7 and key in keys("dev2")[:12]:
        sh1 = env_from_rows(v[:, 1:25], 1.0)
        rec["alpha1_identity"] = bool(np.array_equal(sh1, sh))
        sha = env_from_rows(v[:, 1:25], 1.1)
        ya = synth(f0, sha / sha.mean(1, keepdims=True) * pin, apg)
        _, s0, _ = analyze(yt.astype(np.float64))
        _, s1, _ = analyze(ya.astype(np.float64))
        lf = np.linspace(np.log(300.0), np.log(5000.0), 400)
        a0 = np.log(np.stack([np.interp(np.exp(lf), FQ, r) for r in s0]) + 1e-20)
        a1 = np.log(np.stack([np.interp(np.exp(lf), FQ, r) for r in s1]) + 1e-20)
        sv = np.flatnonzero(f0[: min(len(a0), len(a1))] > 0)
        step = lf[1] - lf[0]
        shifts = []
        for t in sv[::4]:
            x0, x1 = a0[t] - a0[t].mean(), a1[t] - a1[t].mean()
            cc = [float(np.dot(x0[max(0, -m):len(x0) - max(0, m)], x1[max(0, m):len(x1) - max(0, -m)])) for m in range(-30, 31)]
            shifts.append(np.exp((int(np.argmax(cc)) - 30) * step))
        rec["alpha11_ratio_med"] = float(np.median(shifts)) if shifts else float("nan")
    ob = d / f"PB_{k}" / f"{key}_PB_{k}.wav"
    ob.parent.mkdir(parents=True, exist_ok=True)
    sf.write(ob, synth(f0c, lift(spc), fit(apg, len(f0c))), 48000, subtype="PCM_16")
    return rec


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["rust", "render", "rough"])
    a = ap_.parse_args()
    npr = int(os.environ.get("LVC_NPROC", "12"))
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    if a.stage == "rust":
        with Pool(8) as pl:
            res = list(pl.imap_unordered(rust, its))
        print({x: sum(r.startswith(x) for r in res) for x in ("ok", "skip", "FAIL")}, [r for r in res if r.startswith("FAIL")][:3], flush=True)
        return 0
    if a.stage == "render":
        with Pool(npr) as pl:
            recs = pl.map(render, its, chunksize=2)
        (OUT / "render_rec.json").write_text(json.dumps(recs, indent=0))
        fr = np.array([r["f0_ratio_med"] for r in recs])
        kn = [r for r in recs if "alpha11_ratio_med" in r]
        print("rendered", len(recs), "f0 ratio PT/C median", round(float(np.nanmedian(fr)), 4), "lag", {int(v): sum(r["lag"] == v for r in recs) for v in sorted({r["lag"] for r in recs})},
              "knob alpha1 identity", all(r["alpha1_identity"] for r in kn), "alpha1.1 ratio", [round(r["alpha11_ratio_med"], 3) for r in kn], flush=True)
        return 0
    sysn = lambda k: {"G": G[k][0], "PT": f"PT_{k}", "PB": f"PB_{k}", "R634": f"R634_{k}", "C": C[k]}
    prev = json.loads((ROOT / "results/conv_p0/d631/rough_feat.json").read_text())
    paths = sorted({str(H / s / "d570" / v / f"{key}_{v}.wav") for s, k, key in its for v in sysn(k).values()} - set(prev))
    with Pool(npr) as pl:
        res = prev | dict(pl.imap_unordered(measure, paths, chunksize=4))
    (OUT / "rough_feat.json").write_text(json.dumps({p: res[p] for p in paths}, indent=0))
    rep: dict = {}
    for arm in ("PT", "PB", "R634", "C"):
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
        rep[arm]["parity"] = bool(all(rep[arm][k][f]["G_better"] <= 0.60 for k in C for f in FEATS))
    (OUT / "rough.json").write_text(json.dumps(rep, indent=1))
    print("parity", {arm: rep[arm]["parity"] for arm in rep}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
