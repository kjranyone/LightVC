"""段 P52 step a(paper_snv rev124・学習なし): 擬似の操作者の版の声に「粗さ」(息の成分 = 倍音の帯の非周期性・高さの揺らぎ)を足すと、前段の重みが声を落とす隔たりが本物の声の水準へ動くかを測る。
版の作り(snv_p38_wav.py)と同じ声ごとの (r, α, 部屋) と同じ乱数で、検証の声の普通の発話 n 本を WORLD の分析合成 → 録音の鎖 v2 に通す。違いは合成の前の 2 つの加工だけ:
ap = 非周期性を 4 kHz より下で ap + a·(1 − ap) に上げる(息の成分・倍音の櫛が浅くなる)・jit = 有声のフレームの高さに (1 + j·e)(e は 3 フレームでならした正規の乱数・高さの揺らぎ)。
変種 = plain(加工なし = 版と同じ作り)・ap0.2・ap0.4・jit0.01・jit0.02・ap0.3+jit0.015。それぞれに正しい音楽を 10 dB・0 dB で足し、重み(研究の重み)の声の残り・足された音の残り・帯ごとの声を落とす隔たりを出す(p40_pitch.mask_stats)。
並べる: 倍音の櫛の深さ comb_db・周期性 per_lo(p40_pitch.sig_stats)を、本物の男声(段 P40 の sig.json)の分布と比べる。出力は数値だけ。
    cd training && uv run --no-sync python snv_diag/p52_rough.py [n=72] [mask=snv_p45_mask] [out=results/snv_p52/rough.json]
"""
from __future__ import annotations

import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p40_pitch as PP
import snv_d5_chain as CH
import snv_d5_world as W
import snv_p0_front as P
import train_snv_mask as MK
import train_snv_sep as SEP

ROOT = SEP.ROOT
VARIANTS = {"plain": (0.0, 0.0), "ap0.2": (0.2, 0.0), "ap0.4": (0.4, 0.0), "jit0.01": (0.0, 0.01), "jit0.02": (0.0, 0.02), "ap0.3+jit0.015": (0.3, 0.015)}


def parse(name: str) -> dict:
    out = {"ap": 0.0, "jit": 0.0, "tilt": 0.0, "bw": False}
    for t in re.findall(r"ap[0-9.]+|jit[0-9.]+|tilt[+-][0-9.]+|bw|plain", name):
        if t.startswith("ap"):
            out["ap"] = float(t[2:])
        elif t.startswith("jit"):
            out["jit"] = float(t[3:])
        elif t.startswith("tilt"):
            out["tilt"] = float(t[4:])
        elif t == "bw":
            out["bw"] = True
    return out


def rough(x: np.ndarray, p: dict, ap_add: float, jit: float, seed: int, tilt: float = 0.0, bw: bool = False) -> np.ndarray:
    import pyworld
    sr = P.SR
    f0, sp, ap = W.analyze(x, sr)
    if p["alpha"] != 1.0:
        sp, ap = W.warp(sp, p["alpha"]), np.clip(W.warp(ap, p["alpha"]), 0.0, 1.0)
    if tilt != 0.0:
        hz = np.maximum(np.arange(sp.shape[1]) * sr / (2 * (sp.shape[1] - 1)), 50.0)
        sp = np.ascontiguousarray(sp * 10 ** (tilt * np.log2(hz / 1000.0) / 10)[None])
    if bw:
        c = np.fft.irfft(np.log(np.maximum(sp, 1e-20)), axis=1)
        c[:, 41:-40] = 0.0
        sp = np.ascontiguousarray(np.exp(np.fft.rfft(c, axis=1).real))
    f0 = f0 * p["r"]
    g = np.random.default_rng(seed)
    if jit > 0:
        e = np.convolve(g.standard_normal(len(f0)), np.ones(3) / np.sqrt(3), mode="same")
        f0 = np.where(f0 > 0, f0 * (1 + jit * e), 0.0)
    if ap_add > 0:
        hz = np.arange(ap.shape[1]) * sr / (2 * (ap.shape[1] - 1))
        w = np.clip((4000.0 - hz) / 1000.0, 0.0, 1.0)
        ap = np.clip(ap + ap_add * (1 - ap) * w[None], 0.0, 1.0)
    y = pyworld.synthesize(np.ascontiguousarray(f0), sp, ap, sr, 5.0)[: len(x)].astype(np.float32)
    return y


def main() -> int:
    import librosa
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    M = MK.load(ROOT / "results" / opt.get("mask", "snv_p45_mask") / "last.pt", PP.DEV)
    man = json.loads((ROOT / "data/snv_p_feat_x16/manifest.json").read_text())["voices"]
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    va = SEP.utterances(split["val"])
    ok = [w for (q, w) in va if w.parent.parent.name == "snv_e1" and SEP.xop_wav(w).exists()]
    g = np.random.default_rng(51)
    pick = [ok[i] for i in g.choice(len(ok), size=min(int(opt.get("n", 72)), len(ok)), replace=False)]
    pool = SEP.bg_split()["dev"]["music"]
    names = opt["variants"].split(",") if "variants" in opt else list(VARIANTS)
    res: dict = {"prereg": "results/snv_p52/prereg.yaml(step_a)", "mask": opt.get("mask", "snv_p45_mask"), "variants": {n: parse(n) for n in names}, "sets": {}}
    if opt.get("ref") == "cv":
        cman = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
        dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())["ref"]
        cv = [(sp_["speaker"], PP.T.FEAT / f"cv/{sp_['speaker']}_{c['name']}.wav") for sp_ in [x for x in cman["speakers"] if x["split"] == "devA"][:64] for c in sp_["src"][:2] if dv[f"{sp_['speaker']}_{c['name']}"]["text_vs_orig"] <= 0.1]
        rows = []
        for k, (vid, path) in enumerate(cv):
            y = librosa.load(str(path), sr=P.SR, mono=True)[0].astype(np.float64)
            mus = pool[zlib.crc32(f"cv|{vid}|{k}".encode()) % len(pool)]
            rows.append({"voice": vid} | PP.mask_stats(M, y, mus, 1 + k))
        res["sets"]["cv"] = {"n_items": len(rows), "rows": rows}
        print("cv", len(rows), flush=True)
    for name in names:
        vv = parse(name)
        ap_add, jit = vv["ap"], vv["jit"]
        rows = []
        for k, w in enumerate(pick):
            voice = w.parent.name
            p = man[voice]
            x = librosa.load(str(w), sr=P.SR, mono=True)[0].astype(np.float32)
            y = rough(x, p, ap_add, jit, CH.seed_of(f"p52|{voice}|{w.stem}"), vv["tilt"], vv["bw"])
            y = CH.apply(y, p["room"], np.random.default_rng(CH.seed_of(f"p16|{voice}|e1|{w.stem}")))
            y = np.clip(np.asarray(y, np.float64), -1.0, 1.0)
            mus = pool[zlib.crc32(f"version|{voice}|{k}".encode()) % len(pool)]
            st = PP.mask_stats(M, y, mus, 1 + k)
            sg = PP.sig_stats(y, M)
            rows.append({"voice": voice} | st | {f"sig|{a}": sg.get(a) for a in ("comb_db", "per_lo", "per_hi", "jitter", "f0_med")})
        per: dict = {}
        for r in rows:
            per.setdefault(r["voice"], []).append(r)
        summ = {key: round(float(np.mean([np.mean([r[key] for r in rs if r[key] is not None]) for rs in per.values()])), 4) for key in rows[0] if "|" in key}
        res["sets"][name] = {"n_items": len(rows), "n_voices": len(per), "summary": summ, "rows": rows}
        print(name, {k_: summ[k_] for k_ in ("mus0|voice_kept", "mus0|ideal_voice_kept", "mus0|noise_kept", "mus0|kept_b0v", "mus0|ideal_b0v", "mus10|kept_b0v", "mus10|ideal_b0v", "sig|comb_db", "sig|per_lo")}, flush=True)
    out = ROOT / opt.get("out", "results/snv_p52/rough.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
