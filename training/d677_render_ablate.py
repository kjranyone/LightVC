"""D677: 構音の描画(基準 G5 = PR5 + γ 0.5・D669)の粗さ(ザラザラ)と止まった動き(ラジオ)の源を 1 つずつ外す(開発の組・推論だけ)。官能指数 v0(D676)の軸で本物の帯との距離を測る。
腕: B = G5・J = f0 の 30 Hz の慣性なし(条件の log f0 そのまま)・C = c0 の 30 Hz の慣性なし + γ 1・S = 形のパワーの 25 Hz のならしなし・A = 周期性 → 非周期性の HNR を 30·per → 15·per・ALL = J + C + S + A。
組 = dev2 / dev3 の各 k(5 / 7 / 12)の名前順の先頭 20 組。合成 = causal_synth(製品と同じ式)。規則は results/conv_p0/d677/prereg.yaml。
    uv run python d677_render_ablate.py --stage render
    uv run --with mosqito==1.2.1 python d677_render_ablate.py --stage feat
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

import d637_envnet_eval as E7
import pae as PA
from causal_synth import synth_causal
from d631_world_swap import H, ROOT
from d634_artic_render import DCT, FC, FQ, PRE
from d636_traj_gain import onepole
from d639_knob_hf import HBC
from d652_render_cal import curve as curve1
from d653_causal_render import parts
from d666_target_cal import curve3
from d669_level_comp import comp

OUT = ROOT / "results/conv_p0/d677"
A25 = float(np.exp(-2 * np.pi * 25.0 * 0.005))
ARMS = {"B": {}, "J": {"f0raw": True}, "C": {"c0raw": True}, "S": {"noshape": True}, "A": {"hnr": 15.0}, "ALL": {"f0raw": True, "c0raw": True, "noshape": True, "hnr": 15.0}}


def keys(s: str, k: int) -> list:
    return sorted(f.name[:10] for f in (H / s / "d570" / f"G5_{k}").glob("p*.wav"))[:20]


def env4(c0: np.ndarray, rows: np.ndarray, smooth: bool) -> np.ndarray:
    lm = (rows * 10.0) @ DCT[1:25]
    e = np.exp(2.0 * np.stack([np.interp(FQ, FC, r) for r in lm]))
    s = e.mean(1)
    sb = np.exp(onepole(np.log(s)[:, None], A25)[:, 0]) if smooth else s
    return e / s[:, None] * sb[:, None] * np.exp(2.0 * c0 / np.sqrt(128.0))[:, None]


def ap_hnr(per: np.ndarray, f0: np.ndarray, k_hnr: float) -> np.ndarray:
    a2 = 1.0 / (1.0 + 10.0 ** (k_hnr * np.clip(per, 0.0, 1.5) / 10.0))
    lf, lc = np.log(np.maximum(FQ, 1.0)), np.log(HBC)
    ap = np.sqrt(np.stack([np.interp(lf, lc, r) for r in a2]))
    ap[f0 <= 0] = 1.0
    return np.clip(ap, 1e-3, 1.0)


def render(it: tuple) -> int:
    s, k, key = it
    pr = parts(s, k, key)
    rows, _, f0s = pr["PR1"]
    _, _, f0r = pr["PR0"]
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float64)
    cv = curve1() * curve3(s, int(key[7:10]))
    n = 0
    for arm, o_ in ARMS.items():
        o = H / s / "d570" / f"D677{arm}_{k}" / f"{key}_D677{arm}_{k}.wav"
        if o.exists():
            continue
        o.parent.mkdir(parents=True, exist_ok=True)
        if o_.get("c0raw"):
            c0 = (10.0 * rc[:, 0] + PA.C0_SIL)[PRE:PRE + 1600]
        else:
            c0 = comp(rc[:, 0], 0.5)[PRE:PRE + 1600]
        f0 = f0r if o_.get("f0raw") else f0s
        sp = env4(c0, rows, not o_.get("noshape")) * cv
        sf.write(o, synth_causal(f0, sp, ap_hnr(pr["per"], f0, o_.get("hnr", 30.0))), 48000, subtype="PCM_16")
        n += 1
    return n


def feat(it: tuple) -> dict:
    import d676_kansei_band as K
    s, k, key, arm = it
    p = H / s / "d570" / f"D677{arm}_{k}" / f"{key}_D677{arm}_{k}.wav"
    return K.feat({"sess": f"D677|{s}|{k}", "path": str(p), "role": arm, "sys": key})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["render", "feat"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    its = [(s, k, key) for s in ("dev2", "dev3") for k in (5, 7, 12) for key in keys(s, k)]
    npr = int(os.environ.get("LVC_NPROC", "12"))
    if a.stage == "render":
        with Pool(npr) as pl:
            print("rendered", sum(pl.map(render, its, chunksize=2)), flush=True)
        return 0
    with Pool(npr) as pl:
        res = list(pl.imap_unordered(feat, [it + (arm,) for it in its for arm in ARMS], chunksize=2))
    (OUT / "feats.json").write_text(json.dumps(res))
    print("feats", len(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
