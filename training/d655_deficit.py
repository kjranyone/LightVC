"""D655: 構音の描画が目標本人より暗く傾きが急な不足(D654: hf 約 4 dB・tilt 約 2 dB/oct)を、合成に入る包絡の段で部品に分ける(描画なし・0 GPU)。
大きさ = 条件の c0n(PR1 と同じ 30 Hz の慣性)・描画の定数の較正の曲線を掛けたパワー包絡の、発声の区間(最大から 40 dB 以内)の長時間のパワーで hf と tilt(d109 と同じ帯)。
T = 表の引き(+ H1 EQ)・TP = T + 遅い揺らぎ + R の ΔS・TPN = TP + 網の速い補正(= PR1 の行)・REF = 目標本人の参照の録音(音で測る・D654 と同じ)。規則は results/conv_p0/d655/prereg.yaml。
    uv run python d655_deficit.py
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np

from d631_world_swap import C, H, ROOT, SETS, keys
from d634_artic_render import FQ
from d651_copy_hf import CF, best8, hf_tilt
from d653_causal_render import env_full, parts

OUT = ROOT / "results/conv_p0/d655"


def env_stats(e: np.ndarray) -> tuple:
    p = e.sum(1)
    lv = 10 * np.log10(p + 1e-30)
    act = lv > lv.max() - 40
    P = e[act].sum(0)
    hf = float(10 * np.log10(P[(FQ >= 4000) & (FQ < 10000)].sum() / P[(FQ >= 100) & (FQ < 4000)].sum()))
    L = np.array([P[(FQ >= c * 2 ** (-1 / 6)) & (FQ < c * 2 ** (1 / 6))].sum() for c in CF])
    L = 10 * np.log10(L + 1e-30)
    sel = (CF >= 300) & (CF <= 8000)
    return hf, float(np.polyfit(np.log2(CF[sel]), L[sel], 1)[0])


def job(it: tuple) -> dict:
    s, k, key = it
    from d6_unit_cov import load48
    import d637_envnet_eval as E7
    from d634_artic_render import PRE, vdir
    pr = parts(s, k, key)
    rows, c0, _ = pr["PR1"]
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float64)
    vc = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27).astype(np.float64)
    sl = slice(PRE, PRE + 1600)
    tab = rc[sl, 1:25]
    post = (vc[:, 1:25] - rc[:, 295:])[sl]
    out = {"s": s, "k": k, "key": key}
    for a, r in (("T", tab), ("TP", tab + post), ("TPN", rows)):
        out[a] = env_stats(env_full(c0, r))
    out["REF"] = hf_tilt(best8(load48(H / s / "d37" / "_work" / f"ref_{int(key[7:10]):03d}.wav").astype(np.float64)))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    with Pool(int(os.environ.get("LVC_NPROC", "12"))) as pl:
        recs = pl.map(job, its, chunksize=2)
    rep: dict = {}
    for k in C:
        rk = [r for r in recs if r["k"] == k]
        rep[k] = {}
        for i, st in enumerate(("hf", "tilt")):
            rep[k][st] = {a: round(float(np.median([r[a][i] - r["REF"][i] for r in rk])), 3) for a in ("T", "TP", "TPN")}
        print(k, json.dumps(rep[k]), flush=True)
    (OUT / "deficit.json").write_text(json.dumps({"by_k": rep, "items": recs}, indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
