"""D656: 推論の表の引きが目標本人より暗い源を分ける(包絡の段・0 GPU の描画なし)。表の引きの形のパワー(描画の較正の曲線つき)で hf / tilt を、目標本人の参照(音)との差で読む。
A = 推論の事後・推論の大きさ(c0n + 30 Hz の慣性)= D655 の T・HARD = 事後を最大の単位の 1 つに(混ざりなし)・FLAT = 大きさの重みを有声の発声のフレームで一様に・
OWN = 目標本人の参照で本人の事後(C1)と本人の大きさ(PA の c0)で引いた表(内容が本人のとき)。規則は results/conv_p0/d656/prereg.yaml。
    uv run python d656_table_dark.py
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np
import torch

import d637_envnet_eval as E7
from d631_world_swap import C, H, ROOT, SETS, keys
from d634_artic_render import DCT, FC, FQ, PRE
from d651_copy_hf import CF, best8, hf_tilt
from d653_causal_render import A30, CURVE

OUT = ROOT / "results/conv_p0/d656"


def stats(lm: np.ndarray, w: np.ndarray) -> tuple:
    e = np.exp(2.0 * np.stack([np.interp(FQ, FC, r) for r in lm])) * CURVE
    e = e / e.sum(1, keepdims=True) * w[:, None]
    P = e.sum(0)
    hf = float(10 * np.log10(P[(FQ >= 4000) & (FQ < 10000)].sum() / P[(FQ >= 100) & (FQ < 4000)].sum()))
    L = 10 * np.log10(np.array([P[(FQ >= c * 2 ** (-1 / 6)) & (FQ < c * 2 ** (1 / 6))].sum() for c in CF]) + 1e-30)
    sel = (CF >= 300) & (CF <= 8000)
    return hf, float(np.polyfit(np.log2(CF[sel]), L[sel], 1)[0])


def lookup(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return ((p @ T.T) / 10.0 * 10.0) @ DCT[1:25]


def weights(c0: np.ndarray, act_db: float = 40.0) -> np.ndarray:
    from d636_traj_gain import onepole
    c0s = onepole(c0[:, None], A30)[:, 0]
    lv = c0s / np.sqrt(128) * 2 * 10 / np.log(10)
    w = 10 ** (lv / 10)
    w[lv < lv.max() - act_db] = 0.0
    return w


def job(it: tuple) -> dict:
    s, k, key = it
    import pae as PA
    torch.set_num_threads(1)
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float64)[PRE:PRE + 1600]
    ti = int(key[7:10])
    T = np.fromfile(H / s / "d85_work" / f"en_{ti:03d}" / "enroll.bin", "<f4").reshape(24, -1).astype(np.float64)
    p = np.maximum(rc[:, 31:231], 0.0)
    p /= p.sum(1, keepdims=True) + 1e-12
    v = rc[:, 26] > 0.5
    c0 = 10.0 * rc[:, 0] + PA.C0_SIL
    w = weights(c0) * v
    hard = np.zeros_like(p)
    hard[np.arange(len(p)), p.argmax(1)] = 1.0
    out = {"s": s, "k": k, "key": key, "ti": ti}
    out["A"] = stats(lookup(T, p), w)
    out["HARD"] = stats(lookup(T, hard), w)
    out["FLAT"] = stats(lookup(T, p), (w > 0).astype(np.float64))
    return out


def own(it: tuple) -> dict:
    s, ti = it
    import nvoc as N
    import pae as PA
    import pyworld as pw
    import scipy.signal as ss
    import train_c1 as T1
    import c1_content as CC
    from d6_unit_cov import load48
    dev = "cuda"
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"]); c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    x = best8(load48(H / s / "d37" / "_work" / f"ref_{ti:03d}.wav").astype(np.float64))
    Tn = len(x) // N.HOP - 2
    seg = Tn * N.HOP
    xa = np.concatenate([np.zeros(PA.A), x[:seg], np.zeros(PA.A)])
    f0, _ = pw.harvest(ss.resample_poly(x[:seg + N.HOP], 1, 3), 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
    f0 = np.pad(f0, (0, max(0, Tn - len(f0))))[:Tn]
    env = PA.envelope(xa, f0)
    xc = torch.from_numpy(np.concatenate([np.zeros(T1.PRE), x[:seg]]).astype(np.float32))[None].to(dev)
    with torch.no_grad():
        p = torch.softmax(c1n(mfc(xc))[..., -Tn:], 1)[0].cpu().numpy().T.astype(np.float64)
    T = np.fromfile(H / s / "d85_work" / f"en_{ti:03d}" / "enroll.bin", "<f4").reshape(24, -1).astype(np.float64)
    w = weights(env[0].astype(np.float64)) * (f0 > 0)
    return {"s": s, "ti": ti, "OWN": stats(lookup(T, p), w), "REAL": stats((env[1:25].T.astype(np.float64)) @ DCT[1:25], w), "REF": hf_tilt(x)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    with Pool(int(os.environ.get("LVC_NPROC", "12"))) as pl:
        recs = pl.map(job, its, chunksize=4)
    tis = [(s, ti) for s in SETS for ti in sorted({int(k[7:10]) for k in keys(s)})]
    ow = {f"{r['s']}|{r['ti']}": r for r in map(own, tis)}
    rep: dict = {}
    for i, st in enumerate(("hf", "tilt")):
        rep[st] = {}
        for a in ("A", "HARD", "FLAT"):
            rep[st][a] = {k: round(float(np.median([r[a][i] - ow[f"{r['s']}|{r['ti']}"]["REF"][i] for r in recs if r["k"] == k])), 3) for k in C}
        rep[st]["OWN"] = round(float(np.median([o["OWN"][i] - o["REF"][i] for o in ow.values()])), 3)
        rep[st]["REAL"] = round(float(np.median([o["REAL"][i] - o["REF"][i] for o in ow.values()])), 3)
    (OUT / "result.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
