"""D648 の付帯: envnet3 の推論の有声のフレームの暗さ(合成に入る包絡の 4–10 / 0.1–4 kHz の長時間の比・入力の CheapTrick の平均パワーで重みづけ)を、入力を学習の値に替えて切り分ける。
C2N = C2 を学習の平均(D643 と同じ取り出し)・C0N = c0n を学習の中央値へ(D639 の δ)・LFN = log f0 を学習の中央値へ(D643 の差)・ALL = 3 つとも。規則は results/conv_p0/d648/prereg.yaml。
    uv run python d648_dark.py
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np
import torch

import d637_envnet_eval as E7
from d631_world_swap import C, ROOT, SETS, keys
from d634_artic_render import FQ, PRE, env_from_rows, vdir
from d635_split import norm
from d648_envnet3_eval import base

OUT = ROOT / "results/conv_p0/d648"
E7.NET = ROOT / "results/diag_envnet3/last.pt"
HI, LO = (FQ >= 4000) & (FQ < 10000), (FQ >= 100) & (FQ < 4000)
ST: dict = {}


def job(it: tuple) -> dict:
    s, k, key = it
    torch.set_num_threads(1)
    _, f0, pin, _ = base(s, k, key)
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float32)
    vc = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)
    post = (vc[:, 1:25] - rc[:, 295:])
    sl = slice(PRE, PRE + 1600)
    m = (rc[sl, 0] > 0.05) & (f0 > 0)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T))[None]
    c2m = np.broadcast_to(np.array(ST["c2_mean"], np.float32), (len(rc), 64))
    c0s = np.where(rc[:, 0:1] > 0.05, rc[:, 0:1] + ST["dc0"], rc[:, 0:1]).astype(np.float32)
    lfs = np.where(rc[:, 26:27] > 0.5, rc[:, 25:26] + ST["dlf"], rc[:, 25:26]).astype(np.float32)
    var = {"EN": (rc[:, 0:1], rc[:, 25:26], rc[:, 231:295]), "C2N": (rc[:, 0:1], rc[:, 25:26], c2m), "C0N": (c0s, rc[:, 25:26], rc[:, 231:295]),
           "LFN": (rc[:, 0:1], lfs, rc[:, 231:295]), "ALL": (c0s, lfs, c2m)}
    rows = {"TAB": (rc[:, 1:25] + post)[sl], "PT": vc[sl, 1:25]}
    for a, (c0, lf, c2) in var.items():
        with torch.no_grad():
            y = E7.net()(t(rc[:, 31:231]), t(c0), t(lf), t(rc[:, 26:27]), t(rc[:, 1:25]), t(c2))[0].numpy().T
        rows[a] = (y + post)[sl]
    out = {"key": f"{s}|{k}|{key}"}
    for a, r in rows.items():
        e = norm(env_from_rows(r.astype(np.float64)), pin)
        out[a] = float(10 * np.log10(e[m][:, HI].sum() / e[m][:, LO].sum()))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    c0 = json.loads((ROOT / "results/conv_p0/d639/c0stat.json").read_text())
    d3 = json.loads((ROOT / "results/conv_p0/d643/result.json").read_text())
    from d643_infer_inputs import train_stats
    ts = train_stats()
    ST.update(c2_mean=ts["c2_mean"], dc0=c0["delta"], dlf=d3["lf_shift"])
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    with Pool(int(os.environ.get("LVC_NPROC", "12"))) as pl:
        recs = pl.map(job, its, chunksize=2)
    rep: dict = {"dc0": ST["dc0"], "dlf": ST["dlf"]}
    for k in C:
        rk = [r for r in recs if r["key"].split("|")[1] == str(k)]
        med = {a: round(float(np.median([r[a] for r in rk])), 3) for a in ("TAB", "PT", "EN", "C2N", "C0N", "LFN", "ALL")}
        gap = med["TAB"] - med["EN"]
        rec = {a: round((med[a] - med["EN"]) / gap, 3) if abs(gap) > 1e-6 else None for a in ("C2N", "C0N", "LFN", "ALL")}
        rep[k] = {"median_db": med, "gap_TAB_minus_EN": round(gap, 3), "recovered": rec}
        print(k, json.dumps(rep[k]), flush=True)
    rep["cause"] = {a: bool(all(rep[k]["recovered"][a] is not None and rep[k]["recovered"][a] >= 0.5 for k in C)) for a in ("C2N", "C0N", "LFN", "ALL")}
    (OUT / "result.json").write_text(json.dumps(rep, indent=1))
    print("cause", rep["cause"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
