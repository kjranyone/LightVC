"""D645: 構音の描画の高域不足(EN2P 対 PT)を、有声 / 無声の発声のフレームに分ける。
--stage env: 合成に入るパワー包絡(行 → env_from_rows → 入力の CheapTrick の平均パワーで正規化)の 4–10 kHz / 0.1–4 kHz の長時間の比(d109 の hf と同じ帯)を、全体・有声・無声で(PT・EN2P・TAB)。
--stage render: EUV = 有声は EN2P・無声は PT の行 / EVV = 有声は PT・無声は EN2P の行(診断の混ぜ物・非周期性と f0 と大きさは EN2P と同じ)。
規則は results/conv_p0/d645/prereg.yaml。
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

import d637_envnet_eval as E7
from d631_world_swap import C, H, ROOT, SETS, keys
from d634_artic_render import FQ, PRE, env_from_rows, synth, vdir
from d635_split import norm
from d644_envnet2_eval import base

OUT = ROOT / "results/conv_p0/d645"
E7.NET = ROOT / "results/diag_envnet2/last.pt"
HI, LO = (FQ >= 4000) & (FQ < 10000), (FQ >= 100) & (FQ < 4000)


def rows3(s: str, k: int, key: str) -> dict:
    er = E7.env_rows(s, k, key)
    v = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float64)
    post = (v[:, 1:25] - rc[:, 295:])[PRE:PRE + 1600]
    return {"PT": er["PT"].astype(np.float64), "EN2P": er["ENR"].astype(np.float64), "TAB": er["raw"].astype(np.float64) + post}


def env_job(it: tuple) -> dict:
    s, k, key = it
    rows, f0, pin, _ = base(s, k, key)
    v = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)[PRE:PRE + 1600]
    c0 = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319)[PRE:PRE + 1600, 0].astype(np.float64)
    sp_ = c0 > 0.05
    cls = {"all": sp_, "voiced": sp_ & (v[:, 26] > 0.5), "unvoiced": sp_ & (v[:, 26] <= 0.5)}
    r3 = rows3(s, k, key)
    out = {"key": f"{s}|{k}|{key}", "n": {c: int(m.sum()) for c, m in cls.items()}}
    for a, r in r3.items():
        e = norm(env_from_rows(r), pin)
        h, l_ = e[:, HI].sum(1), e[:, LO].sum(1)
        out[a] = {c: {"hi": float(h[m].sum()), "lo": float(l_[m].sum())} for c, m in cls.items()}
    return out


def render(it: tuple) -> int:
    s, k, key = it
    rows, f0, pin, ap = base(s, k, key)
    r3 = rows3(s, k, key)
    vo = (f0 > 0)[:, None]
    for a, r in (("EUV", np.where(vo, r3["EN2P"], r3["PT"])), ("EVV", np.where(vo, r3["PT"], r3["EN2P"]))):
        o = H / s / "d570" / f"{a}_{k}" / f"{key}_{a}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            sf.write(o, synth(f0, norm(env_from_rows(r), pin), ap), 48000, subtype="PCM_16")
    return 1


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["env", "render"])
    a = ap_.parse_args()
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    with Pool(int(os.environ.get("LVC_NPROC", "12"))) as pl:
        recs = pl.map(env_job if a.stage == "env" else render, its, chunksize=2)
    if a.stage == "render":
        print("rendered", sum(recs), flush=True)
        return 0
    rep: dict = {}
    for k in C:
        rk = [r for r in recs if r["key"].split("|")[1] == str(k)]
        rep[k] = {}
        for arm in ("PT", "EN2P", "TAB"):
            rep[k][arm] = {}
            for c in ("all", "voiced", "unvoiced"):
                hs = np.array([r[arm][c]["hi"] for r in rk])
                ls = np.array([r[arm][c]["lo"] for r in rk])
                rep[k][arm][c] = round(float(np.median(10 * np.log10((hs + 1e-30) / (ls + 1e-30)))), 2)
        allhi = {arm: np.array([r[arm]["all"]["hi"] for r in rk]) for arm in ("PT", "EN2P")}
        rep[k]["share_of_hi_unvoiced_PT"] = round(float(np.median([r["PT"]["unvoiced"]["hi"] / (r["PT"]["all"]["hi"] + 1e-30) for r in rk])), 3)
        rep[k]["share_of_hi_unvoiced_EN2P"] = round(float(np.median([r["EN2P"]["unvoiced"]["hi"] / (r["EN2P"]["all"]["hi"] + 1e-30) for r in rk])), 3)
        rep[k]["unvoiced_frames_frac"] = round(float(np.median([r["n"]["unvoiced"] / max(1, r["n"]["all"]) for r in rk])), 3)
        print(k, json.dumps(rep[k]), flush=True)
    (OUT / "env.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
