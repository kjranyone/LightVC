"""D638: 構音の描画 ENR(D637)の関門 G(d109 v2 の 4 軸・gate_mf_v5 の式)での位置と、声道長のつまみの検査(傾きを除いた測り方・合成音で)。開発の組 dev2 / dev3。
--stage links: d620 の名前の規約(d570/OURS{k}{版}/{key}_OURS{k}{版}.wav)に合わせた写しの置き場(シンボリックリンク)を作る。
--stage knob: dev2 k7 の 12 本で α = 0.9 / 1.1 の合成音の包絡のずれ(対数周波数 300〜5000 Hz・直線の傾きを除いた相互相関の中央値)。規則は results/conv_p0/d638/prereg.yaml。
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np

from d631_world_swap import C, G, H, ROOT, SETS, analyze, keys
from d632_f0env_swap import fit, lag, shift, warp
from d634_artic_render import FQ, PRE, env_from_rows, synth, vdir
from d635_split import norm
from d637_envnet_eval import env_rows

OUT = ROOT / "results/conv_p0/d638"
VERS = {"ENR": "ENR", "EN": "EN", "PT": "PT", "GU": None}
LF = np.linspace(np.log(300.0), np.log(5000.0), 400)


def links() -> None:
    n = 0
    for s in SETS:
        d = H / s / "d570"
        for k in C:
            for v, src in VERS.items():
                sd = d / (G[k][0] if src is None else f"{src}_{k}")
                sn = G[k][0] if src is None else f"{src}_{k}"
                od = d / f"OURS{k}{v}"
                od.mkdir(exist_ok=True)
                for f in sorted(sd.glob(f"p*_{sn}.wav")):
                    o = od / f"{f.name[:10]}_OURS{k}{v}.wav"
                    if not o.exists():
                        o.symlink_to(f)
                    n += 1
    print("links", n, flush=True)


def det(a: np.ndarray) -> np.ndarray:
    return a - np.polyval(np.polyfit(LF, a, 1), LF)


def env_shift(s0: np.ndarray, s1: np.ndarray, vo: np.ndarray) -> float:
    step = LF[1] - LF[0]
    out = []
    for t in np.flatnonzero(vo[: min(len(s0), len(s1))])[::4]:
        x0 = det(np.log(np.interp(np.exp(LF), FQ, s0[t]) + 1e-20))
        x1 = det(np.log(np.interp(np.exp(LF), FQ, s1[t]) + 1e-20))
        cc = [float(np.dot(x0[max(0, -m):len(x0) - max(0, m)], x1[max(0, m):len(x1) - max(0, -m)])) for m in range(-30, 31)]
        out.append(np.exp((int(np.argmax(cc)) - 30) * step))
    return float(np.median(out)) if out else float("nan")


def knob(key: str) -> dict:
    s, k = "dev2", 7
    from d6_unit_cov import load48
    d = H / s / "d570"
    v = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)[PRE:PRE + 1600].astype(np.float64)
    xc = load48(d / C[k] / f"{key}_{C[k]}.wav").astype(np.float64)
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    lg = lag(xc, xs)
    _, sps, aps = (shift(m, lg) for m in analyze(xs))
    n = len(v)
    apg = fit(warp(aps, G[k][1]), n)
    pin = fit(sps, n).mean(1, keepdims=True)
    f0 = np.where(v[:, 26] > 0.5, 200.0 * np.exp(v[:, 25]), 0.0)
    r = env_rows(s, k, key)["ENR"].astype(np.float64)
    ys = {a: synth(f0, norm(env_from_rows(r, a), pin), apg) for a in (1.0, 0.9, 1.1)}
    sp = {a: analyze(y.astype(np.float64))[1] for a, y in ys.items()}
    vo = f0 > 0
    rec = {"key": key, "alpha1_identity": bool(np.array_equal(env_from_rows(r, 1.0), env_from_rows(r)))}
    for a in (0.9, 1.1):
        rec[f"render_{a}"] = env_shift(sp[1.0], sp[a], vo)
        rec[f"env_{a}"] = env_shift(env_from_rows(r), env_from_rows(r, a), vo)
    return rec


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", required=True, choices=["links", "knob"])
    a = ap_.parse_args()
    if a.stage == "links":
        links()
        return 0
    with Pool(12) as pl:
        recs = pl.map(knob, keys("dev2")[:12])
    lo = sum(r["render_0.9"] <= 0.96 for r in recs)
    hi = sum(r["render_1.1"] >= 1.04 for r in recs)
    rep = {"items": recs, "n_lo_ok": lo, "n_hi_ok": hi, "alpha1_identity": all(r["alpha1_identity"] for r in recs),
           "median_render": {x: round(float(np.median([r[f"render_{x}"] for r in recs])), 3) for x in (0.9, 1.1)},
           "median_env": {x: round(float(np.median([r[f"env_{x}"] for r in recs])), 3) for x in (0.9, 1.1)}}
    rep["knob_ok"] = bool(rep["alpha1_identity"] and lo >= 10 and hi >= 10)
    (OUT / "knob.json").write_text(json.dumps(rep, indent=1))
    print("knob", {k_: v_ for k_, v_ in rep.items() if k_ != "items"}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
