"""D640: つまみの縮みの切り分け(dev2 k7 の 12 本)。CALS = 錨の作りの包絡を DCT 24 に縮めて一律に伸縮・ENRA = ENR の行 + 非周期性にも α(記録)。規則は results/conv_p0/d640/prereg.yaml。
    uv run python d640_knob_smooth.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np

from d631_world_swap import C, G, H, ROOT, analyze, keys
from d632_f0env_swap import fit, lag, shift, warp
from d634_artic_render import PRE, env_from_rows, lift, synth, vdir
from d635_split import norm
from d637_envnet_eval import env_rows
from d638_artic_gate_g import env_shift

OUT = ROOT / "results/conv_p0/d640"


def one(key: str) -> dict:
    s, k = "dev2", 7
    from d6_unit_cov import load48
    xs = load48(H / s / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * 48000]
    f0s, sps, aps = analyze(xs)
    sg, apg0 = lift(warp(sps, G[k][1])), warp(aps, G[k][1])
    f0g = f0s * 2 ** (k / 12)
    vo = f0g > 0
    ys = {a: synth(f0g, sg if a == 1.0 else warp(sg, a), apg0) for a in (1.0, 0.9, 1.1)}
    sp = {a: analyze(y.astype(np.float64))[1] for a, y in ys.items()}
    rec = {"key": key}
    for a in (0.9, 1.1):
        rec[f"cals_render_{a}"] = env_shift(sp[1.0], sp[a], vo)
        rec[f"cals_env_{a}"] = env_shift(sg, warp(sg, a), vo)
    d = H / s / "d570"
    v = np.fromfile(vdir(s, k) / f"{key}.vcond", "<f4").reshape(-1, 27)[PRE:PRE + 1600].astype(np.float64)
    xc = load48(d / C[k] / f"{key}_{C[k]}.wav").astype(np.float64)
    lg = lag(xc, xs)
    _, sps2, aps2 = (shift(m, lg) for m in (f0s, sps, aps))
    n = len(v)
    apg = fit(warp(aps2, G[k][1]), n)
    pin = fit(sps2, n).mean(1, keepdims=True)
    f0 = np.where(v[:, 26] > 0.5, 200.0 * np.exp(v[:, 25]), 0.0)
    r = env_rows(s, k, key)["ENR"].astype(np.float64)
    y1 = synth(f0, norm(env_from_rows(r), pin), apg)
    s1 = analyze(y1.astype(np.float64))[1]
    for a in (0.9, 1.1):
        ya = synth(f0, norm(env_from_rows(r, a), pin), np.clip(warp(apg, a), 0.0, 1.0))
        rec[f"enra_render_{a}"] = env_shift(s1, analyze(ya.astype(np.float64))[1], f0 > 0)
    return rec


def main() -> int:
    with Pool(12) as pl:
        recs = pl.map(one, keys("dev2")[:12])
    enr = json.loads((ROOT / "results/conv_p0/d638/knob.json").read_text())
    med = lambda f: round(float(np.median([r[f] for r in recs])), 3)
    rep = {"items": recs, "cals_render": {x: med(f"cals_render_{x}") for x in (0.9, 1.1)}, "cals_env": {x: med(f"cals_env_{x}") for x in (0.9, 1.1)},
           "enra_render": {x: med(f"enra_render_{x}") for x in (0.9, 1.1)}, "enr_render": enr["median_render"], "enr_env": enr["median_env"]}
    rep["smooth_explains"] = bool(all(abs(rep["cals_render"][x] - enr["median_render"][str(x)]) <= 0.015 and abs(rep["cals_env"][x] - enr["median_env"][str(x)]) <= 0.015 for x in (0.9, 1.1)))
    (OUT / "knob_smooth.json").write_text(json.dumps(rep, indent=1))
    print({k_: v_ for k_, v_ in rep.items() if k_ != "items"}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
