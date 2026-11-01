"""D668: k5 の shimmer の源を分ける(基準 PR5・k5 だけ・開発の組 dev2 / dev3)。L2 = 大きさ(c0)の軌道を 1 極 2 Hz でならす・F2 = f0 を有声の続きの中で 2 Hz でならす・LF2 = 両方。
診断(製品にしない)。規則は results/conv_p0/d668/prereg.yaml。
    uv run python d668_k5_shimmer.py
"""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

import d637_envnet_eval as E7
import pae as PA
from causal_synth import synth_causal
from d631_world_swap import H, SETS, keys
from d634_artic_render import PRE
from d636_traj_gain import onepole
from d639_knob_hf import ap_from_per
from d652_render_cal import curve as curve1
from d653_causal_render import parts
from d663_shape_power import env4
from d666_target_cal import curve3

A2 = float(np.exp(-2 * np.pi * 2.0 * 0.005))


def sm_f0(f0: np.ndarray) -> np.ndarray:
    v = f0 > 0
    lf = np.where(v, np.log(np.maximum(f0, 1.0)), 0.0)
    y = lf.copy()
    for t in range(1, len(lf)):
        if v[t] and v[t - 1]:
            y[t] = A2 * y[t - 1] + (1 - A2) * lf[t]
    return np.where(v, np.exp(y), 0.0)


def render(it: tuple) -> int:
    s, key = it
    k = 5
    pr = parts(s, k, key)
    rows, c0, f0 = pr["PR1"]
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float64)
    c0all = 10.0 * rc[:, 0] + PA.C0_SIL
    c02 = onepole(c0all[:, None], A2)[:, 0][PRE:PRE + 1600]
    cv = curve1() * curve3(s, int(key[7:10]))
    f2 = sm_f0(f0)
    for a, cc, ff in (("L2", c02, f0), ("F2", c0, f2), ("LF2", c02, f2)):
        o = H / s / "d570" / f"OURS5{a}" / f"{key}_OURS5{a}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            sf.write(o, synth_causal(ff, env4(cc, rows) * cv, ap_from_per(pr["per"], ff)), 48000, subtype="PCM_16")
    return 1


def main() -> int:
    its = [(s, key) for s in SETS for key in keys(s)]
    with Pool(int(os.environ.get("LVC_NPROC", "12"))) as pl:
        print("rendered", sum(pl.map(render, its, chunksize=2)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
