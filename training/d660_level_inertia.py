"""D660: shimmer(k5 / k7)の切り分け。PR2(D659)の大きさ(条件の c0n の対数)の因果な 1 極の慣性を 30 Hz → 15 Hz(L15)/ 8 Hz(L8)に強める。f0 の慣性・包絡・較正は PR2 と同じ。
規則は results/conv_p0/d660/prereg.yaml。
    uv run python d660_level_inertia.py
"""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

import d637_envnet_eval as E7
import pae as PA
from d631_world_swap import C, H, SETS, keys
from d634_artic_render import PRE, synth
from d636_traj_gain import onepole
from d639_knob_hf import ap_from_per
from d653_causal_render import env_full, parts
from d659_table_cal import curve2

ARMS = {"L15": 15.0, "L8": 8.0}


def render(it: tuple) -> int:
    s, k, key = it
    pr = parts(s, k, key)
    rows, _, f0 = pr["PR1"]
    rc = np.fromfile(E7.rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, 319).astype(np.float64)
    c0 = 10.0 * rc[:, 0] + PA.C0_SIL
    cv = curve2()
    for a, fc in ARMS.items():
        o = H / s / "d570" / f"{a}_{k}" / f"{key}_{a}_{k}.wav"
        o.parent.mkdir(parents=True, exist_ok=True)
        if not o.exists():
            c0s = onepole(c0[:, None], float(np.exp(-2 * np.pi * fc * 0.005)))[:, 0][PRE:PRE + 1600]
            sf.write(o, synth(f0, env_full(c0s, rows) * cv, ap_from_per(pr["per"], f0)), 48000, subtype="PCM_16")
    return 1


def main() -> int:
    its = [(s, k, key) for s in SETS for k in C for key in keys(s)]
    with Pool(int(os.environ.get("LVC_NPROC", "12"))) as pl:
        print("rendered", sum(pl.map(render, its, chunksize=2)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
