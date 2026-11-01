"""D680: 合成の土台の天井を官能指数 v1 で測る(変換なしの写し合成)。本物の女声 30 本(D678 の test_pool = 検出器の学習に入っていない話者)の 8 s を:
ORIG = 録音そのもの・W0 = harvest + CheapTrick + D4C の WORLD の往復・W24P = 包絡を DCT 1〜24 に縮め非周期性を 4 帯の周期性から(WORLD の合成)・
RC = 構音の描画そのもの(c0 = PA.envelope の 0 次・行 = 1〜24 次 / 10・形のパワーの 25 Hz・描画の較正の曲線 1・周期性 → 非周期性・causal_synth = 製品と同じ式・慣性と γ なし)。
値 = 官能指数 v1 の値 − τ(W0 / W24P = world を外した fold・RC = render を外した fold・ORIG = none)。規則は results/conv_p0/d680/prereg.yaml。
    uv run python d680_copy_ceiling.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import scipy.signal as ss
import soundfile as sf
import torch

import pae as PA
from causal_synth import synth_causal
from d634_artic_render import env_from_rows
from d635_split import norm
from d639_knob_hf import ap_from_per
from d652_render_cal import curve as curve1
from d677_render_ablate import env4

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/conv_p0/d680"
W = Path("/home/kojirotanaka/lightvc_scratch_20261013/d680")
SR = 48000


def job(it: tuple) -> str:
    i, path = it
    from d651_copy_hf import best8
    from d6_unit_cov import load48
    torch.set_num_threads(1)
    x = best8(load48(Path(path)).astype(np.float64))
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    f0, t = pw.harvest(ss.resample_poly(x, 1, 3), 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    T = len(f0)
    xa = np.concatenate([np.zeros(PA.A), x, np.zeros(PA.A + 240 * 2)])
    env = PA.envelope(xa, f0)
    per = PA.periodicity(torch.from_numpy(xa)[None].float(), torch.from_numpy(f0)[None].float())[0].numpy().T.astype(np.float64)[:T]
    rows = env[1:25].T.astype(np.float64)[:T] / 10.0
    c0 = env[0].astype(np.float64)[:T]
    e24 = norm(env_from_rows(rows), sp.mean(1, keepdims=True))
    ys = {"ORIG": x, "W0": pw.synthesize(f0, sp, ap, SR, 5.0), "W24P": pw.synthesize(f0, e24, ap_from_per(per, f0), SR, 5.0),
          "RC": synth_causal(f0, env4(c0, rows, True) * curve1(), ap_from_per(per, f0))}
    for a, y in ys.items():
        y = np.asarray(y, dtype=np.float64)
        (W / a).mkdir(parents=True, exist_ok=True)
        sf.write(W / a / f"r{i:03d}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), SR, subtype="PCM_16")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    L = json.loads((ROOT / "results/conv_p0/d678/lists.json").read_text())
    its = list(enumerate(r["path"] for r in L["real_test_pool"][:30]))
    with Pool(10) as pl:
        pl.map(job, its)
    from kansei_v1 import KanseiV1
    rep = {}
    for a, fold in (("ORIG", "none"), ("W0", "world"), ("W24P", "world"), ("RC", "render")):
        m = KanseiV1(fold).margin([W / a / f"r{i:03d}.wav" for i, _ in its])
        rep[a] = {"fold": fold, "median": round(float(np.median(m)), 2), "p10_p90": [round(float(np.percentile(m, 10)), 2), round(float(np.percentile(m, 90)), 2)], "pass": round(float(np.mean(np.array(m) <= 0)), 3)}
        print(a, rep[a], flush=True)
    (OUT / "ceiling.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
