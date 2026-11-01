"""D469(10-21 査読: fline は amline と同じ定義 = 検定の統計を学習で下げた): 学習の損失が見ていない独立の物差しで、フレーム周期の線が消えたかを確かめる。
(a) 損失の帯域(2–12 kHz)で損失の外の倍音 800 / 1000 / 1200 Hz の線・(b) 損失の外の帯域 12–20 kHz の 200 / 400 / 600 Hz の線・(c) 包絡を Hilbert ではなく 1/3 ms の RMS で取った 2–12 kHz の線。
いずれも ±3 Hz 対 ±40 Hz の近傍の比(dB)・先頭 8 s。学習の外の devff2(30 組)の D2 / D3(P43)/ 写し合成 CP / 入力そのもの。
    uv run python d469_buzz_indep.py --out ../results/conv_p0/d469.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
SR = 48000


def line(env: np.ndarray, fs: float, harm: tuple) -> float:
    f, P = ss.welch(env - env.mean(), fs=fs, nperseg=4096)
    pk, bg = 0.0, 0.0
    for h in harm:
        on = (f >= h - 3) & (f <= h + 3)
        nb = (f >= h - 40) & (f <= h + 40) & ~on
        pk += P[on].sum()
        bg += P[nb].mean() * on.sum()
    return float(10 * np.log10(pk / max(bg, 1e-30)))


def stats(x: np.ndarray) -> dict:
    out = {}
    hi_h, lo_h = (800.0, 1000.0, 1200.0), (200.0, 400.0, 600.0)
    va, vb, vc = [], [], []
    for lo, hi in ((2000, 4000), (4000, 8000), (8000, 12000)):
        y = ss.sosfiltfilt(ss.butter(6, [lo, hi], btype="band", fs=SR, output="sos"), x.astype(np.float64))
        env = np.abs(ss.hilbert(y))
        va.append(line(ss.resample_poly(env, 1, 12), SR / 12, hi_h))
        n = len(y) // 16
        rms = np.sqrt((y[: n * 16].reshape(n, 16) ** 2).mean(1))
        vc.append(line(rms, SR / 16, lo_h))
    y = ss.sosfiltfilt(ss.butter(6, [12000, 20000], btype="band", fs=SR, output="sos"), x.astype(np.float64))
    vb.append(line(ss.resample_poly(np.abs(ss.hilbert(y)), 1, 12), SR / 12, lo_h))
    out["a_harm_800_1200"] = float(np.mean(va))
    out["b_band_12_20k"] = float(np.mean(vb))
    out["c_rms_env"] = float(np.mean(vc))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    groups = {"SRC": sorted((H / "devff2/d56").glob("p*_SRC.wav")), "CP": sorted((H / "devff2/d444/CP").glob("p*_CP.wav")),
              "D2": sorted((H / "devff2/d444/D2").glob("p*_D2.wav")), "P43": sorted((H / "devff2/d444/P43").glob("p*_P43.wav")),
              "P43OP": sorted((H / "devff2/d444/P43OP").glob("p*_P43OP.wav"))}
    rep = {}
    for g, fs in groups.items():
        S = [stats(load48(f)[: 8 * SR]) for f in fs]
        rep[g] = {"n": len(S)} | {k: round(float(np.median([s[k] for s in S])), 3) for k in S[0]} | {f"{k}_p85": round(float(np.percentile([s[k] for s in S], 85)), 3) for k in S[0]}
        print(g, rep[g], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
