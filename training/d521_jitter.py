"""D521(耳の所見「F0 がしっかり出ず・かすれ・ピッチの不自然な振幅」の候補 = 周期ごとの f0 の乱れ = ジッター):
(1) 条件の f0(F1)の隣り合う 5 ms のフレームの差 |Δ半音| の中央値・90% 点(両方有声)を、入力 SRC の WORLD の f0 の同じ量と並べる(WORLD は滑らか寄り = 参考)、
(2) 波形の周期ごとの局所ジッター = 有声の区間(WORLD)で、f0 の 0.6〜1.6 倍の帯域に通した信号の上向きのゼロ交差の間隔 T_i から mean|T_i − T_{i−1}| / mean T(%)を
SRC・REAL_cen・CP・OP15・PROD で求める(48 kHz・ゼロ交差は線形補間)。対の差(群 − SRC)も。出力 = results/conv_p0/d521.json。
    uv run python d521_jitter.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P
from d517_f1_track import f0w

ROOT = Path(__file__).resolve().parent.parent
S = P.H / "devff5"
R = ROOT / "results/conv_p0/gate_ff_v3/render/devff5"


def period_jitter(x: np.ndarray, sr: int) -> float:
    import pyworld as pw
    x16 = resample_poly(x.astype(np.float64), 16000, sr)
    f0, t = pw.dio(x16, 16000, f0_floor=90, f0_ceil=1100, frame_period=5.0)
    f0 = pw.stonemask(x16, f0, t, 16000)
    js = []
    v = f0 > 0
    s = None
    for i, b in enumerate(np.r_[v, False]):
        if b and s is None:
            s = i
        elif not b and s is not None:
            if i - s >= 40:
                fm = float(np.median(f0[s:i]))
                a0, a1 = int(t[s] * sr), int(t[i - 1] * sr)
                seg = x[a0:a1].astype(np.float64)
                if len(seg) > sr // 10 and fm * 1.6 < sr / 2:
                    y = sosfiltfilt(butter(2, [fm * 0.6, fm * 1.6], btype="band", fs=sr, output="sos"), seg)
                    z = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0))
                    if len(z) > 5:
                        zc = z + (-y[z] / (y[z + 1] - y[z] + 1e-12))
                        T = np.diff(zc)
                        T = T[(T > sr / fm * 0.7) & (T < sr / fm * 1.4)]
                        if len(T) > 4:
                            js.append(float(np.mean(np.abs(np.diff(T))) / np.mean(T) * 100))
            s = None
    return float(np.median(js)) if js else float("nan")


def main() -> int:
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    ed = P.H / "e2e" / "eval_devff5_pre"
    groups = {"SRC": lambda k: S / "d56" / f"{k}_SRC.wav", "CP": lambda k: R / "CP" / f"{k}_CP.wav", "OP15": lambda k: R / "OP15" / f"{k}_OP15.wav",
              "PROD": lambda k: S / "d114" / f"{k}_G03S10ZH.wav"}
    jit: dict = {g: [] for g in list(groups) + ["REAL_cen"]}
    dcond, dworld = [], []
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        c, _ = P.load_dump(ed, k)
        c = c[12 * 200:12 * 200 + 8 * 200]
        vc = c[:, 26] > 0.5
        stc = 12 * np.log2(200.0 * np.exp(c[:, 25]))
        fw = f0w(S / "d56" / f"{k}_SRC.wav")
        T = min(len(fw), len(stc))
        vw = fw[:T] > 0
        both = vc[:T] & vw
        b2 = both[1:] & both[:-1]
        dcond.extend(np.abs(np.diff(stc[:T]))[b2].tolist())
        stw = 12 * np.log2(np.maximum(fw[:T], 1e-3))
        dworld.extend(np.abs(np.diff(stw))[b2].tolist())
        for g, fn in groups.items():
            x, sr = sf.read(str(fn(k)), dtype="float32", always_2d=True)
            jit[g].append(period_jitter(x.mean(1)[: 8 * sr], sr))
    for t_ in sorted({r["ti"] for r in pairs}):
        x, sr = sf.read(str(S / "d37" / f"cen_{t_:03d}.wav"), dtype="float32", always_2d=True)
        x = x.mean(1)
        for o in range(0, max(1, len(x) - 8 * sr), 8 * sr):
            jit["REAL_cen"].append(period_jitter(x[o:o + 8 * sr], sr))
    rep = {"frame_delta_st": {"COND_F1": [round(float(np.median(dcond)), 4), round(float(np.percentile(dcond, 90)), 4)],
                              "SRC_WORLD": [round(float(np.median(dworld)), 4), round(float(np.percentile(dworld, 90)), 4)]},
           "period_jitter_pct": {g: [round(float(np.nanmedian(v)), 3), round(float(np.nanpercentile(v, 15)), 3), round(float(np.nanpercentile(v, 85)), 3)] for g, v in jit.items()},
           "paired_minus_SRC": {g: [round(float(np.nanmedian(np.array(jit[g]) - np.array(jit["SRC"]))), 3), round(float(np.nanmean(np.array(jit[g]) > np.array(jit["SRC"]))), 3)] for g in ("CP", "OP15", "PROD")}}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    (ROOT / "results/conv_p0/d521.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
