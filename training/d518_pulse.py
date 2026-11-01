"""D518(耳の所見「フェーザー・コーラス」「F0 がしっかり出ず・かすれ」の共通の候補 = 倍音の位相のそろい(声門のパルス感)の欠落): 有声のフレームの LPC の残差の尖りを比べる。
16 kHz・25 ms の窓(10 ms ごと)・WORLD の有声のフレームで LPC(18 次)の残差を取り、尖度(kurtosis)と ピーク / RMS(dB)を求め、発話ごとの中央値を群で並べる。
位相のそろった自然な有声音 = 周期ごとに鋭いパルス(尖度が高い)・位相のぼけ(フェーザー様)= パルスが鈍る。
群 = SRC(入力 = 実女声)・REAL_cen(目標の別の録音)・CP(写し合成 rvoc3hi)・OFF / OP15(候補)・PROD(製品)。対の差(群 − SRC)も。出力 = results/conv_p0/d518.json。
    uv run python d518_pulse.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P

ROOT = Path(__file__).resolve().parent.parent
S = P.H / "devff5"
R = ROOT / "results/conv_p0/gate_ff_v3/render/devff5"


def lpc(x: np.ndarray, p: int) -> np.ndarray:
    r = np.correlate(x, x, "full")[len(x) - 1:len(x) + p]
    r[0] *= 1.0 + 1e-6
    a = np.zeros(p + 1)
    a[0], e = 1.0, r[0]
    for i in range(1, p + 1):
        k = -(r[i] + a[1:i] @ r[1:i][::-1]) / e
        a[1:i + 1] = a[1:i + 1] + k * np.r_[a[1:i][::-1], 1.0]
        e *= 1 - k * k
    return a


def measure(x: np.ndarray, sr: int) -> dict:
    import pyworld as pw
    x16 = resample_poly(x.astype(np.float64), 16000, sr)
    f0, t = pw.dio(x16, 16000, f0_floor=90, f0_ceil=1100, frame_period=10.0)
    f0 = pw.stonemask(x16, f0, t, 16000)
    xe = lfilter([1, -0.97], [1], x16)
    W = 400
    win = np.hanning(W)
    ks, prs = [], []
    for i, f in enumerate(f0):
        if f <= 0:
            continue
        c = int(t[i] * 16000)
        if c - W // 2 < 0 or c + W // 2 > len(xe):
            continue
        seg = xe[c - W // 2:c + W // 2]
        if np.sqrt(np.mean(seg ** 2)) < 1e-4:
            continue
        a = lpc(seg * win, 18)
        res = lfilter(a, [1], seg)[40:-40]
        ks.append(float(kurtosis(res)))
        prs.append(float(20 * np.log10(np.abs(res).max() / (np.sqrt(np.mean(res ** 2)) + 1e-12))))
    return {"kurt": float(np.median(ks)) if ks else np.nan, "peak_rms_db": float(np.median(prs)) if prs else np.nan, "n": len(ks)}


def main() -> int:
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    groups = {"SRC": lambda k: S / "d56" / f"{k}_SRC.wav", "CP": lambda k: R / "CP" / f"{k}_CP.wav", "OFF": lambda k: R / "OFF" / f"{k}_OFF.wav",
              "OP15": lambda k: R / "OP15" / f"{k}_OP15.wav", "PROD": lambda k: S / "d114" / f"{k}_G03S10ZH.wav"}
    rows: dict = {g: [] for g in list(groups) + ["REAL_cen"]}
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        for g, fn in groups.items():
            x, sr = sf.read(str(fn(k)), dtype="float32", always_2d=True)
            rows[g].append(measure(x.mean(1)[: 8 * sr], sr))
    for t in sorted({r["ti"] for r in pairs}):
        x, sr = sf.read(str(S / "d37" / f"cen_{t:03d}.wav"), dtype="float32", always_2d=True)
        x = x.mean(1)
        for o in range(0, max(1, len(x) - 8 * sr), 8 * sr):
            rows["REAL_cen"].append(measure(x[o:o + 8 * sr], sr))
    summ, paired = {}, {}
    for g, rs in rows.items():
        summ[g] = {m: [round(float(np.nanmedian([r[m] for r in rs])), 3), round(float(np.nanpercentile([r[m] for r in rs], 15)), 3), round(float(np.nanpercentile([r[m] for r in rs], 85)), 3)] for m in ("kurt", "peak_rms_db")}
        print(g, len(rs), json.dumps(summ[g]), flush=True)
    for g in ("CP", "OFF", "OP15", "PROD"):
        d = {m: np.array([a[m] - b[m] for a, b in zip(rows[g], rows["SRC"])]) for m in ("kurt", "peak_rms_db")}
        paired[g] = {m: [round(float(np.nanmedian(v)), 3), round(float(np.nanmean(v < 0)), 3)] for m, v in d.items()}
        print("paired", g, "- SRC [median, frac<0]", json.dumps(paired[g]), flush=True)
    (ROOT / "results/conv_p0/d518.json").write_text(json.dumps({"summary_median_p15_p85": summ, "paired_median_fracneg": paired}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
