"""D520(耳の所見「フェーザー・コーラス」の候補 = 倍音の振幅列の溝が動く): 有声のフレーム(WORLD の f0・10 ms)で倍音 k·f0(4 kHz まで)の振幅 A_k(dB)を取り、
凸凹 = |A_k − (A_{k−1} + A_{k+1}) / 2| の中央値(溝の深さ)と、動き = 50 ms 離れたフレームの凸凹の列(同じ倍音番号)の相関の 1 − r(溝の模様が入れ替わる速さ)を求める。
群 = SRC・REAL_cen・CP・OFF(遅い揺らぎなし)・OP15(候補)・RND15C・PROD。対の差(群 − SRC)も。出力 = results/conv_p0/d520.json。
    uv run python d520_notch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P

ROOT = Path(__file__).resolve().parent.parent
S = P.H / "devff5"
R = ROOT / "results/conv_p0/gate_ff_v3/render/devff5"
NF = 4096


def measure(x: np.ndarray, sr: int) -> dict:
    import pyworld as pw
    x16 = resample_poly(x.astype(np.float64), 16000, sr)
    f0, t = pw.dio(x16, 16000, f0_floor=90, f0_ceil=1100, frame_period=10.0)
    f0 = pw.stonemask(x16, f0, t, 16000)
    win = np.hanning(NF)
    fr = np.fft.rfftfreq(NF, 1 / sr)
    xp = np.pad(x.astype(np.float64), NF)
    dev = {}
    for i, f in enumerate(f0):
        if f <= 0 or 4000 / f < 5:
            continue
        c = int(t[i] * sr) + NF
        seg = xp[c - NF // 2:c + NF // 2] * win
        if np.sqrt(np.mean(seg ** 2)) < 1e-4:
            continue
        A = 20 * np.log10(np.abs(np.fft.rfft(seg)) + 1e-9)
        K = int(4000 / f)
        hw = lambda fq: max(0.03 * fq, 1.5 * sr / NF)
        a = np.array([A[(fr > k * f - hw(k * f)) & (fr < k * f + hw(k * f))].max() for k in range(1, K + 1)])
        dev[i] = a[1:-1] - 0.5 * (a[:-2] + a[2:])
    rough = [float(np.median(np.abs(v))) for v in dev.values()]
    mv = []
    for i, v in dev.items():
        j = i + 5
        if j in dev:
            n = min(len(v), len(dev[j]))
            if n >= 4 and np.std(v[:n]) > 0 and np.std(dev[j][:n]) > 0:
                mv.append(1 - float(np.corrcoef(v[:n], dev[j][:n])[0, 1]))
    return {"rough_db": float(np.median(rough)) if rough else np.nan, "move": float(np.median(mv)) if mv else np.nan}


def main() -> int:
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    groups = {"SRC": lambda k: S / "d56" / f"{k}_SRC.wav", "CP": lambda k: R / "CP" / f"{k}_CP.wav", "OFF": lambda k: R / "OFF" / f"{k}_OFF.wav",
              "OP15": lambda k: R / "OP15" / f"{k}_OP15.wav", "RND15C": lambda k: R / "RND15C" / f"{k}_RND15C.wav", "PROD": lambda k: S / "d114" / f"{k}_G03S10ZH.wav"}
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
        summ[g] = {m: [round(float(np.nanmedian([r[m] for r in rs])), 3), round(float(np.nanpercentile([r[m] for r in rs], 15)), 3), round(float(np.nanpercentile([r[m] for r in rs], 85)), 3)] for m in ("rough_db", "move")}
        print(g, len(rs), json.dumps(summ[g]), flush=True)
    for g in ("CP", "OFF", "OP15", "RND15C", "PROD"):
        paired[g] = {m: round(float(np.nanmedian([a[m] - b[m] for a, b in zip(rows[g], rows["SRC"])])), 3) for m in ("rough_db", "move")}
        print("paired", g, "- SRC", json.dumps(paired[g]), flush=True)
    (ROOT / "results/conv_p0/d520.json").write_text(json.dumps({"summary_median_p15_p85": summ, "paired_median_minus_SRC": paired}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
