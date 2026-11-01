"""D516(オーナーの耳の所見 10-22「F0 がしっかり出ず、かすれ感」「フェーザー・コーラス」の切り分け): 出力の倍音の明瞭さを実女声と比べる。
各音声で WORLD(dio + stonemask・16 kHz)の f0 の有声のフレーム(10 ms ごと)について、48 kHz の 4096 点の窓(Hann)で
  倍音の山と谷の比(dB)= 倍音 k·f0 の ±3% の最大 − 倍音の間 (k+0.5)·f0 の ±3% の最小 を帯域(0〜1 / 1〜3 / 3〜6 kHz)で平均、
  H1 − H2(dB)= 基本波の強さ、
  WORLD の非周期性(D4C・帯域 0〜1 / 1〜3 / 3〜6 kHz の平均 dB)= かすれ、
を求め、発話ごとの中央値を群(SRC 入力 = 実女声・REAL_cen 目標の別の録音 = 実女声・CP 写し合成・OP15 候補・OFF 遅い揺らぎなし・PROD 製品)で並べる。出力 = results/conv_p0/d516.json。
    uv run python d516_harmonicity.py
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
BANDS = ((0, 1000), (1000, 3000), (3000, 6000))
NF = 4096


def measure(x: np.ndarray, sr: int) -> dict:
    import pyworld as pw
    x16 = resample_poly(x.astype(np.float64), 16000, sr)
    f0, t = pw.dio(x16, 16000, f0_floor=90, f0_ceil=1100, frame_period=10.0)
    f0 = pw.stonemask(x16, f0, t, 16000)
    ap = pw.d4c(x16, f0, t, 16000)
    fa = np.linspace(0, 8000, ap.shape[1])
    win = np.hanning(NF)
    fr = np.fft.rfftfreq(NF, 1 / sr)
    con = {b: [] for b in BANDS}
    h12, aps = [], {b: [] for b in BANDS}
    xp = np.pad(x.astype(np.float64), NF)
    for i, f in enumerate(f0):
        if f <= 0:
            continue
        c = int(t[i] * sr) + NF
        seg = xp[c - NF // 2:c + NF // 2] * win
        if np.sqrt(np.mean(seg ** 2)) < 1e-4:
            continue
        A = 20 * np.log10(np.abs(np.fft.rfft(seg)) + 1e-9)
        pk = lambda fq: A[(fr > fq * 0.97) & (fr < fq * 1.03)]
        for lo, hi in BANDS:
            vals = []
            k = max(1, int(np.ceil(lo / f)))
            while k * f < hi and (k + 0.5) * f < sr / 2:
                a, b = pk(k * f), pk((k + 0.5) * f)
                if len(a) and len(b):
                    vals.append(a.max() - b.min())
                k += 1
            if vals:
                con[(lo, hi)].append(float(np.mean(vals)))
        a1, a2 = pk(f), pk(2 * f)
        if len(a1) and len(a2):
            h12.append(float(a1.max() - a2.max()))
        for lo, hi in BANDS:
            m = (fa >= lo) & (fa < hi)
            aps[(lo, hi)].append(float(20 * np.log10(np.mean(ap[i, m]) + 1e-9)))
    out = {f"contrast_{lo // 1000}_{hi // 1000}k": float(np.median(v)) if v else np.nan for (lo, hi), v in con.items()}
    out["h1_h2"] = float(np.median(h12)) if h12 else np.nan
    out.update({f"aper_{lo // 1000}_{hi // 1000}k": float(np.median(v)) if v else np.nan for (lo, hi), v in aps.items()})
    out["voiced_frac"] = float((f0 > 0).mean())
    return out


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
    summ = {}
    for g, rs in rows.items():
        summ[g] = {m: [round(float(np.nanmedian([r[m] for r in rs])), 2), round(float(np.nanpercentile([r[m] for r in rs], 15)), 2), round(float(np.nanpercentile([r[m] for r in rs], 85)), 2)] for m in rs[0]}
        print(g, len(rs), json.dumps(summ[g]), flush=True)
    paired = {}
    for g in ("CP", "OFF", "OP15", "PROD"):
        paired[g] = {m: round(float(np.nanmedian([a[m] - b[m] for a, b in zip(rows[g], rows["SRC"])])), 2) for m in rows[g][0]}
        print("paired", g, "- SRC", json.dumps(paired[g]), flush=True)
    (ROOT / "results/conv_p0/d516.json").write_text(json.dumps({"summary_median_p15_p85": summ, "paired_median_minus_SRC": paired}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
