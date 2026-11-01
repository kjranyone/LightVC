"""D522(耳の所見「F0 がしっかり出ず・かすれ」の候補 = 周期性の定数化・D47〜D50 の前例「f0 見失い・ガビガビ = 周期性の定数化が主因」):
候補(P43OP)の周期性の条件は目標の定数 × 有声 = 入力の息っぽさの時間の動き(はっきりした声 ⇔ かすれた声)を受け取らない。
入力 SRC と出力の WORLD D4C の非周期性(1〜3 kHz・3〜6 kHz の帯の平均 dB)の時系列を、両方が有声のフレーム(10 ms)で比べる:
入力との相関(息っぽさの動きを追うか)・時間の標準偏差(動きの大きさ)・入力が息っぽい(上位 25%)フレームと はっきりした(下位 25%)フレームの差。群 = CP(実の周期性)・OFF・OP15・PROD。
出力 = results/conv_p0/d522.json。
    uv run python d522_aper_track.py
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
BANDS = ((1000, 3000), (3000, 6000))


def ap_series(p: Path) -> tuple:
    import pyworld as pw
    x, sr = sf.read(str(p), dtype="float64", always_2d=True)
    x16 = resample_poly(x.mean(1)[: 8 * sr], 16000, sr)
    f0, t = pw.dio(x16, 16000, f0_floor=90, f0_ceil=1100, frame_period=10.0)
    f0 = pw.stonemask(x16, f0, t, 16000)
    ap = pw.d4c(x16, f0, t, 16000)
    fa = np.linspace(0, 8000, ap.shape[1])
    out = np.stack([20 * np.log10(ap[:, (fa >= lo) & (fa < hi)].mean(1) + 1e-9) for lo, hi in BANDS], 1)
    return out, f0 > 0


def main() -> int:
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    groups = {"CP": "CP", "OFF": "OFF", "OP15": "OP15"}
    res: dict = {g: {"corr": [[], []], "std_ratio": [[], []], "breathy_minus_clear_out": [[], []]} for g in list(groups) + ["PROD"]}
    src_bc = [[], []]
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        a_in, v_in = ap_series(S / "d56" / f"{k}_SRC.wav")
        outs = {g: ap_series(R / d / f"{k}_{d}.wav") for g, d in groups.items()}
        outs["PROD"] = ap_series(S / "d114" / f"{k}_G03S10ZH.wav")
        for g, (a_o, v_o) in outs.items():
            T = min(len(a_in), len(a_o))
            m = v_in[:T] & v_o[:T]
            if m.sum() < 40:
                continue
            for b in range(len(BANDS)):
                xi, xo = a_in[:T, b][m], a_o[:T, b][m]
                res[g]["corr"][b].append(float(np.corrcoef(xi, xo)[0, 1]))
                res[g]["std_ratio"][b].append(float(np.std(xo) / (np.std(xi) + 1e-9)))
                q1, q3 = np.percentile(xi, [25, 75])
                hi_, lo_ = xi >= q3, xi <= q1
                res[g]["breathy_minus_clear_out"][b].append(float(xo[hi_].mean() - xo[lo_].mean()))
                if g == "CP":
                    src_bc[b].append(float(xi[hi_].mean() - xi[lo_].mean()))
    rep = {}
    for g, d in res.items():
        rep[g] = {f"{m}_{BANDS[b][0] // 1000}_{BANDS[b][1] // 1000}k": round(float(np.nanmedian(d[m][b])), 3) for m in d for b in range(len(BANDS))}
        print(g, json.dumps(rep[g]), flush=True)
    rep["SRC_breathy_minus_clear"] = {f"{BANDS[b][0] // 1000}_{BANDS[b][1] // 1000}k": round(float(np.median(src_bc[b])), 3) for b in range(len(BANDS))}
    print("SRC breathy − clear (dB)", rep["SRC_breathy_minus_clear"], flush=True)
    (ROOT / "results/conv_p0/d522.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
