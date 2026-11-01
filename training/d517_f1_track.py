"""D517(耳の所見「F0 がしっかり出ず・かすれ・ピッチの不自然な振幅」の切り分け・D515 の続き): 条件の f0(Rust F1 = 因果な推定器・目標の声域へ一定のずらし)の速い揺れは 2.10 半音で
実音声(WORLD の SRC 1.17)の約 2 倍。F1 の f0 を入力 SRC の WORLD の f0 と同じ時刻で比べる(ずらしの定数 = 発話の中央値・時刻の遅れ 0〜8 フレームの最良)。
指標(両方が有声のフレーム): |差| > 0.5 / 1 / 3 半音・オクターブ誤り(|差 ± 12| < 1.5)の割合・差の速い成分(0.3 s の移動平均を引いた RMS)・有声の判定の不一致(F1 有声で WORLD 無声 / 逆)。
出力の WORLD の f0 と F1 の差も同じく(出力が条件に従っているか)。耳の 3 試行(t01〜t03)の値を別に並べる。出力 = results/conv_p0/d517.json。
    uv run python d517_f1_track.py
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
from d515_chorus import wobble

ROOT = Path(__file__).resolve().parent.parent
S = P.H / "devff5"
R = ROOT / "results/conv_p0/gate_ff_v3/render/devff5"


def f0w(p: Path) -> np.ndarray:
    import pyworld as pw
    x, sr = sf.read(str(p), dtype="float64", always_2d=True)
    x16 = resample_poly(x.mean(1)[: 8 * sr], 16000, sr)
    f0, t = pw.dio(x16, 16000, f0_floor=90, f0_ceil=1100, frame_period=5.0)
    return pw.stonemask(x16, f0, t, 16000)


def compare(stc: np.ndarray, vc: np.ndarray, f0: np.ndarray) -> dict:
    v = f0 > 0
    st = np.where(v, 12 * np.log2(np.maximum(f0, 1e-3)), 0.0)
    best = None
    for lag in range(0, 9):
        T = min(len(stc) - lag, len(st))
        both = vc[lag:lag + T] & v[:T]
        if both.sum() < 50:
            continue
        d = stc[lag:lag + T][both] - st[:T][both]
        sc = np.median(np.abs(d - np.median(d)))
        if best is None or sc < best[0]:
            best = (sc, lag)
    if best is None:
        return {}
    lag = best[1]
    T = min(len(stc) - lag, len(st))
    a, b = stc[lag:lag + T], st[:T]
    va, vb = vc[lag:lag + T], v[:T]
    both = va & vb
    d = a[both] - b[both]
    d = d - np.median(d)
    dd = np.where(both, a - b - np.median(a[both] - b[both]), 0.0)
    return {"lag_fr": lag, "gt0p5": float(np.mean(np.abs(d) > 0.5)), "gt1": float(np.mean(np.abs(d) > 1)), "gt3": float(np.mean(np.abs(d) > 3)),
            "octave": float(np.mean((np.abs(d - 12) < 1.5) | (np.abs(d + 12) < 1.5))), "fast_rms": wobble(dd, both),
            "v_only_cond": float(np.mean(va & ~vb)), "v_only_world": float(np.mean(~va & vb))}


def main() -> int:
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    ed = P.H / "e2e" / "eval_devff5_pre"
    key = json.loads((ROOT / "results/earbattery/ff_v3_ear/_key_聴取後に開く.json").read_text())
    ear = {v["pair"]: t for t, v in key.items()}
    rows = {"COND_vs_SRC": [], "OUT_vs_COND": [], "CP_vs_COND": []}
    ear_rows = {}
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        c, _ = P.load_dump(ed, k)
        c = c[12 * 200:12 * 200 + 8 * 200]
        vc = c[:, 26] > 0.5
        stc = np.where(vc, 12 * np.log2(200.0 * np.exp(c[:, 25])), 0.0)
        m1 = compare(stc, vc, f0w(S / "d56" / f"{k}_SRC.wav"))
        m2 = compare(stc, vc, f0w(R / "OP15" / f"{k}_OP15.wav"))
        m3 = compare(stc, vc, f0w(R / "CP" / f"{k}_CP.wav"))
        for nm, m in (("COND_vs_SRC", m1), ("OUT_vs_COND", m2), ("CP_vs_COND", m3)):
            if m:
                rows[nm].append(m)
        if k in ear:
            ear_rows[ear[k]] = {"pair": k, "COND_vs_SRC": m1, "OUT_vs_COND": m2}
    summ = {nm: {m: [round(float(np.median([x[m] for x in rs])), 4), round(float(np.percentile([x[m] for x in rs], 85)), 4)] for m in rs[0]} for nm, rs in rows.items()}
    for nm, s in summ.items():
        print(nm, len(rows[nm]), json.dumps(s), flush=True)
    for t in sorted(ear_rows):
        e = ear_rows[t]
        print("ear", t, e["pair"], "COND_vs_SRC", {m: round(v, 3) for m, v in e["COND_vs_SRC"].items()}, flush=True)
    (ROOT / "results/conv_p0/d517.json").write_text(json.dumps({"summary_median_p85": summ, "ear": ear_rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
