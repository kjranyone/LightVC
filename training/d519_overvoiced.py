"""D519(D517 の続き): F1(条件の有声)だけが有声と言い WORLD が入力を無声とするフレーム(以下 F1 のみ・中央値 12%)で、出力に入力に無い周期性(息に乗ったピッチ)が出ているか。
フレームの種類 = 両方有声 / F1 のみ / 両方無声。各フレームで正規化自己相関の最大(条件の f0 の周期 ±10% の範囲・25 ms 窓・16 kHz・800 Hz 以下に帯域制限した信号)= 周期性 を
入力 SRC・出力 OP15・写し合成 CP で求め、種類ごとの中央値と、F1 のみのフレームの 出力 − 入力 を発話ごとに並べる。F1 のみのフレームの位置(有声の区間の端からの距離)と音量(入力の dB・発話の最大から)も。
出力 = results/conv_p0/d519.json。
    uv run python d519_overvoiced.py
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

ROOT = Path(__file__).resolve().parent.parent
S = P.H / "devff5"
R = ROOT / "results/conv_p0/gate_ff_v3/render/devff5"
SR = 16000
HOPS = 80


def load16(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float64", always_2d=True)
    return resample_poly(x.mean(1)[: 8 * sr], SR, sr)


def periodicity(x: np.ndarray, f0: np.ndarray) -> np.ndarray:
    lp = sosfiltfilt(butter(4, 800, fs=SR, output="sos"), x)
    W = 400
    out = np.full(len(f0), np.nan)
    for t, f in enumerate(f0):
        if f <= 0:
            continue
        c = t * HOPS
        if c - W // 2 < 0 or c + W // 2 + 400 > len(lp):
            continue
        a = lp[c - W // 2:c + W // 2]
        if np.sqrt(np.mean(a ** 2)) < 1e-5:
            continue
        T0 = SR / f
        best = 0.0
        for L in range(max(int(T0 * 0.9), 2), int(T0 * 1.1) + 1):
            b = lp[c - W // 2 + L:c + W // 2 + L]
            r = float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-12))
            best = max(best, r)
        out[t] = best
    return out


def main() -> int:
    import pyworld as pw
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    ed = P.H / "e2e" / "eval_devff5_pre"
    agg = {"both": {"SRC": [], "OP15": [], "CP": []}, "f1only": {"SRC": [], "OP15": [], "CP": []}}
    per, edge, lev = [], [], []
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        c, _ = P.load_dump(ed, k)
        c = c[12 * 200:12 * 200 + 8 * 200]
        fc = np.where(c[:, 26] > 0.5, 200.0 * np.exp(c[:, 25]), 0.0)
        xs = load16(S / "d56" / f"{k}_SRC.wav")
        f0w, tw = pw.dio(xs, SR, f0_floor=90, f0_ceil=1100, frame_period=5.0)
        f0w = pw.stonemask(xs, f0w, tw, SR)
        T = min(len(fc), len(f0w))
        fc, f0w = fc[:T], f0w[:T]
        vc, vw = fc > 0, f0w > 0
        # 条件の f0 は目標の声域へずれているので、入力の周期は WORLD の f0(両方有声)・F1 のみは条件の f0 を声域の差で戻した値で測る
        both = vc & vw
        shift = np.median(np.log2(fc[both] / f0w[both])) if both.sum() > 20 else 0.0
        f_in = np.where(vw, f0w, np.where(vc, fc / 2 ** shift, 0.0))
        p_in = periodicity(xs, f_in)
        p_out = periodicity(load16(R / "OP15" / f"{k}_OP15.wav"), fc)
        p_cp = periodicity(load16(R / "CP" / f"{k}_CP.wav"), fc)
        f1o = vc & ~vw
        for nm, m in (("both", both), ("f1only", f1o)):
            for g, pv in (("SRC", p_in), ("OP15", p_out), ("CP", p_cp)):
                v = pv[:T][m]
                agg[nm][g].extend(v[~np.isnan(v)].tolist())
        if f1o.sum() > 5:
            d = p_out[:T][f1o] - p_in[:T][f1o]
            per.append(float(np.nanmedian(d)))
            idx = np.flatnonzero(f1o)
            vidx = np.flatnonzero(vw)
            if len(vidx):
                edge.extend((np.min(np.abs(idx[:, None] - vidx[None, :]), 1) * 5).tolist())
            e = np.array([np.sqrt(np.mean(xs[i * HOPS:(i + 1) * HOPS] ** 2) + 1e-12) for i in range(T)])
            lev.extend((20 * np.log10(e[f1o] / e.max())).tolist())
    rep = {nm: {g: [round(float(np.median(v)), 3), len(v)] for g, v in d.items()} for nm, d in agg.items()}
    rep["f1only_out_minus_in_per_utt"] = [round(float(np.median(per)), 3), round(float(np.mean(np.array(per) > 0)), 3), len(per)]
    rep["f1only_dist_to_world_voiced_ms"] = [round(float(np.percentile(edge, q)), 1) for q in (25, 50, 75, 90)]
    rep["f1only_level_db_rel_max"] = [round(float(np.percentile(lev, q)), 1) for q in (25, 50, 75)]
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    (ROOT / "results/conv_p0/d519.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
