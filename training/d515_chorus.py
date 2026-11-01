"""D515(オーナーの耳の所見 10-22「フェーザー・コーラスのよう・ピッチの不自然な振幅」の切り分け): devff5 の gate_ff_v3 の描画で、
(1) f0 の揺れ = 有声の区間の log2 f0(半音)から 0.3 s の移動平均を引いた残りの RMS(3 Hz 以上の速い揺れ)を 入力 SRC・条件の f0(Rust の書き出しの列 25・目標の声域へ移した軌跡)・各版の出力 で、
(2) 追従の誤差 = 出力の実測の f0 − 条件の f0(半音・中央値を引く)の RMS と速い成分、
(3) スペクトルの形の遅い掃引 = 有声のフレームの log-mel(水準を引いた形)の 0.25〜2 Hz の変調の強さ(フェーザーの型)と 2〜8 Hz(コーラスの型)、
を比べる。f0 は WORLD(dio + stonemask・16 kHz・5 ms)= 生成器・F1 と独立。実女声の基準 = SRC(操作者)と目標の別の録音 cen。出力 = results/conv_p0/d515.json。
    uv run python d515_chorus.py
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
FP = 5.0


def f0_world(x: np.ndarray, sr: int) -> np.ndarray:
    import pyworld as pw
    x16 = resample_poly(x.astype(np.float64), 16000, sr)
    f0, t = pw.dio(x16, 16000, f0_floor=90, f0_ceil=1100, frame_period=FP)
    return pw.stonemask(x16, f0, t, 16000)


def segments(v: np.ndarray, min_len: int) -> list:
    out, s = [], None
    for i, b in enumerate(np.r_[v, False]):
        if b and s is None:
            s = i
        elif not b and s is not None:
            if i - s >= min_len:
                out.append((s, i))
            s = None
    return out


def wobble(st: np.ndarray, v: np.ndarray) -> float:
    w = int(0.3 * 1000 / FP)
    res = []
    for a, b in segments(v, int(0.2 * 1000 / FP)):
        y = st[a:b]
        k = np.ones(min(w, len(y))) / min(w, len(y))
        tr = np.convolve(np.pad(y, (len(k) // 2, len(k) - 1 - len(k) // 2), mode="edge"), k, mode="valid")
        res.append(y - tr)
    return float(np.sqrt(np.mean(np.concatenate(res) ** 2))) if res else float("nan")


def shape_mod(x: np.ndarray, sr: int, f0: np.ndarray) -> tuple:
    import librosa
    m = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=sr, n_fft=2048, hop_length=int(sr * FP / 1000), n_mels=64, fmin=60, fmax=12000)
    lm = np.log(m + 1e-8)
    T = min(lm.shape[1], len(f0))
    lm, v = lm[:, :T], f0[:T] > 0
    lm = lm - lm.mean(0, keepdims=True)
    fs = 1000 / FP
    lo, hi = butter(2, [0.25, 2.0], btype="band", fs=fs, output="sos"), butter(2, [2.0, 8.0], btype="band", fs=fs, output="sos")
    a, b, tot = [], [], []
    for s0, s1 in segments(v, int(0.5 * fs)):
        seg = lm[:, s0:s1]
        if seg.shape[1] < 60:
            continue
        a.append(np.mean(sosfiltfilt(lo, seg, axis=1) ** 2))
        b.append(np.mean(sosfiltfilt(hi, seg, axis=1) ** 2))
        tot.append(np.mean((seg - seg.mean(1, keepdims=True)) ** 2))
    if not a:
        return float("nan"), float("nan")
    return float(np.mean(a) / np.mean(tot)), float(np.mean(b) / np.mean(tot))


def main() -> int:
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    groups = {"SRC": lambda k: S / "d56" / f"{k}_SRC.wav", "CP": lambda k: R / "CP" / f"{k}_CP.wav", "OFF": lambda k: R / "OFF" / f"{k}_OFF.wav",
              "OP15": lambda k: R / "OP15" / f"{k}_OP15.wav", "RND15C": lambda k: R / "RND15C" / f"{k}_RND15C.wav", "PROD": lambda k: S / "d114" / f"{k}_G03S10ZH.wav"}
    ed = P.H / "e2e" / "eval_devff5_pre"
    rep: dict = {g: {"wobble_st": [], "track_rms_st": [], "track_fast_st": [], "sweep_0p25_2": [], "mod_2_8": []} for g in list(groups) + ["COND", "REAL_cen"]}
    n0 = 12 * 200
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        c, _ = P.load_dump(ed, k)
        c = c[n0:n0 + 8 * 200]
        fc = np.where(c[:, 26] > 0.5, 200.0 * np.exp(c[:, 25]), 0.0)
        fc5 = np.repeat(fc, 1)
        vc = fc5 > 0
        stc = np.where(vc, 12 * np.log2(np.maximum(fc5, 1e-3)), 0.0)
        rep["COND"]["wobble_st"].append(wobble(stc, vc))
        for g, fn in groups.items():
            p = fn(k)
            if not p.exists():
                continue
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            x = x.mean(1)[: 8 * sr]
            f0 = f0_world(x, sr)
            v = f0 > 0
            st = np.where(v, 12 * np.log2(np.maximum(f0, 1e-3)), 0.0)
            rep[g]["wobble_st"].append(wobble(st, v))
            if g != "SRC":
                T = min(len(st), len(stc))
                both = v[:T] & vc[:T]
                if both.sum() > 50:
                    d = st[:T][both] - stc[:T][both]
                    d = d - np.median(d)
                    d = d[np.abs(d) < 6]
                    rep[g]["track_rms_st"].append(float(np.sqrt(np.mean(d ** 2))))
                    dd = np.where(both, st[:T] - stc[:T], 0.0)
                    rep[g]["track_fast_st"].append(wobble(dd - np.median(dd[both]), both))
            sw, md = shape_mod(x, sr, f0)
            rep[g]["sweep_0p25_2"].append(sw)
            rep[g]["mod_2_8"].append(md)
    for t in sorted({r["ti"] for r in pairs}):
        x, sr = sf.read(str(S / "d37" / f"cen_{t:03d}.wav"), dtype="float32", always_2d=True)
        x = x.mean(1)
        for o in range(0, max(1, len(x) - 8 * sr), 8 * sr):
            xx = x[o:o + 8 * sr]
            f0 = f0_world(xx, sr)
            v = f0 > 0
            st = np.where(v, 12 * np.log2(np.maximum(f0, 1e-3)), 0.0)
            rep["REAL_cen"]["wobble_st"].append(wobble(st, v))
            sw, md = shape_mod(xx, sr, f0)
            rep["REAL_cen"]["sweep_0p25_2"].append(sw)
            rep["REAL_cen"]["mod_2_8"].append(md)
    out = {}
    for g, d in rep.items():
        out[g] = {m: ([round(float(np.nanmedian(v)), 4), round(float(np.nanpercentile(v, 15)), 4), round(float(np.nanpercentile(v, 85)), 4), len(v)] if v else None) for m, v in d.items()}
        print(g, json.dumps(out[g]), flush=True)
    (ROOT / "results/conv_p0/d515.json").write_text(json.dumps({"summary_median_p15_p85_n": out, "raw": rep}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
