"""D542(耳の所見「ジャギーのような感覚」(10-22・t02 = p0063_t023)の物差しの候補 = パラメータの階段 = ジッパー雑音の型): 発声中(発話の最大から −35 dB 以内)のフレームで、
log-mel 40 帯(50 Hz〜12 kHz・窓 20 ms・2.5 ms 刻み)の各帯の時系列から水準(全帯の平均)を引いた「形」の変調を、低(<1 kHz)・中(1〜4 kHz)・高(4〜12 kHz)の帯の群ごとに
速い変調 fast = 15〜50 Hz の変調のエネルギー / 2〜8 Hz の変調のエネルギー(dB)と、水準そのものの速い変調 lvl_fast(同じ比)で出す。発声中の区間(0.3 s 以上)ごとに FFT・発話でまとめる。
群 = NAME=<dir>:<接尾辞>・REAL_cen(目標の別録音 8 s 区切り)。発話ごとの値・耳の 3 試行の位置・群の対の差(B − A)。
    uv run python d542_fastmod.py --set devff5 --groups SRC=<d56>:SRC CP=<dir>:CP CAND=<dir>:OP15 --cen --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P

SR = 48000
HOPM = 120
WIN = 960
FPS = SR / HOPM
FB = None
EAR = {"t01": "p0076_t017", "t02": "p0063_t023", "t03": "p0056_t001"}


def measure(x: np.ndarray, sr: int) -> dict:
    global FB
    import librosa
    import scipy.signal as ss
    if sr != SR:
        from scipy.signal import resample_poly
        x = resample_poly(x, SR, sr)
    x = x.astype(np.float64)
    if FB is None:
        FB = librosa.filters.mel(sr=SR, n_fft=WIN, n_mels=40, fmin=50, fmax=12000)
    _, _, Z = ss.stft(x, SR, window="hann", nperseg=WIN, noverlap=WIN - HOPM, boundary=None)
    lm = 10 * np.log10(FB @ (np.abs(Z) ** 2) + 1e-10)
    lv = lm.mean(0)
    act = lv > lv.max() - 35
    shp = lm - lv[None]
    cfm = librosa.mel_frequencies(n_mels=42, fmin=50, fmax=12000)[1:-1]
    grp = {"low": cfm < 1000, "mid": (cfm >= 1000) & (cfm < 4000), "high": cfm >= 4000}
    segs = []
    s = None
    for i, b in enumerate(np.r_[act, False]):
        if b and s is None:
            s = i
        elif not b and s is not None:
            if (i - s) / FPS >= 0.3:
                segs.append((s, i))
            s = None
    out = {}
    acc = {g: [0.0, 0.0] for g in list(grp) + ["lvl"]}
    for a0, a1 in segs:
        n = a1 - a0
        fq = np.fft.rfftfreq(n, 1 / FPS)
        hw = np.hanning(n)[None]
        fast, slow = (fq >= 15) & (fq <= 50), (fq >= 2) & (fq <= 8)
        for g, m in grp.items():
            y = shp[m, a0:a1]
            Y = np.abs(np.fft.rfft((y - y.mean(1, keepdims=True)) * hw, axis=1)) ** 2
            acc[g][0] += Y[:, fast].sum()
            acc[g][1] += Y[:, slow].sum()
        y = lv[a0:a1]
        Y = np.abs(np.fft.rfft((y - y.mean()) * np.hanning(n))) ** 2
        acc["lvl"][0] += Y[fast].sum()
        acc["lvl"][1] += Y[slow].sum()
    for g, (f_, s_) in acc.items():
        out[g] = float(10 * np.log10((f_ + 1e-20) / (s_ + 1e-20))) if s_ > 0 else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--cen", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = P.H / a.set
    G = {g.split("=")[0]: (Path(g.split("=")[1].split(":")[0]), g.split(":")[-1]) for g in a.groups}
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    rows: dict = {g: {} for g in G}
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        for g, (d, sfx) in G.items():
            p = d / f"{k}_{sfx}.wav"
            if p.exists():
                x, sr = sf.read(str(p), dtype="float32", always_2d=True)
                rows[g][k] = measure(x.mean(1)[: 8 * sr], sr)
    if a.cen:
        rows["REAL_cen"] = {}
        for t_ in sorted({r["ti"] for r in pairs}):
            p = S / "d37" / f"cen_{t_:03d}.wav"
            if not p.exists():
                continue
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            x = x.mean(1)
            for o in range(0, max(1, len(x) - 8 * sr), 8 * sr):
                rows["REAL_cen"][f"{t_}_{o}"] = measure(x[o:o + 8 * sr], sr)
    ms = ("low", "mid", "high", "lvl")
    rep = {"summary_median_p15_p85": {g: {m: [round(float(np.nanpercentile([v[m] for v in d.values()], q)), 2) for q in (50, 15, 85)] for m in ms} for g, d in rows.items()}}
    ref = list(G)[0]
    rep["paired_minus_" + ref] = {g: {m: round(float(np.nanmedian([rows[g][k][m] - rows[ref][k][m] for k in rows[g] if k in rows[ref]])), 2) for m in ms} for g in G if g != ref}
    rep["ear"] = {t: {g: {m: round(rows[g][k][m], 2) for m in ms} for g in G if k in rows[g]} for t, k in EAR.items()}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps({**rep, "per_utt": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
