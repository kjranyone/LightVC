"""コーラス代理指標(耳ラベルで検証してから使う・hi_midはコーラス盲目と実測済み)。

P1 period_ncc: 帯域通過(既定2-8kHz)後の隣接ピッチ周期の正規化相互相関(±3%ラグ探索・エネルギー重み)。
P2 comb_db   : 帯域(既定2-6kHz)の調波ビン対 調波間ビンのパワー比[dB](有声フレーム・エネルギー重み)。
どちらも同一発話の参照(codec往復/GT decode)との差で使う。f0は各クリップ自身から推定。

    uv run python chorus_proxy.py --validate   # results/earbattery/chorus_proxy_validation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pyworld
import scipy.signal
import soundfile

SR = 48000
HOP_MS = 5.0
ROOT = Path(__file__).resolve().parent.parent


def load48(p: Path) -> np.ndarray:
    y, sr = soundfile.read(str(p), always_2d=False)
    y = y.mean(1) if y.ndim == 2 else y
    if sr != SR:
        y = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=SR)
    return y.astype(np.float64)


def f0_track(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = librosa.resample(y, orig_sr=SR, target_sr=16000)
    f0, t = pyworld.harvest(w, 16000, f0_floor=65, f0_ceil=1000, frame_period=HOP_MS)
    return pyworld.stonemask(w, f0, t, 16000), t


def bandpass(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = scipy.signal.butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return scipy.signal.sosfiltfilt(sos, y)


def period_ncc(y: np.ndarray, f0: np.ndarray, t: np.ndarray,
               band: tuple[float, float] = (2000.0, 8000.0)) -> float:
    x = bandpass(y, *band)
    num, den = 0.0, 0.0
    for fi, ti in zip(f0, t):
        if fi < 70:
            continue
        T0 = SR / fi
        W = int(T0)
        n = int(ti * SR) - W
        lags = np.arange(int(T0 * 0.97), int(np.ceil(T0 * 1.03)) + 1)
        if n < 0 or n + lags[-1] + W > len(x):
            continue
        a = x[n:n + W]
        na = float(np.sqrt((a * a).sum()))
        best, wgt = -1.0, 0.0
        for L in lags:
            b = x[n + L:n + L + W]
            nb = float(np.sqrt((b * b).sum()))
            if na * nb <= 1e-12:
                continue
            c = float((a * b).sum()) / (na * nb)
            if c > best:
                best, wgt = c, na * nb
        if wgt > 0:
            num += best * wgt
            den += wgt
    return num / den if den > 0 else float("nan")


def comb_db(y: np.ndarray, f0: np.ndarray, t: np.ndarray,
            band: tuple[float, float] = (2000.0, 6000.0), n_fft: int = 4096) -> float:
    hop = int(SR * HOP_MS / 1000)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")) ** 2
    fr = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    num, den = 0.0, 0.0
    for i, fi in enumerate(f0):
        if fi < 120 or i >= S.shape[1]:
            continue
        h = np.arange(np.ceil(band[0] / fi), np.floor(band[1] / fi) + 1)
        if len(h) < 2:
            continue
        col = S[:, i]
        dist_pk = np.abs(fr[:, None] - (h * fi)[None]).min(1)
        dist_vl = np.abs(fr[:, None] - ((h + 0.5) * fi)[None]).min(1)
        inb = (fr >= band[0]) & (fr <= band[1])
        pk = col[inb & (dist_pk <= 0.2 * fi)].sum()
        vl = col[inb & (dist_vl <= 0.2 * fi)].sum()
        if pk <= 0 or vl <= 0:
            continue
        e = pk + vl
        num += 10 * np.log10(pk / vl) * e
        den += e
    return num / den if den > 0 else float("nan")


def hi_mid(y: np.ndarray) -> float:
    w44 = librosa.resample(y, orig_sr=SR, target_sr=44100)
    S = np.abs(librosa.stft(w44, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=44100, n_fft=2048)
    return float(S[(f > 2000) & (f < 6000)].sum() / (S.sum() + 1e-12))


def measure(y: np.ndarray) -> dict:
    f0, t = f0_track(y)
    return {"period_ncc": round(period_ncc(y, f0, t), 4),
            "comb_db": round(comb_db(y, f0, t), 3),
            "hi_mid": round(hi_mid(y), 4),
            "voiced": round(float((f0 > 70).mean()), 3)}


VOICED_GROUPS = ("dlg_mid", "dlg_low", "moan_high", "ab_chorus")


def label_set() -> dict:
    eb = ROOT / "results/earbattery"
    g: dict = {}
    for cid in ("dlg_mid", "dlg_low", "moan_high", "whis_0", "whis_1", "whis_2"):
        g[cid] = {"neg": {"codec": eb / f"codec_ceiling/{cid}_codec.wav",
                          "source": eb / f"s16/{cid}_source.wav"},
                  "pos": {"s11_K8": eb / f"s11/{cid}_s11.wav",
                          "s11_K16": eb / f"s11_K16/{cid}_s11k16.wav"}}
    ab = eb / "ab_chorus"
    g["ab_chorus"] = {"neg": {"codec": ab / "chorus_gtdecode.wav", "source": ab / "chorus_source.wav"},
                      "pos": {"e1_K16": ab / "chorus_e1_K16.wav", "cv_K16": ab / "chorus_cv_K16.wav"}}
    return g


def validate() -> dict:
    """事前固定の合格基準: 有声4群(VOICED_GROUPS)の全群で、全陽性が全陰性より悪い
    (period_ncc・comb_dbは小さいほど悪い)。囁き3群は記録のみ(有声率が低く判定対象外)。"""
    out: dict = {"criterion": "voiced groups: every pos worse than every neg (strict)",
                 "labels": "system-level ear labels: codec/source=no chorus, s11 K8/K16 & e1/cv K16=chorus",
                 "groups": {}, "verdict": {}}
    for gid, g in label_set().items():
        r = {"neg": {k: measure(load48(p)) for k, p in g["neg"].items()},
             "pos": {k: measure(load48(p)) for k, p in g["pos"].items()}}
        for m in ("period_ncc", "comb_db", "hi_mid"):
            sgn = -1.0 if m == "hi_mid" else 1.0
            worst_neg = min(sgn * v[m] for v in r["neg"].values())
            best_pos = max(sgn * v[m] for v in r["pos"].values())
            r[f"sep_{m}"] = round(worst_neg - best_pos, 4)
        out["groups"][gid] = r
        print(gid, json.dumps(r, ensure_ascii=False), flush=True)
    for m in ("period_ncc", "comb_db", "hi_mid"):
        seps = [out["groups"][g][f"sep_{m}"] for g in VOICED_GROUPS]
        out["verdict"][m] = {"voiced_seps": seps,
                             "PASS": bool(all(s > 0 for s in seps))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        res = validate()
        p = ROOT / "results/earbattery/chorus_proxy_validation.json"
        p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(json.dumps(res["verdict"], ensure_ascii=False), "->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
