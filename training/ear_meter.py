"""耳が聞いている劣化軸を測る計器 v2（オフライン診断専用）。

- abrupt: バンド別フレームエネルギーの急変（|ΔdB| > 6 dB / 11.6ms）。
  「バリバリ」= 広帯域バースト / 帯域間不連続の直接検出。
- flux: 正規化スペクトラルフラックス（クリック性の鋭さ）。
- hnr_h: 合成に使った f0 列での調波/非調波ビン比（「かすれ」＝非周期成分）。
- band_med/p95: バンド別エネルギー分布（高域欠落＝かすれ別軸）。

    uv run python ear_meter.py a.wav b.wav ...   # --f0 cols.pt で hnr_h を追加
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF

SR = 44100
HOP = 512
NFFT = 2048
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 9000), (9000, 22050)]


def band_series(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    S = np.abs(librosa.stft(w, n_fft=NFFT, hop_length=HOP)) ** 2
    hz = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    tot = 10 * np.log10(S.sum(0) + 1e-12)
    active = tot > tot.max() - 35.0
    out = []
    for lo, hi in BANDS:
        m = (hz >= lo) & (hz < hi)
        out.append(10 * np.log10(S[m].sum(0) + 1e-12))
    return np.stack(out), active


def flux_series(w: np.ndarray) -> np.ndarray:
    S = np.abs(librosa.stft(w, n_fft=NFFT, hop_length=HOP))
    S = S / (S.sum(0, keepdims=True) + 1e-12)
    return np.maximum(np.diff(S, axis=-1), 0).sum(0) / 2


def hnr_from_f0(w: np.ndarray, f0: np.ndarray, hop_a: int) -> float:
    win = 1024
    fr = np.lib.stride_tricks.as_strided(
        w, shape=((len(w) - win) // HOP + 1, win),
        strides=(w.strides[0] * HOP, w.strides[0]),
    ) * np.hanning(win)
    idx = (np.arange(fr.shape[0]) * HOP / hop_a).astype(int)
    idx = np.clip(idx, 0, len(f0) - 1)
    fi = f0[idx]
    hnr = []
    for i, f in enumerate(fi):
        if f < 50:
            continue
        x = fr[i]
        if float(np.sqrt((x**2).mean())) < 1e-4:
            continue
        lag = SR / f
        lags = np.arange(int(lag * 0.9), int(lag * 1.1) + 1)
        lags = lags[(lags > 0) & (lags < win // 2)]
        if len(lags) == 0:
            continue
        ac = []
        for l in lags:
            a, b = x[:-l], x[l:]
            ac.append(np.dot(a, b) / (np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-30))
        r = float(np.max(ac))
        r = min(max(r, 1e-4), 0.999)
        hnr.append(10 * np.log10(r / (1 - r)))
    return float(np.median(hnr)) if hnr else float("nan")


def measure(path: str, f0: np.ndarray | None, hop_a: int) -> dict:
    w, _ = librosa.load(path, sr=SR, mono=True)
    B, active = band_series(w)
    Ba = B[:, active]
    jump = np.abs(np.diff(B, axis=-1)) > 6.0
    nbands_jump = jump.sum(0)
    mins = len(w) / SR / 60
    r = {
        "file": Path(path).name,
        "active_frac": round(float(active.mean()), 3),
        "band_med_db": [round(float(np.median(b)), 1) for b in Ba],
        "band_p95_db": [round(float(np.quantile(b, 0.95)), 1) for b in Ba],
        "abrupt_ge1_per_min": round(float((nbands_jump >= 1).sum() / mins), 1),
        "abrupt_ge3_per_min": round(float((nbands_jump >= 3).sum() / mins), 1),
        "abrupt_ge6_per_min": round(float((nbands_jump >= 6).sum() / mins), 1),
        "flux_p99": round(float(np.quantile(flux_series(w), 0.99)), 4),
        "flux_max": round(float(flux_series(w).max()), 4),
    }
    if f0 is not None:
        r["hnr_h_db"] = round(hnr_from_f0(w, f0, hop_a), 2)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--f0", default=None, help="cols pt (合成f0列) を指定した wav と同順で")
    a = ap.parse_args()
    cols = None
    if a.f0:
        d = torch.load(a.f0, map_location="cpu")
        cols = d["f0"].numpy() if torch.is_tensor(d["f0"]) else d["f0"]
    for p in a.wavs:
        f0 = cols if cols is not None else None
        print(json.dumps(measure(p, f0, SF.HOP_A), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
