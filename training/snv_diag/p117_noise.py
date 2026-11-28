"""段 P117(学習なし): 描画器が乗せる雑音の在り処。元の音と描画器を通した音(写し・変換)を比べる。
(1) 無音の区間(元の音が −60 dBFS 未満)での出力の音量。(2) 声の区間で、帯域ごとの「倍音の間の成分 ÷ 倍音の成分」(dB・高いほど雑音が多い)。(3) 帯域ごとの音量の差。
    cd training && uv run --no-sync python snv_diag/p117_noise.py <元の音> <比べる音> [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

BANDS = [(80, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]


def analyze(p: str, f0: np.ndarray | None = None) -> dict:
    x, sr = librosa.load(p, sr=48000, mono=True)
    if f0 is None:
        f, vf, _ = librosa.pyin(librosa.resample(x, orig_sr=sr, target_sr=16000), fmin=70, fmax=700, sr=16000, frame_length=1024, hop_length=160, center=True)
        f0 = np.where(vf, np.nan_to_num(f), 0.0)
    S = np.abs(librosa.stft(x, n_fft=4096, hop_length=480, window="hann", center=True)) ** 2
    fr = np.fft.rfftfreq(4096, 1 / sr)
    n = min(S.shape[1], len(f0))
    S, f0 = S[:, :n], f0[:n]
    e = S.sum(0)
    lev = 10 * np.log10(e / (4096 * 1.5) + 1e-20)
    out = {"f0": f0, "lev": lev, "S": S, "fr": fr}
    hn = {b: [] for b in BANDS}
    for t in np.where(f0 > 0)[0]:
        k = fr / f0[t]
        near = np.abs(k - np.round(k)) < 0.2
        for b in BANDS:
            m = (fr >= b[0]) & (fr < b[1])
            h, i = S[m & near, t].sum(), S[m & ~near, t].sum()
            if h > 0 and i > 0:
                hn[b].append(10 * np.log10(i / h))
    out["inter"] = {b: float(np.median(v)) if v else float("nan") for b, v in hn.items()}
    out["band"] = {b: float(10 * np.log10(S[(fr >= b[0]) & (fr < b[1])][:, f0 > 0].mean() + 1e-20)) for b in BANDS}
    return out


def main() -> None:
    a = analyze(sys.argv[1])
    sil = a["lev"] < a["lev"].max() - 60
    print(Path(sys.argv[1]).name, "| 無音のフレームの率", round(float(sil.mean()), 2), "| 声の区間の 倍音の間 ÷ 倍音(dB)", {f"{b[0]}-{b[1]}": round(v, 1) for b, v in a["inter"].items()})
    for p in sys.argv[2:]:
        b = analyze(p, a["f0"])
        n = min(len(sil), len(b["lev"]))
        s = sil[:n]
        print(" →", Path(p).name, "| 元が無音の所の音量(元 / これ・最大からの dB)", round(float(np.median((a["lev"] - a["lev"].max())[:n][s])), 1) if s.any() else None, "/", round(float(np.median((b["lev"][:n] - a["lev"].max())[s])), 1) if s.any() else None,
              "| 倍音の間 ÷ 倍音", {f"{q[0]}-{q[1]}": round(v, 1) for q, v in b["inter"].items()}, "| 帯域の音量の差(これ − 元)", {f"{q[0]}-{q[1]}": round(b["band"][q] - a["band"][q], 1) for q in BANDS})


if __name__ == "__main__":
    main()
