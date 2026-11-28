"""段 P114(学習なし): かすれの物差し。声が鳴っているフレームの「倍音と雑音の比」(HNR・自己相関の山から)と、高い帯(4〜8 kHz)と低い帯(0.3〜3 kHz)の音量の比を、音のファイルごとに出す。
    cd training && uv run --no-sync python snv_diag/p114_hnr.py <wav ...>
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np


def measure(p: str) -> dict:
    x, sr = librosa.load(p, sr=16000, mono=True)
    f0, vf, _ = librosa.pyin(x, fmin=70, fmax=1200, sr=sr, frame_length=1024, hop_length=160, center=True)
    hn = []
    for t in np.where(vf & np.isfinite(f0))[0]:
        T0 = int(round(sr / f0[t]))
        c = t * 160
        a, b = c - 2 * T0, c + 2 * T0
        if a < 0 or b + T0 > len(x) or T0 < 8:
            continue
        s0, s1 = x[a:b], x[a + T0 : b + T0]
        r = float((s0 * s1).sum() / (np.sqrt((s0 * s0).sum() * (s1 * s1).sum()) + 1e-12))
        r = min(max(r, 1e-4), 0.9999)
        hn.append(10 * np.log10(r / (1 - r)))
    y, sr2 = librosa.load(p, sr=48000, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=480)) ** 2
    fr = np.fft.rfftfreq(2048, 1 / sr2)
    e = S.sum(0)
    act = e > np.percentile(e, 60)
    hi, lo = S[(fr >= 4000) & (fr < 8000)][:, act].sum(), S[(fr >= 300) & (fr < 3000)][:, act].sum()
    return {"hnr_med": round(float(np.median(hn)), 1) if hn else None, "hnr_p25": round(float(np.percentile(hn, 25)), 1) if hn else None, "voiced": round(float(np.mean(vf)), 2), "f0_med": int(np.nanmedian(f0[vf])) if vf.any() else 0, "hi_lo_db": round(float(10 * np.log10(hi / lo)), 1)}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(Path(p).name.ljust(28), measure(p), flush=True)
