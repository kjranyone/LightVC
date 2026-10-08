"""倍音間の谷の深さ(500–3000Hz・有声フレーム・log|STFT| の p90 − p10・nfft 4096)。"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss


def contrast(y: np.ndarray, f0: np.ndarray) -> float:
    f, _, Z = ss.stft(y, 48000, nperseg=4096, noverlap=4096 - 480)
    L = np.log(np.abs(Z) + 1e-6)
    band = (f >= 500) & (f <= 3000)
    T = min(L.shape[1], len(f0))
    v = f0[:T] > 0
    Lb = L[band][:, :T][:, v]
    return float(np.mean(np.percentile(Lb, 90, axis=0) - np.percentile(Lb, 10, axis=0)))
