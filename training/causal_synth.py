"""因果なパラメトリック合成(構音の描画の製品の合成・Rust の参照の実装)。フレーム t の出力ブロック(HOP 標本)はフレーム t−1 → t の間を直線補間した f0・包絡・非周期性で作る(先読み 0)。
有声: 位相の積算でパルスの時刻(小数)を決め、各パルスに最小位相の応答(振幅 = √(sp·(1 − ap²)·P)・P = 周期の標本数・小数の遅れは周波数の位相で・一定の 1 標本の遅れ)を重ね足す。
雑音: ブロックごとの白色雑音を最小位相の √(sp·ap²) で畳み込んで重ね足す。最小位相 = ケプストラムの折り返し。
"""
from __future__ import annotations

import numpy as np


def minphase(amp: np.ndarray, n: int) -> np.ndarray:
    c = np.fft.irfft(np.log(np.maximum(amp, 1e-12)), n)
    c[1:n // 2] *= 2.0
    c[n // 2 + 1:] = 0.0
    return np.exp(np.fft.rfft(c, n))


class CausalSynth:
    def __init__(self, sr: int = 48000, hop: int = 240, nfft: int = 2048, seed: int = 0):
        self.sr, self.hop, self.n = sr, hop, nfft
        self.buf = np.zeros(nfft + 2 * hop)
        self.phase = 0.0
        self.prev: tuple | None = None
        self.rng = np.random.default_rng(seed)
        self.k = np.arange(nfft // 2 + 1)

    def block(self, f0: float, sp: np.ndarray, ap: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = (f0, sp, ap)
        f0p, spp, app = self.prev
        h, n = self.hop, self.n
        for i in range(h):
            w = (i + 1) / h
            if f0 <= 0 and f0p <= 0:
                continue
            f = f0 if f0p <= 0 else f0p if f0 <= 0 else (1 - w) * f0p + w * f0
            if (f0 <= 0 and w > 0.5) or (f0p <= 0 and w <= 0.5):
                continue
            step = f / self.sr
            self.phase += step
            if self.phase >= 1.0:
                self.phase -= 1.0
                d = 1.0 - self.phase / step
                s_ = (1 - w) * spp + w * sp
                a_ = np.clip((1 - w) * app + w * ap, 0.0, 1.0)
                amp = np.sqrt(np.maximum(s_ * (1.0 - a_ ** 2), 0.0) * (self.sr / f))
                H = minphase(amp, n) * np.exp(-2j * np.pi * self.k * d / n)
                ir = np.fft.irfft(H, n)
                self.buf[i:i + n] += ir
        s_m, a_m = 0.5 * (spp + sp), np.clip(0.5 * (app + ap), 0.0, 1.0)
        g = minphase(np.sqrt(np.maximum(s_m * a_m ** 2, 0.0)), n)
        z = np.fft.irfft(np.fft.rfft(self.rng.standard_normal(h), 2 * n) * np.fft.rfft(np.fft.irfft(g, n), 2 * n), 2 * n)[: h + n]
        self.buf[: h + n] += z
        out = self.buf[:h].copy()
        self.buf[:-h] = self.buf[h:]
        self.buf[-h:] = 0.0
        self.prev = (f0, sp, ap)
        return out


def synth_causal(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray, seed: int = 0) -> np.ndarray:
    cs = CausalSynth(seed=seed)
    y = np.concatenate([cs.block(float(f0[t]), sp[t], ap[t]) for t in range(len(f0))])
    return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)
