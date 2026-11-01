"""D379: 実男声の組 devreal(外郎売)の録音の雑音の床(話し声比 −37 dB・TTS −67・本物の女声 −70)が変換の局所の偽物らしさと明瞭さを落とすかの診断。
入力(評価の発話 d56/*_SRC.wav と操作者の参照 d56/_work/ref_uiro.wav)にスペクトル減算(雑音のスペクトル = 録音全体の静かな 10% のフレームの振幅の中央値・過剰減算 1.5・下限 0.1)をかけた組 devreal_dn を作り、
操作者の登録を作り直す(目標は同じ)。雑音の推定は録音全体の統計 = 診断専用(製品にするなら因果な走行推定)。
    uv run python d379_denoise.py --s <dev の親>
    uv run python d274_evaldump.py --s <dev の親> --set devreal_dn --out <dev の親>/e2e
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d273_data import BIN

ROOT = Path(__file__).resolve().parent.parent
NF, HOP = 2048, 512


def stft(x: np.ndarray) -> np.ndarray:
    w = np.hanning(NF)
    n = 1 + (len(x) - NF) // HOP
    return np.stack([np.fft.rfft(x[i * HOP:i * HOP + NF] * w) for i in range(n)])


def istft(X: np.ndarray, length: int) -> np.ndarray:
    w = np.hanning(NF)
    y, s = np.zeros(length + NF), np.zeros(length + NF)
    for i, F in enumerate(X):
        y[i * HOP:i * HOP + NF] += np.fft.irfft(F, NF) * w
        s[i * HOP:i * HOP + NF] += w ** 2
    return (y / np.maximum(s, 1e-8))[:length]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--beta", type=float, default=0.1)
    a = ap.parse_args()
    S = Path(a.s)
    src, dst = S / "devreal", S / "devreal_dn"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    full, _ = sf.read(src / "d56/_work/uiro_full.wav", dtype="float64")
    X = stft(full)
    mag = np.abs(X)
    en = (mag ** 2).sum(1)
    N = np.median(mag[en <= np.percentile(en, 10)], 0)

    def dn(x: np.ndarray) -> np.ndarray:
        Y = stft(np.r_[x, np.zeros(NF)])
        m = np.abs(Y)
        g = np.maximum(1 - a.alpha * N[None] / np.maximum(m, 1e-12), a.beta)
        return istft(Y * g, len(x)).astype(np.float32)

    files = sorted((dst / "d56").glob("p00*_SRC.wav")) + [dst / "d56/_work/ref_uiro.wav"]
    for f in files:
        x, sr = sf.read(f, dtype="float64")
        sf.write(f, dn(x), sr)
    spd = dst / "d85_work" / "spk_uiro"
    shutil.rmtree(spd)
    subprocess.run([str(BIN), "artic-enroll", "-r", str(dst / "d56/_work/ref_uiro.wav"), "-o", str(spd)], cwd=ROOT, env={"PATH": "/usr/bin:/bin"}, check=True, capture_output=True)

    def floor(f: Path) -> float:
        x, sr = sf.read(f, dtype="float64")
        hop = sr // 100
        e = 10 * np.log10(np.array([np.mean(x[i * hop:(i + 1) * hop] ** 2) for i in range(len(x) // hop)]) + 1e-12)
        return float(np.percentile(e, 5) - np.percentile(e, 95))

    print("denoised", len(files), "files | noise floor before %.1f dB → after %.1f dB(中央値・20 本)" % (np.median([floor(f) for f in sorted((src / "d56").glob("p00*_SRC.wav"))[:20]]),
                                                                                              np.median([floor(f) for f in sorted((dst / "d56").glob("p00*_SRC.wav"))[:20]])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
