"""D257(学習なし・results/conv_p0/d257_prereg.yaml): 音色の差を時間不変の EQ(D141 と同じ作り方・登録の定数だけ)で音声に直接かける。SRCEQ = 実女声の入力 × EQ(目標の参照 − 操作者の参照)・S000EQ = D256 の S000 × EQ(目標の参照 − 同じ設定で操作者の参照を変換した出力)。
    uv run python d257_eq.py --s <dev の親>
"""
from __future__ import annotations

import argparse
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target_dev/release/lightvc-app"
SR = 48000
CF = 1000.0 * 2.0 ** (np.arange(-13, 15) / 3.0)
CF = CF[(CF >= 60) & (CF <= 20000)]
S000 = ["--rvoc", "results/rvoc3hi/export", "--mix", "0", "--per-p75", "--med-from-operator", "--f4", "results/f4_1/export", "--warp", "--causal-ct", "--global-map", "--alpha", "1", "--gmap-scale", "0"]


def load48(p: Path) -> np.ndarray:
    from math import gcd
    from scipy.signal import resample_poly
    x, sr = sf.read(str(p), dtype="float64", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x


def ltas(x: np.ndarray) -> np.ndarray:
    n = len(x) // 240
    e = 10 * np.log10((x[: n * 240].reshape(n, 240) ** 2).mean(1) + 1e-12)
    act = e > e.max() - 40
    fq, _, Z = ss.stft(x, SR, nperseg=2048, noverlap=2048 - 240)
    P = np.abs(Z[:, :n]) ** 2
    P = P[:, act[: P.shape[1]]].mean(1)
    L = 10 * np.log10(np.array([P[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() + 1e-20 for c in CF]))
    return L - L.mean()


def eq(x: np.ndarray, g: np.ndarray, strength: float) -> np.ndarray:
    g = np.convolve(np.pad(g, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    g = np.clip(strength * (g - np.median(g)), -12, 12)
    fr = np.r_[0.0, CF / (SR / 2), 1.0]
    gl = 10 ** (np.r_[g[0], g, g[-1]] / 20)
    h = ss.minimum_phase(ss.firwin2(1023, np.clip(fr, 0, 1), gl ** 2, fs=2.0), method="homomorphic")
    z = ss.lfilter(h, 1.0, x)
    return (z / (np.abs(z).max() + 1e-9) * 0.9).astype(np.float32)


def job(cmd: list) -> None:
    if not Path(cmd[-1]).exists():
        subprocess.run([str(c) for c in cmd], cwd=ROOT, env={"PATH": "/usr/bin:/bin"}, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    a = ap.parse_args()
    F = Path(a.s) / "devff"
    od = F / "d257"
    (od / "_cal").mkdir(parents=True, exist_ok=True)
    srcs = sorted((F / "d56").glob("p00*_SRC.wav"))
    jobs = []
    for f in srcs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        jobs.append([BIN, "artic", "--enroll", F / "d85_work" / f"en_{ti:03d}", "--operator-enroll", F / "d85_work" / f"op_{pi:04d}"] + S000 + ["-i", F / "d56/_work" / f"op_{pi:04d}.wav", "-o", od / "_cal" / f"op{pi:04d}_t{ti:03d}.wav"])
    with Pool(8) as pl:
        list(pl.imap_unordered(job, jobs))
    tl: dict = {}
    for f in srcs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        k = f.stem[:-4]
        if ti not in tl:
            tl[ti] = ltas(load48(F / "d37/_work" / f"ref_{ti:03d}.wav"))
        greal = tl[ti] - ltas(load48(F / "d56/_work" / f"op_{pi:04d}.wav"))
        gcal = tl[ti] - ltas(load48(od / "_cal" / f"op{pi:04d}_t{ti:03d}.wav"))
        sf.write(od / f"{k}_SRCEQ.wav", eq(load48(f), greal, 1.0), SR)
        y = load48(F / "d256" / f"{k}_S000.wav")
        sf.write(od / f"{k}_S000EQ.wav", eq(y, gcal, 1.0), SR)
        sf.write(od / f"{k}_S000EQ5.wav", eq(y, gcal, 0.5), SR)
    print("written", len(srcs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
