"""D336: 男声の分のうち静的な音色の傾きの分。曲線 = 帯ごとの平均の長時間平均スペクトルの差(MF = MT − FT・REF = MT − 目標の参照・dev の 60 本)を
全ての発話に同じ固定の補正として当て(STFT の振幅・ゼロ位相・診断)、v1⊥ を測る。dev2 / dev3 の変換(曲線を作っていない組)にも同じ曲線を当てる。
    uv run python d336_eq.py --out ../results/conv_p0/d336
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P

ROOT = Path(__file__).resolve().parent.parent
MEL = torchaudio.transforms.MelSpectrogram(48000, n_fft=2048, win_length=1920, hop_length=240, n_mels=80, f_min=0, f_max=24000, power=2.0)
CF = torchaudio.functional.melscale_fbanks(1025, 0, 24000, 80, 48000).argmax(0).numpy() * 24000 / 1024


def read(f) -> np.ndarray:
    x, sr = sf.read(f, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == 48000, (f, sr)
    return x


def ltas(x: np.ndarray) -> np.ndarray:
    l = 10 * np.log10(MEL(torch.from_numpy(x)[None])[0].numpy() + 1e-10)
    e = 10 * np.log10((10 ** (l / 10)).sum(0))
    g = l[:, e > e.max() - 40].mean(1)
    return g - g.mean()


def gain(curve_db: np.ndarray, n_fft: int = 2048) -> torch.Tensor:
    fr = np.linspace(0, 24000, n_fft // 2 + 1)
    o = np.argsort(CF)
    d = np.interp(fr, CF[o], curve_db[o])
    k = np.hanning(31)
    d = np.convolve(np.pad(d, 15, mode="edge"), k / k.sum(), "valid")
    return torch.from_numpy(10 ** (-d / 20)).float()


def apply(x: np.ndarray, g: torch.Tensor) -> np.ndarray:
    w = torch.hann_window(2048)
    X = torch.stft(torch.from_numpy(x), 2048, 512, window=w, return_complex=True)
    y = torch.istft(X * g[:, None], 2048, 512, window=w, length=len(x)).numpy()
    return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=str(ROOT / "results/conv_p0/d328/conv"))
    ap.add_argument("--confirm", default=str(ROOT / "results/conv_p0/d325"))
    a = ap.parse_args()
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    R = P.H / "dev" / "d37" / "_work"
    mt = sorted(glob.glob(f"{a.src}/p00*_CV.wav"))
    ft = sorted(glob.glob(f"{a.src}/p01*_CV.wav"))
    Lm = np.array([ltas(read(f)) for f in mt])
    Lf = np.array([ltas(read(f)) for f in ft])
    tis = [int(re.search(r"_t(\d{3})", f).group(1)) for f in mt]
    Lr = np.array([ltas(read(R / f"ref_{ti:03d}.wav")) for ti in tis])
    curves = {"MF": Lm.mean(0) - Lf.mean(0), "REF": Lm.mean(0) - Lr.mean(0)}
    json.dump({k: v.round(2).tolist() for k, v in curves.items()} | {"cf_hz": CF.round(0).tolist()}, open(O / "curves.json", "w"))
    groups = {"dev": [f"MT={a.src}/p00*_CV.wav", f"FT={a.src}/p01*_CV.wav"]}
    for k, cv in curves.items():
        g = gain(cv)
        od = O / f"dev_{k}"
        od.mkdir(exist_ok=True)
        for f in mt:
            sf.write(od / Path(f).name, apply(read(f), g), 48000)
        groups["dev"].append(f"MT_{k}={od}/p00*_CV.wav")
        for s in ("dev2", "dev3"):
            src = sorted(glob.glob(f"{a.confirm}/{s}/PPG18/p00*.wav"))
            od = O / f"{s}_{k}"
            od.mkdir(exist_ok=True)
            for f in src:
                sf.write(od / Path(f).name, apply(read(f), g), 48000)
            groups.setdefault(s, [f"PPG18={a.confirm}/{s}/PPG18/p00*.wav"]).append(f"PPG18_{k}={od}/p00*.wav")
        print("applied", k, flush=True)
    for s, gs in groups.items():
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *gs, "--out", str(O / f"{s}_v1perp.json")], capture_output=True, text=True)
        print(s, r.stdout[-2500:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
