"""D335: 自己再構成(SR)の出力は本物より時間方向に均されているか(平均への寄せの徴候)。log-mel(48 kHz・hop 240 = 200 fps・80 帯)の各帯の軌道の変調の強さを
変調周波数の帯(2〜4・4〜8・8〜16・16〜32・32〜64 Hz)× 周波数の帯(〜1k・1〜4k・4〜12k・12k〜)で求め、本物(同じクリップ)との比(dB)を群ごとに出す。
声のある区間だけでなく全体(無音の端は −60 dB 相対の床で切る)。整列: SR / CP は入力と同じ時間軸(遅延を除いた描画)= 同じ長さに切る。
    uv run python d335_modspec.py --groups G0=../results/conv_p0/d334/G0 CP=../results/diag_e2e1/eval:CP --out ../results/conv_p0/d335.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P

MB = [(2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
FB = [(0, 1000), (1000, 4000), (4000, 12000), (12000, 24000)]


def logmel(x: np.ndarray, mel) -> np.ndarray:
    m = mel(torch.from_numpy(x).float()[None])[0].numpy()
    lm = np.log10(m + 1e-10) * 10
    return np.maximum(lm, lm.max() - 60.0)


def modpow(lm: np.ndarray) -> np.ndarray:
    z = lm - lm.mean(1, keepdims=True)
    T = z.shape[1]
    S = np.abs(np.fft.rfft(z * np.hanning(T)[None], axis=1)) ** 2
    fr = np.fft.rfftfreq(T, 1 / 200.0)
    return np.stack([S[:, (fr >= a) & (fr < b)].mean(1) for a, b in MB], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True, help="NAME=dir[:suffix](既定の suffix = SR)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slow", action="store_true", help="変調の帯を 0.125〜0.25・0.25〜0.5・0.5〜1・1〜2・2〜4 Hz にする(遅い音色の揺らぎ・D338 の続き)")
    a = ap.parse_args()
    if a.slow:
        MB[:] = [(0.125, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4)]
    mel = torchaudio.transforms.MelSpectrogram(48000, n_fft=2048, win_length=1920, hop_length=240, n_mels=80, f_min=0, f_max=24000, power=2.0)
    cf = torchaudio.functional.melscale_fbanks(1025, 0, 24000, 80, 48000).argmax(0).numpy() * 24000 / 1024
    held = P.e2e_held()
    W = P.H / "e2e" / "wav"
    out = {}
    for g in a.groups:
        name, rest = g.split("=", 1)
        d, suf = (rest.split(":") + ["SR"])[:2]
        rat = []
        for i, (_, ka, _) in enumerate(held):
            f = Path(d) / f"h{i:04d}_t{i:03d}_{suf}.wav"
            if not f.exists():
                continue
            y, sr = sf.read(f, dtype="float32")
            x, sx = sf.read(W / f"{ka}.wav", dtype="float32")
            assert sr == 48000 and sx == 48000, (sr, sx)
            n = min(len(x), len(y))
            px, py = modpow(logmel(x[:n], mel)), modpow(logmel(y[:n], mel))
            r = np.zeros((len(FB), len(MB)))
            for j, (lo, hi) in enumerate(FB):
                b = (cf >= lo) & (cf < hi)
                r[j] = 10 * np.log10(py[b].mean(0) / px[b].mean(0))
            rat.append(r)
        R = np.array(rat)
        out[name] = {"n": len(R), "mean_db": R.mean(0).round(2).tolist(), "se_db": (R.std(0, ddof=1) / np.sqrt(len(R))).round(2).tolist()}
        print(name, "n", len(R), flush=True)
        for j, (lo, hi) in enumerate(FB):
            print(f"  {lo:>5}-{hi:<5}Hz", " ".join(f"{m:+.1f}±{s:.1f}" for m, s in zip(out[name]["mean_db"][j], out[name]["se_db"][j])), flush=True)
    out["mod_bands_hz"], out["freq_bands_hz"] = MB, FB
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
