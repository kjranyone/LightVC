"""D430: 表への追従と自然さの引き換えの検査 — 出力の包絡の時間の動き(log-mel 40 帯の時間の標準偏差の平均・有声の上位 50% のフレーム・0.5〜8 Hz の変調のパワー)を系ごとに比べる。
表に強く従う系ほど動きが平板(単位の平均へ寄る)なら、自然さとの引き換えは『動きの供給源』の問題。CPU だけ。
    uv run python d430_dyn.py --out ../results/conv_p0/d430.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

R = Path(__file__).resolve().parent.parent / "results"
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
SYS = {
    "REAL_ref": f"{H}/dev/d37/_work/ref_*.wav",
    "B": f"{R}/conv_p0/d374m/conv/p00*_CV.wav",
    "D2": f"{R}/conv_p0/d428m_002500/conv/p00*_CV.wav",
    "P38": f"{R}/conv_p0/d426m_005000/conv/p00*_CV.wav",
    "P40_5k": f"{R}/conv_p0/d429m_005000/conv/p00*_CV.wav",
    "P40_15k": f"{R}/conv_p0/d429m_015000/conv/p00*_CV.wav",
    "PROD": f"{H}/dev/d114/p00*_G03S10ZH.wav",
    "SRC": f"{H}/dev/d56/p00*_SRC.wav",
}


def logmel(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    import librosa
    m = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=512, hop_length=160, n_mels=40, fmax=8000)
    return np.log(m + 1e-6)


def feats(p: str) -> tuple:
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = resample_poly(x.mean(1)[: 8 * sr], 16000, sr).astype(np.float32)
    L = logmel(x)
    e = L.mean(0)
    keep = e > np.percentile(e, 50)
    Lk = L[:, keep]
    lmstd = float(Lk.std(1).mean())
    d = np.abs(np.diff(L, axis=1)).mean(0)
    flux = float(d[keep[1:]].mean())
    Lc = L - L.mean(1, keepdims=True)
    S = np.abs(np.fft.rfft(Lc, axis=1)) ** 2
    f = np.fft.rfftfreq(L.shape[1], 0.01)
    mod = float(10 * np.log10(S[:, (f >= 0.5) & (f < 8)].sum() / S.sum()))
    return lmstd, flux, mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {}
    for n, pat in SYS.items():
        F = np.array([feats(p) for p in sorted(glob.glob(pat))])
        rep[n] = {"n": len(F), "lmstd": round(float(F[:, 0].mean()), 4), "flux": round(float(F[:, 1].mean()), 4), "mod_0.5_8Hz_db": round(float(F[:, 2].mean()), 2)}
        print(n.ljust(9), rep[n], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
