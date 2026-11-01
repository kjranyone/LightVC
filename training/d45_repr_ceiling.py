"""D45(診断・学習なし): 包絡の表現(メル 128 帯・DCT 25 係数)そのものが明瞭さを削るか。元の音声の励起と位相を残し、スペクトル包絡だけを表現の解像度に落とす(STFT のゲイン → 逆 STFT)。
CheapTrick の包絡 sp(線形 1025 点・教師 f0)に対し: MEL = メル 128 帯の平均の包絡(帯の重みで正規化)を線形の点へ補間・TRUNC = log メル 128(製品の表現の前)を DCT 25 係数に切り詰め・BOTH = 両方(= 製品の表現)・NONE = ゲイン 0(逆 STFT の透明さの検算)。
ゲイン = (表現の包絡 / 元の包絡)を STFT(n_fft 2048・hop 240・Hann・フレーム t の中心 = 240t = CheapTrick の時刻)に掛ける。元 = D37 の男声 TTS の 60 文。CER は d23_cer.py。
    uv run python d45_repr_ceiling.py --src_dir <scratchpad>/d37 --out_dir <scratchpad>/d45
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    import pyworld
    import f0hi as H
    import nvoc as N
    import pae as PA
    import zsvc as Z
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    fb = N.mel_fb().numpy().astype(np.float64)
    D = Z.dct_mat(N.N_MEL).numpy().astype(np.float64)
    nb = fb.shape[1]
    lin = np.arange(nb) * (N.SR / 2) / (nb - 1)
    fc = (fb * lin[None]).sum(1) / fb.sum(1)
    fbn = fb / fb.sum(1, keepdims=True)
    win = torch.hann_window(PA.NFFT, dtype=torch.float64)
    for f in sorted(Path(a.src_dir).glob("p*_SRC.wav")):
        base = f.name[: -len("_SRC.wav")]
        x, sr = sf.read(f, dtype="float64")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0t, _ = H.teacher_f0(x.astype(np.float32), n)
        f0t = f0t[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A), x, np.zeros(PA.A)])
        tt = (PA.A + np.arange(n) * N.HOP) / N.SR
        sp = pyworld.cheaptrick(xa, f0t, tt, N.SR, fft_size=PA.NFFT)
        amp = np.sqrt(np.maximum(sp, 1e-20))
        la = np.log(amp)
        lm_avg = np.log(np.maximum(amp @ fbn.T, 1e-10))
        lmel = np.log(np.maximum(amp @ fb.T, 1e-5))
        ltr = (D[:25].T @ (D[:25] @ lmel.T)).T - lmel
        to_lin = lambda m: np.stack([np.interp(lin, fc, row) for row in m])
        g_mel = to_lin(lm_avg) - la
        g_tr = to_lin(ltr)
        X = torch.stft(torch.from_numpy(x), PA.NFFT, N.HOP, window=win, center=True, return_complex=True)
        T = min(X.shape[1], n)
        for c, g in {"NONE": np.zeros_like(g_mel), "MEL": g_mel, "TRUNC": g_tr, "BOTH": g_mel + g_tr}.items():
            G = torch.from_numpy(np.exp(np.clip(g[:T].T, -12, 12)))
            Y = X.clone()
            Y[:, :T] = X[:, :T] * G
            y = torch.istft(Y, PA.NFFT, N.HOP, window=win, center=True, length=len(x)).numpy()
            sf.write(od / f"{base}_{c}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        shutil.copy(f, od / f.name)
        print(base, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
