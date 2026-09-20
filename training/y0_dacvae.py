"""Y-0: DACVAE latent の再構成品質確認（案 Y の全ての関門）。

GT → DACVAE encode → decode の往復が耳で透明なら latent 表現の天井は十分。
namikawa(実男声 mp3) と held 女声 GT で確認。

    cd Irodori-TTS && HF_HUB_OFFLINE=1 .venv/bin/python ../LightVC/training/y0_dacvae.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Irodori-TTS"))
from irodori_tts.codec import DACVAECodec

OUT = Path("/home/kojirotanaka/kjranyone/LightVC/results/y0")


def recon(codec: DACVAECodec, path: str, name: str, sr: int = 44100) -> None:
    w, _ = librosa.load(path, sr=sr, mono=True)
    x = torch.from_numpy(w)[None]                     # [1, T]
    z = codec.encode_waveform(x, sr)
    y = codec.decode_latent(z)
    y = y[0, 0].detach().cpu().numpy() if y.dim() == 3 else y[0].detach().cpu().numpy()
    T = min(len(w), len(y))
    soundfile.write(OUT / f"{name}_dacvae.wav", np.clip(y[:T], -1, 1), sr)
    soundfile.write(OUT / f"{name}_gt.wav", w[:T], sr)
    print(f"  {name}: latent {tuple(z.shape)} -> {T/sr:.1f}s")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    codec = DACVAECodec.load(device="cuda", deterministic_encode=True,
                             deterministic_decode=True, normalize_db=None)
    LVC = Path("/home/kojirotanaka/kjranyone/LightVC")
    recon(codec, str(LVC / "namikawa.mp3"), "namikawa")
    for i, f in enumerate(["fe659435bbd284e8_00005555.wav",
                           "fe659435bbd284e8_00023675.wav"]):
        p = LVC / "female-dataset/fe659435bbd284e8" / f
        if p.exists():
            recon(codec, str(p), f"female{i}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
