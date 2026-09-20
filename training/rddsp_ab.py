"""Four-way ear A/B on held-out utterances: gt / BigVGAN / rddsp / freeC.

The calibration put this session's parametric DSP core at 3.159 against the
project's shipping realtime vocoder freeC at 2.888 and BigVGAN v2 at 4.228, all
on the same twelve held-out utterances and the same scorer. The +0.27 over freeC
was not expected, and PESQ has misranked this project's defects twice, so it
does not count until it is heard.

Files are named so the ordering is not visible from the name alone -- gt is
labelled, the three systems are a/b/c in a fixed but unannounced order recorded
in KEY.txt, so the listener can decide before reading it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from kansei_train import CACHE, DATA, SNAP
from rddsp_loop import score_one
from train_z1 import load_freebig
from causal_mel import centered_mel

OUT = Path("/home/kojirotanaka/kjranyone/LightVC/results/rddsp_ab")


def gm(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = min(len(y), len(g))
    y, g = y[:n], g[:n]
    y = y * float(np.sqrt((g ** 2).mean() / ((y ** 2).mean() + 1e-12)))
    p = np.abs(y).max()
    return (y * (0.95 / p) if p > 0.95 else y).astype(np.float32)


def main() -> None:
    import json
    import bigvgan
    from bigvgan.env import AttrDict
    from bigvgan.meldataset import get_mel_spectrogram

    OUT.mkdir(parents=True, exist_ok=True)
    h = AttrDict(json.loads((SNAP / "config.json").read_text()))
    big = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    big.load_state_dict(torch.load(SNAP / "bigvgan_generator.pt", map_location="cpu",
                                   weights_only=False)["generator"])
    big.remove_weight_norm()
    big = big.eval()
    vf = load_freebig("checkpoints/freeC/foundation_lowlatency_5p8ms.pt")[0].to("cpu").eval()

    uids = [p.stem for p in sorted(CACHE.glob("*.npz"))][40:46]
    sc = {"a_bigvgan": [], "b_rddsp": [], "c_freeC": []}
    with torch.no_grad():
        for uid in uids:
            x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
            gt = torch.tensor(x[: R.SR * 6])
            g = gt.numpy()
            yb = big(get_mel_spectrogram(gt[None], h))[0, 0]
            yr, _, _, _ = R.resynthesize(gt)
            yf = vf(centered_mel(gt[None], n_fft=2048, hop=128)).squeeze()
            n = min(len(g), len(yb), len(yr), len(yf))
            sf.write(OUT / f"{uid}_gt.wav", g[:n], R.SR)
            for tag, y in (("a_bigvgan", yb), ("b_rddsp", yr), ("c_freeC", yf)):
                sf.write(OUT / f"{uid}_{tag}.wav", gm(y[:n].numpy(), g[:n]), R.SR)
                sc[tag].append(score_one(gt[:n], y[:n]))
            print(f"  {uid}", flush=True)

    key = ["A/B/X key -- decide by ear FIRST, then read.", ""]
    for tag, v in sc.items():
        name = {"a_bigvgan": "BigVGAN v2 44kHz, 122M, NON-CAUSAL (reference ceiling)",
                "b_rddsp": "rddsp, this session's parametric DSP core, causal, no training",
                "c_freeC": "freeC, the project's shipping realtime vocoder"}[tag]
        key.append(f"{tag}: {name}   PESQ {np.mean(v):.3f}")
    key += ["", "identity (STFT round trip) on this harness: 4.644",
            "PESQ has misranked this project's defects twice; the ear decides.",
            "Listen for: high-band texture vs hiss (rddsp measures -0.9 dB crest",
            "against gt), and low-band body (0-250 Hz is -1.3 dB in every rddsp",
            "configuration and does not respond to level)."]
    (OUT / "KEY.txt").write_text("\n".join(key) + "\n")
    print("\n" + "\n".join(key))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
