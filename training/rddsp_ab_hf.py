"""Ear gate on unseen speakers: gt / rddsp core / freeC / BigVGAN.

rddsp_ab.py did this on af1ad5575a3fa383 only -- the one speaker 10 discloses as
an F0 octave-error case and 12.14 shows is freeC's best. On the twelve unseen
speakers the ordering PESQ reports is very different:

    BigVGAN 4.0097   rddsp core 2.9726   freeC 2.2012

CLAUDE.md makes the ear the promotion gate and forbids promoting on a proxy
alone, and this project's record has PESQ misranking its own defects twice. A
+0.77 gap over the shipping vocoder is exactly the size of claim that has to be
heard before it is written down.

Files are named so the ordering is not readable from the name: gt is labelled,
the three systems are a/b/c in a fixed but unannounced order recorded in
KEY.txt. Decide first, read afterwards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import score_one
from rddsp_gpu import build

OUT = Path("/home/kojirotanaka/kjranyone/LightVC/results/rddsp_ab_hf")
N = 6


def gm(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = min(len(y), len(g))
    y, g = y[:n], g[:n]
    y = y * float(np.sqrt((g ** 2).mean() / ((y ** 2).mean() + 1e-12)))
    p = np.abs(y).max()
    return (y * (0.95 / p) if p > 0.95 else y).astype(np.float32)


def main() -> None:
    from kansei_train import SNAP
    import bigvgan
    from bigvgan.env import AttrDict
    from bigvgan.meldataset import get_mel_spectrogram
    from train_z1 import load_freebig
    from causal_mel import centered_mel

    _, te = build(80, 12)
    te = te[:N]
    h = AttrDict(json.loads((SNAP / "config.json").read_text()))
    big = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    big.load_state_dict(torch.load(SNAP / "bigvgan_generator.pt", map_location="cpu",
                                   weights_only=False)["generator"])
    big.remove_weight_norm()
    big = big.eval()
    vf = load_freebig("checkpoints/freeC/foundation_lowlatency_5p8ms.pt")[0].to("cpu").eval()

    OUT.mkdir(parents=True, exist_ok=True)
    sc = {"a_freeC": [], "b_rddsp": [], "c_bigvgan": []}
    with torch.no_grad():
        for i, it in enumerate(te):
            gt = it["gt"]
            g = gt.numpy()
            outs = [("a_freeC", vf(centered_mel(gt[None], n_fft=2048, hop=128)).squeeze()),
                    ("b_rddsp", it["y"]),
                    ("c_bigvgan", big(get_mel_spectrogram(gt[None], h))[0, 0])]
            m = min([len(g)] + [len(y) for _, y in outs])
            sf.write(OUT / f"s{i:02d}_gt.wav", g[:m], R.SR)
            for tag, y in outs:
                sf.write(OUT / f"s{i:02d}_{tag}.wav", gm(y[:m].numpy(), g[:m]), R.SR)
                sc[tag].append(score_one(gt[:m], y[:m]))
            print(f"  s{i:02d}  " + "  ".join(f"{t} {sc[t][-1]:.3f}" for t in sc),
                  flush=True)

    names = {"a_freeC": "freeC -- the project's SHIPPING realtime vocoder (mel, 27.9M)",
             "b_rddsp": "rddsp DSP core -- causal, no training, but receives MEASURED "
                        "glottal phase, so it has no product path (9.4 / 12.9')",
             "c_bigvgan": "BigVGAN v2 44 kHz -- mel, NON-causal, 122M (ceiling)"}
    key = ["ear gate -- 6 speakers never used in any training here.", "",
           "decide BY EAR first, then read.", ""]
    for tag, v in sc.items():
        key.append(f"{tag}: {names[tag]}   PESQ {np.mean(v):.3f}")
    key += ["", "identity on this harness: 4.644",
            "",
            "the claim under test: on UNSEEN speakers PESQ puts the untrained DSP",
            "core 0.77 above the shipping vocoder. On freeC's own best speaker the",
            "same code puts them level (2.87 vs 3.16). PESQ has misranked this",
            "project's defects twice, so this ordering does not count until heard.",
            "",
            "listen for: freeC's failure mode on unseen speakers (does it lose",
            "identity, or add noise, or dull the band?) versus the DSP core's",
            "known defects (high-band hiss, and 0-250 Hz measured 1.3 dB low in",
            "every configuration)."]
    (OUT / "KEY.txt").write_text("\n".join(key) + "\n")
    print("\n" + "\n".join(key))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
