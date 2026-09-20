"""What does PESQ 4.0 mean IN THIS HARNESS?

The literature survey put BigVGAN at PESQ 4.027 on LibriTTS copy-synthesis and
Vocos at 3.487 -- but on a different corpus, a different sampling rate, and
somebody else's PESQ call. Our harness resamples 44.1 kHz to 16 kHz, aligns by
cross-correlation, and scores PESQ-wb, and it reads 4.644 for the identity. None
of those numbers are commensurable until a known-good system is run through OUR
scorer on OUR utterances.

BigVGAN v2 44 kHz is already on this machine (it is what kansei_train.py uses),
so this is measurable rather than arguable. It is a 112M-parameter GAN vocoder
trained on large-scale data -- an upper reference for what a mel-conditioned
neural vocoder achieves, against which our 3.16 DSP core and the 4.0 target can
both be placed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from kansei_train import CACHE, DATA, SNAP
from rddsp_loop import score_one


def main() -> None:
    import json
    import bigvgan  # noqa: E402
    from bigvgan.env import AttrDict  # noqa: E402
    from bigvgan.meldataset import get_mel_spectrogram  # noqa: E402

    h = AttrDict(json.loads((SNAP / "config.json").read_text()))
    model = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    ck = torch.load(SNAP / "bigvgan_generator.pt", map_location="cpu",
                    weights_only=False)
    model.load_state_dict(ck["generator"])
    model.remove_weight_norm()
    model = model.eval()
    print(f"BigVGAN v2: sr {h.sampling_rate} hop {h.hop_size} mels {h.num_mels} "
          f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    uids = [p.stem for p in sorted(CACHE.glob("*.npz"))][40:52]
    big, dsp, ident = [], [], []
    with torch.no_grad():
        for uid in uids:
            x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
            gt = torch.tensor(x[: R.SR * 6])
            mel = get_mel_spectrogram(gt[None], h)
            y = model(mel)[0, 0]
            n = min(len(y), len(gt))
            big.append(score_one(gt[:n], y[:n]))
            yd, _, _, _ = R.resynthesize(gt)
            dsp.append(score_one(gt, yd))
            ident.append(score_one(gt, R.istft(R.stft(gt), gt.shape[-1])))
            print(f"  {uid}  BigVGAN {big[-1]:.3f}   our DSP {dsp[-1]:.3f}", flush=True)

    print(f"\n{'system':38s} {'PESQ (our harness)':>18s}")
    print(f"{'identity (STFT round trip)':38s} {np.mean(ident):18.3f}")
    print(f"{'BigVGAN v2 44kHz (112M, GAN, mel)':38s} {np.mean(big):18.3f}")
    print(f"{'our DSP core (parametric, causal)':38s} {np.mean(dsp):18.3f}")
    print(f"\nliterature, other corpora/harnesses: BigVGAN 4.027, Vocos 3.487, "
          f"APNet2 2.56-3.61")


if __name__ == "__main__":
    main()
