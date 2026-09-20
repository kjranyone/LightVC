"""Re-calibrate the reference systems on the SAME set the H-F net is scored on.

12.9' put these five numbers in one table:

    rddsp DSP core   2.648      <- 8 utterances of the 12-SPEAKER H-F test set,
                                   and not the core either: it is the core AFTER
                                   the prior noise was added
    + neural         2.885      <- 12 speakers
    freeC            2.888      <- 12 utterances of ONE speaker (af1ad...)
    BigVGAN v2       4.228      <- same one speaker
    identity         4.644      <- same one speaker

Three of the five come from a different harness: 9.4's calibration set is twelve
utterances of a single speaker, and that speaker is the F0 octave-error case
disclosed in 10. The H-F set is one utterance from each of twelve unseen
speakers. Putting them in one column and reading "the net drew level with the
shipping vocoder" compares systems across different audio -- which is the exact
error this report is about, applied to itself.

This file scores every reference on the H-F test set, so 12.9' can be rewritten
with one harness. freeC is fed centred mel at hop 128 (its own interface) and
BigVGAN its own mel; the DSP core is scored clean, without the prior noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import score_one
from rddsp_gpu import build, stft, istft


def main() -> None:
    _, te = build(80, 12)
    print(f"  H-F test set: {len(te)} unseen speakers, one utterance each\n", flush=True)

    sc: dict[str, list[float]] = {k: [] for k in
                                  ("identity", "bigvgan", "rddsp_core", "freeC")}

    from kansei_train import SNAP
    import bigvgan
    from bigvgan.env import AttrDict
    from bigvgan.meldataset import get_mel_spectrogram
    from train_z1 import load_freebig
    from causal_mel import centered_mel

    h = AttrDict(json.loads((SNAP / "config.json").read_text()))
    big = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    big.load_state_dict(torch.load(SNAP / "bigvgan_generator.pt", map_location="cpu",
                                   weights_only=False)["generator"])
    big.remove_weight_norm()
    big = big.eval()
    # load_freebig places the model on the accelerator; everything else here is
    # CPU-resident (build() returns raw tensors), so pin it to CPU rather than
    # moving twelve utterances back and forth.
    vf = load_freebig("checkpoints/freeC/foundation_lowlatency_5p8ms.pt")[0].to("cpu").eval()

    with torch.no_grad():
        for i, it in enumerate(te):
            gt = it["gt"]
            n = gt.shape[-1]
            sc["identity"].append(score_one(gt, istft(stft(gt), n)))
            sc["rddsp_core"].append(score_one(gt, it["y"][:n]))
            yb = big(get_mel_spectrogram(gt[None], h))[0, 0]
            sc["bigvgan"].append(score_one(gt, yb[:n]))
            yf = vf(centered_mel(gt[None], n_fft=2048, hop=128)).squeeze()
            sc["freeC"].append(score_one(gt, yf[:n]))
            print(f"  u{i:02d}  id {sc['identity'][-1]:.3f}  big {sc['bigvgan'][-1]:.3f}  "
                  f"core {sc['rddsp_core'][-1]:.3f}  freeC {sc['freeC'][-1]:.3f}",
                  flush=True)

    print(f"\n  ---- same {len(te)} unseen speakers, same scorer ----")
    for k in ("identity", "bigvgan", "rddsp_core", "freeC"):
        v = np.array(sc[k])
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        print(f"  {k:12s} {v.mean():.4f}  +-{ci:.4f} (95% CI)")
    print("\n  9.4's calibration used twelve utterances of ONE speaker and is not")
    print("  comparable to these. Both sets are reported; neither replaces the other.")


if __name__ == "__main__":
    main()
