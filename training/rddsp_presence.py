"""The presence dip, and the correction PESQ said was worthless.

The ear returned 音像が遠い on the DSP core. rddsp_distance.py located it: the
2-4 kHz presence band sits 1.29 dB below gt, against BigVGAN's 0.31 dB, and the
envelope contrast is down most in the same band. Crest factor is NOT down (it is
0.2-0.9 dB higher than gt), so transients are not the cause.

rddsp_staticeq.py already measured the fix -- a fixed frequency response fitted
on the training speakers -- and scored it at +0.0018 PESQ, i.e. nothing. On that
basis the static-EQ hypothesis was recorded as refuted.

That conclusion was about PESQ, not about the ear, and this project has a written
record of PESQ misranking its own defects twice, plus a measurement in 9.3 of a
0.9-1.9 dB high-band difference that moved PESQ by less than 0.05. A 1.3 dB
presence dip is exactly that kind of defect: inaudible to the metric, and the
first thing a listener names.

So render it and let the ear decide. Three files per speaker, level-matched:

    gt        the reference
    a_flat    the DSP core as it ships
    b_eq      the same core through the fitted curve

The curve is fitted on the 80 TRAINING speakers only, so this is not an oracle;
it is a deployable fixed EQ. Its PESQ delta is +0.002 -- if it sounds closer,
that is a fact about the metric, not about the fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import score_one
from rddsp_gpu import build, stft, istft
from rddsp_staticeq import curve

OUT = Path("/home/kojirotanaka/kjranyone/LightVC/results/rddsp_presence")
N = 6


def gm(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = min(len(y), len(g))
    y, g = y[:n], g[:n]
    y = y * float(np.sqrt((g ** 2).mean() / ((y ** 2).mean() + 1e-12)))
    p = np.abs(y).max()
    return (y * (0.95 / p) if p > 0.95 else y).astype(np.float32)


def band_report(y: torch.Tensor, g: torch.Tensor) -> str:
    out = []
    for lo, hi in ((1000, 2000), (2000, 4000), (4000, 8000)):
        w = torch.hann_window(1024)
        f = torch.fft.rfftfreq(1024, 1 / R.SR)
        m = (f >= lo) & (f < hi)
        def lvl(z):
            S = torch.stft(z, 1024, 256, 1024, w, center=True, return_complex=True).abs()
            return 20 * np.log10(max(float((S[m] ** 2).sum().sqrt()), 1e-12))
        out.append(f"{lo//1000}-{hi//1000}k {lvl(y)-lvl(g):+.2f}")
    return "  ".join(out)


def main() -> None:
    tr, te = build(80, 12)
    g_eq = curve(tr, weighted=True, pn=0.0)
    lo = 20 * np.log10(g_eq.clamp(min=1e-6))
    f = torch.fft.rfftfreq(512, 1 / R.SR)
    print("  fitted curve (dB), fitted on the 80 TRAINING speakers only:")
    for hz in (250, 1000, 2000, 3000, 4000, 8000, 16000):
        i = int(torch.argmin((f - hz).abs()))
        print(f"    {hz:6d} Hz  {float(lo[i]):+6.2f} dB")

    OUT.mkdir(parents=True, exist_ok=True)
    sc = {"a_flat": [], "b_eq": []}
    for i, it in enumerate(te[:N]):
        n = min(it["gt"].shape[-1], it["y"].shape[-1])
        gt = it["gt"][:n]
        flat = it["y"][:n]
        eq = istft(stft(flat) * g_eq[:, None], n)
        m = min(len(gt), len(flat), len(eq))
        sf.write(OUT / f"p{i:02d}_gt.wav", gt[:m].numpy(), R.SR)
        for tag, y in (("a_flat", flat), ("b_eq", eq)):
            sf.write(OUT / f"p{i:02d}_{tag}.wav", gm(y[:m].numpy(), gt[:m].numpy()), R.SR)
            sc[tag].append(score_one(gt[:m], y[:m]))
        print(f"  p{i:02d}  flat {sc['a_flat'][-1]:.3f} [{band_report(flat[:m], gt[:m])}]"
              f"   eq {sc['b_eq'][-1]:.3f} [{band_report(eq[:m], gt[:m])}]", flush=True)

    a, b = np.array(sc["a_flat"]), np.array(sc["b_eq"])
    d = b - a
    ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    key = ["presence-dip A/B -- decide BY EAR first.", "",
           "the ear said the DSP core's image is distant. rddsp_distance.py put",
           "the 2-4 kHz band 1.29 dB below gt (BigVGAN: 0.31 dB), with the",
           "envelope contrast down most in the same band. Crest factor is NOT",
           "down, so this is not blunted transients.", "",
           f"a_flat: the DSP core as it ships          PESQ {a.mean():.3f}",
           f"b_eq:   + a fixed EQ fitted on TRAINING   PESQ {b.mean():.3f}",
           f"        delta {d.mean():+.4f} +-{ci:.4f}", "",
           "the EQ is a deployable fixed curve, not an oracle. PESQ says it is",
           "worth nothing. The question for the ear is whether the image moves",
           "closer. If it does, PESQ is the thing that is wrong here."]
    (OUT / "KEY.txt").write_text("\n".join(key) + "\n")
    print("\n" + "\n".join(key))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
