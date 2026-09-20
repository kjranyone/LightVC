"""A proxy that can see what the ear saw, and a blind re-test of the ear result.

The listening gate said the DSP core's image is distant; rddsp_distance.py
located a 1.29 dB dip at 2-4 kHz; a fixed EQ fitted on the training speakers
closes it to 0.19 dB; PESQ scores that correction at +0.0017 +-0.0039, i.e.
nothing, and the ear says it moves the image closer.

Two consequences, and this file addresses both.

(1) THE FIRST LISTEN WAS NOT BLIND -- the file roles were stated before the
    audition. That is exactly the condition under which a listener confirms what
    they were told. Section `blind` writes the same material under scrambled
    names with the key held in a separate file, so the result can be re-taken
    properly.

(2) PESQ IS BLIND TO THIS DEFECT, so every ranking made today under PESQ alone
    is suspect in the same way -- including the neural refinement's +0.02. Add a
    proxy that does see it: octave-band level error against gt, in dB, which is
    the quantity the ear named. It is not a replacement for PESQ (it is blind to
    everything PESQ is good at, e.g. phase and intelligibility); it is the second
    axis that was missing.

    BLE = mean over bands of |20log10(rms_y / rms_gt)|

Report both axes for every system measured today, so a system that improves PESQ
while making the band error worse becomes visible instead of being promoted.
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
from rddsp_gpu import (build, stft, istft, to_dev, prior_wave, feats, CACHE_DIR,
                       safe_score)
from rddsp_hf import Wavehax2D
from rddsp_staticeq import curve

BANDS = [(125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 8000), (8000, 16000)]
BLIND = Path("/home/kojirotanaka/kjranyone/LightVC/results/rddsp_blind")


def band_levels(x: torch.Tensor) -> np.ndarray:
    w = torch.hann_window(1024)
    S = torch.stft(x, 1024, 256, 1024, w, center=True, return_complex=True).abs()
    f = torch.fft.rfftfreq(1024, 1 / R.SR)
    out = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out.append(20 * np.log10(max(float((S[m] ** 2).sum().sqrt()), 1e-12)))
    return np.array(out)


def ble(y: torch.Tensor, g: torch.Tensor) -> float:
    n = min(len(y), len(g))
    return float(np.abs(band_levels(y[:n]) - band_levels(g[:n])).mean())


def main() -> None:
    tr, te = build(80, 12)
    g_eq = curve(tr, weighted=True, pn=0.0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    nets = {}
    for tag in ("convfix", "coredirw"):
        f = CACHE_DIR / f"{tag}_ch48_s0.pt"
        if not f.exists():
            continue
        ck = torch.load(f, map_location=dev, weights_only=False)
        n = Wavehax2D(cin=4, ch=ck["args"]["ch"], layers=ck["args"]["layers"]).to(dev)
        n.load_state_dict(ck["net"]); n.eval()
        nets[tag] = (n, ck["args"])

    items = to_dev(te, dev)
    res = {k: {"pesq": [], "ble": []} for k in
           ["core", "core+EQ"] + list(nets) + [f"{k}+EQ" for k in nets]}
    ge = torch.Generator(device=dev).manual_seed(999)
    with torch.no_grad():
        for it in items:
            n = min(it["gt"].shape[-1], it["y"].shape[-1])
            gt = it["gt"][:n].cpu()
            outs = {"core": it["y"][:n].cpu()}
            outs["core+EQ"] = istft(stft(outs["core"]) * g_eq[:, None], n)
            for tag, (net, cfg) in nets.items():
                pw = prior_wave(it, ge, dev, cfg.get("pnoise", 0.05),
                                cfg.get("prior", "core"))
                f_, P = feats(it, pw)
                o = net(f_[None])[0]
                S = (torch.complex(o[0], o[1]) if cfg.get("direct") else
                     torch.complex(P.real + o[0], P.imag + o[1]))
                y = istft(S, it["gt"].shape[-1]).cpu()[:n]
                outs[tag] = y
                outs[f"{tag}+EQ"] = istft(stft(y) * g_eq[:, None], n)
            for k, y in outs.items():
                res[k]["pesq"].append(safe_score(gt, y))
                res[k]["ble"].append(ble(y, gt))

    print(f"\n  {len(te)} unseen speakers. PESQ higher is better; BLE lower is better.")
    print(f"  BLE = mean |octave-band level error| vs gt, in dB -- the axis the ear named.\n")
    print(f"  {'system':<16} {'PESQ':>8} {'BLE dB':>9}")
    base = None
    for k in res:
        p = np.mean(res[k]["pesq"]); b = np.mean(res[k]["ble"])
        if k == "core":
            base = (p, b)
        print(f"  {k:<16} {p:8.4f} {b:9.3f}"
              + ("" if base is None or k == "core" else
                 f"   ({p-base[0]:+.4f} PESQ, {b-base[1]:+.3f} dB)"))
    print("\n  A row that gains PESQ while raising BLE is being promoted by a metric")
    print("  that cannot hear the defect the listener named.")

    # blind re-test of the one result that was auditioned non-blind
    BLIND.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(20260804)
    key = ["blind A/B. Two systems, scrambled per trial. Say which is CLOSER.", ""]
    for i, it in enumerate(items[:6]):
        n = min(it["gt"].shape[-1], it["y"].shape[-1])
        gt = it["gt"][:n].cpu().numpy()
        a = it["y"][:n].cpu()
        b = istft(stft(a) * g_eq[:, None], n)
        pair = [("flat", a), ("eq", b)]
        if rng.rand() > 0.5:
            pair = pair[::-1]
        sf.write(BLIND / f"t{i:02d}_ref.wav", gt, R.SR)
        for slot, (name, y) in zip(("X", "Y"), pair):
            z = y.numpy()[:n]
            z = z * float(np.sqrt((gt ** 2).mean() / ((z ** 2).mean() + 1e-12)))
            p_ = np.abs(z).max()
            sf.write(BLIND / f"t{i:02d}_{slot}.wav",
                     (z * (0.95 / p_) if p_ > 0.95 else z).astype(np.float32), R.SR)
            key.append(f"t{i:02d} {slot} = {name}")
    (BLIND / "ANSWER.txt").write_text("\n".join(key) + "\n")
    (BLIND / "README.txt").write_text(
        "For each trial t00..t05: listen to _ref, then X and Y.\n"
        "Say which of X / Y sounds CLOSER to the reference in IMAGE DISTANCE.\n"
        "One is the DSP core as it ships, the other is the same core through a\n"
        "fixed EQ fitted on training speakers only. The order is scrambled per\n"
        "trial and PESQ cannot tell them apart (+0.0017 +-0.0039).\n"
        "Do not open ANSWER.txt until all six are decided.\n")
    print(f"\n  blind re-test written to {BLIND} (6 trials, order scrambled)")


if __name__ == "__main__":
    main()
