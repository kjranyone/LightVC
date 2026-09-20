"""Residual R (design doc 4.7) as a proper overfit gate.

The first attempt (rddsp_neural.py) asked a net to predict the whole breath
magnitude from mel and scored 3.263, below the 3.435 DSP core. That is an
implementation failing, not a verdict on the method -- the rule this project
already wrote down after four such accidents in one day.

This is the formulation the design doc actually specifies: the net predicts a
RESIDUAL on top of the DSP skeleton, and the output layer is zero-initialised,
so at step 0 the rendered signal is byte-identical to the DSP core. Training can
therefore only move away from 3.435, and whether it moves up is the answer.

Rate ledger of the conditioning: mel 80 x 172 fps = 13.8k reals/s = 0.31x the
sample rate, plus f0 and voicing. The net never sees a full-resolution
spectrum -- the DSP envelope it corrects is the cepstrally truncated one the
declared interface already carries, so the fine structure it adds is GENERATED,
which is the whole point of the exercise.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from kansei_train import CACHE, DATA
from rddsp_loop import score_one
from rddsp_neural import CausalTCN, mel_of, N_MEL

STEPS = 6000
DEV = "xpu" if torch.xpu.is_available() else "cpu"


def resample_rows(v: torch.Tensor, T: int) -> torch.Tensor:
    idx = (torch.arange(T) * (v.shape[-1] - 1) / max(T - 1, 1)).long()
    return v[:, idx]


def main() -> None:
    nf = R.NOISE_NFFT
    nh = nf // R.NOISE_HOPDIV
    items = []
    for uid in [p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]]:
        x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
        gt = torch.tensor(x[: R.SR * 8])
        y, h, nz, p = R.resynthesize(gt)
        base = p["noisefine"]                       # what the renderer is handed
        tgt = torch.log(R.stft(gt - h, nf, nh).abs() + 1e-5)
        T = min(base.shape[-1], tgt.shape[-1])
        base, tgt = base[:, :T], tgt[:, :T]
        f0 = p["f0"]
        cond = torch.cat([resample_rows(mel_of(gt), T),
                          resample_rows(torch.log(R.fill_f0(f0)[None] + 1.0), T),
                          resample_rows((f0 > 50).float()[None], T)], dim=0).float()
        items.append((uid, gt, h, base.float(), cond, tgt.float(),
                      score_one(gt, y)))
        print(f"[{uid}] T {T}  cond {tuple(cond.shape)}  DSP {items[-1][-1]:.4f}", flush=True)

    net = CausalTCN(items[0][4].shape[0], nf // 2 + 1).to(DEV)
    # zero-init the last layer: step 0 renders exactly the DSP core.
    last = net.out[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 8e-4, total_steps=STEPS)

    def render(h, nm, n, seed=1):
        g = torch.Generator().manual_seed(seed)
        return h + R._shaped_noise(nm, torch.randn(n, generator=g), n, 0, nf, nh)

    t0 = time.time()
    best = (-1.0, None)
    for step in range(STEPS):
        loss = 0.0
        for _, _, _, base, cond, tgt, _ in items:
            d = net(cond[None].to(DEV))[0]
            pred = torch.log(base.to(DEV) + 1e-5) + d
            loss = loss + torch.nn.functional.l1_loss(pred, tgt.to(DEV))
        opt.zero_grad(set_to_none=True)
        (loss / len(items)).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 1000 == 0 or step == STEPS - 1:
            net.eval()
            with torch.no_grad():
                ss = []
                for _, gt, h, base, cond, _, _ in items:
                    d = net(cond[None].to(DEV))[0].cpu()
                    nm = (torch.exp(torch.log(base + 1e-5) + d) - 1e-5).clamp(min=0.0)
                    ss.append(score_one(gt, render(h, nm, gt.shape[-1])))
                m = float(np.mean(ss))
            net.train()
            if m > best[0]:
                best = (m, step)
            print(f"  step {step:5d}  L1 {float(loss)/len(items):.4f}  "
                  f"PESQ {m:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    dsp = float(np.mean([it[-1] for it in items]))
    print(f"\nDSP core                          {dsp:.4f}")
    print(f"residual net, best during training {best[0]:.4f} (step {best[1]})")
    print(f"oracle-phase ceiling of the DSP    3.8506")
    print(f"target                             4.0000")


if __name__ == "__main__":
    main()
