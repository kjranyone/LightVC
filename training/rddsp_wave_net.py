"""Residual R trained THROUGH the renderer, on a waveform loss.

rddsp_resid_net.py built the gate correctly (zero-init, starts at the DSP core)
and got a clean answer: driving the log-magnitude L1 down by 3.4x moved PESQ
DOWN by 0.12, and the best score in the whole run was step 0. That refutes the
OBJECTIVE, not the method -- matching |STFT(x-h)| exactly just makes the breath
branch reproduce the harmonic fine structure that leaked into the residual, and
rendering that with an arbitrary phase on top of the harmonic branch is what
roughens it.

So train through the renderer instead. The breath branch is a time-varying
min-phase FIR built from nm, and every step of it is a differentiable torch op,
so the gradient of a multi-resolution STFT loss on the OUTPUT WAVEFORM reaches
nm. The net then learns what the renderer can actually realise, rather than a
magnitude that looks right and sounds worse.

Still an overfit gate (train and evaluate on the same three utterances), still
zero-initialised so step 0 is exactly the DSP core, and still conditioned only
on mel + f0 + voicing (0.31x the sample rate).
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
from rddsp_neural import CausalTCN, mel_of
from rddsp_resid_net import resample_rows

STEPS = 1500
CROP = 64          # breath-grid frames per training step
MARGIN = 10        # frames of context dropped from the loss on each side
DEV = "xpu" if torch.xpu.is_available() else "cpu"
FFTS = [(512, 128), (1024, 256), (2048, 512)]


def mrstft(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for nfft, hop in FFTS:
        w = torch.hann_window(nfft, device=y.device)
        Y = torch.stft(y, nfft, hop, nfft, w, center=True, return_complex=True).abs()
        T = torch.stft(t, nfft, hop, nfft, w, center=True, return_complex=True).abs()
        loss = loss + (Y - T).norm() / (T.norm() + 1e-8)
        loss = loss + torch.nn.functional.l1_loss(torch.log(Y + 1e-5),
                                                  torch.log(T + 1e-5))
    return loss / len(FFTS)


def main() -> None:
    nf, nh = R.NOISE_NFFT, R.NOISE_NFFT // R.NOISE_HOPDIV
    items = []
    for uid in [p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]]:
        x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
        gt = torch.tensor(x[: R.SR * 8])
        y, h, nz, p = R.resynthesize(gt)
        base = p["noisefine"]
        T = base.shape[-1]
        f0 = p["f0"]
        cond = torch.cat([resample_rows(mel_of(gt), T),
                          resample_rows(torch.log(R.fill_f0(f0)[None] + 1.0), T),
                          resample_rows((f0 > 50).float()[None], T)], dim=0).float()
        g = torch.Generator().manual_seed(1)
        src = torch.randn(gt.shape[-1], generator=g)
        items.append(dict(uid=uid, gt=gt, h=h, base=base.float(), cond=cond,
                          src=src, dsp=score_one(gt, y), T=T))
        print(f"[{uid}] T {T}  DSP {items[-1]['dsp']:.4f}", flush=True)

    net = CausalTCN(items[0]["cond"].shape[0], nf // 2 + 1).to(DEV)
    nn.init.zeros_(net.out[-1].weight)
    nn.init.zeros_(net.out[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-4, total_steps=STEPS)
    gen = torch.Generator().manual_seed(0)

    def full_render(it, delta_fn):
        with torch.no_grad():
            d = delta_fn(it["cond"])
            nm = (torch.exp(torch.log(it["base"] + 1e-5) + d) - 1e-5).clamp(min=0.0)
            return it["h"] + R._filtered_noise(nm, it["src"], it["gt"].shape[-1], nf, nh)

    def evaluate():
        net.eval()
        ss = []
        for it in items:
            y = full_render(it, lambda c: net(c[None].to(DEV))[0].cpu())
            ss.append(score_one(it["gt"], y))
        net.train()
        return float(np.mean(ss))

    print(f"  step {0:5d}  PESQ {evaluate():.4f}  (zero-init == DSP core)", flush=True)
    t0, best = time.time(), (-1.0, None)
    for step in range(1, STEPS + 1):
        loss = 0.0
        for it in items:
            a = int(torch.randint(MARGIN, it["T"] - CROP - MARGIN, (1,), generator=gen))
            lo, hi = a - MARGIN, a + CROP + MARGIN
            nmc = it["base"][:, lo:hi].to(DEV)
            cc = it["cond"][:, :hi].to(DEV)                 # causal net: keep history
            d = net(cc[None])[0][:, lo:hi]
            nm = torch.exp(torch.log(nmc + 1e-5) + d) - 1e-5
            nm = nm.clamp(min=0.0)
            n = (hi - lo) * nh
            s0 = lo * nh
            src = it["src"][s0: s0 + n].to(DEV)
            if src.shape[-1] < n:
                break
            nz = R._filtered_noise(nm, src, n, nf, nh)
            hh = it["h"][s0: s0 + n].to(DEV)
            tt = it["gt"][s0: s0 + n].to(DEV)
            m0, m1 = MARGIN * nh, n - MARGIN * nh
            loss = loss + mrstft((hh + nz)[m0:m1], tt[m0:m1])
        opt.zero_grad(set_to_none=True)
        (loss / len(items)).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 250 == 0 or step == STEPS:
            m = evaluate()
            if m > best[0]:
                best = (m, step)
            print(f"  step {step:5d}  MRSTFT {float(loss)/len(items):.4f}  "
                  f"PESQ {m:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nDSP core                           {float(np.mean([i['dsp'] for i in items])):.4f}")
    print(f"residual net through the renderer  {best[0]:.4f} (step {best[1]})")
    print(f"oracle-phase ceiling of the DSP    3.8506")
    print(f"target                             4.0000")


if __name__ == "__main__":
    main()
