"""Overfit gate: can a net GENERATE the breath-branch fine structure from mel?

The DSP core caps at 3.435, and at 3.851 even with oracle phase, because the
breath branch can only be handed a smooth envelope -- anything finer is gt being
transported, which is the degeneracy this file's history is made of. Mel is not:
80 bands at 172 fps is 13.8k reals/s = 0.31x the sample rate, and it is what the
VC front end already produces.

So the question that decides whether the ceiling is in the ARCHITECTURE or in
the PARAMETERISATION is: given only mel and f0, can a causal net produce a
breath magnitude good enough to beat 3.851? This is an overfit gate -- train and
evaluate on the same three utterances -- so it answers "can this be expressed at
all", not "does it generalise". CLAUDE.md allows exactly that for 切り分け.

The harmonic branch keeps its measured parameters (0.07x of the sample rate,
at or below critical for the band it covers), so this isolates the breath branch.
"""
from __future__ import annotations

import math
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

N_MEL = 80
STEPS = 4000
DEV = "xpu" if torch.xpu.is_available() else "cpu"


class CausalTCN(nn.Module):
    """Causal dilated conv stack. groups=1 throughout (XPU backward fails on
    depthwise), and left-padding only, so no frame ever sees its future."""

    def __init__(self, cin: int, cout: int, ch: int = 320, layers: int = 8):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch, 1)
        self.convs = nn.ModuleList()
        self.res = nn.ModuleList()
        self.dil = [2 ** (i % 4) for i in range(layers)]
        for d in self.dil:
            self.convs.append(nn.Conv1d(ch, 2 * ch, 3, dilation=d))
            self.res.append(nn.Conv1d(ch, ch, 1))
        self.out = nn.Sequential(nn.LeakyReLU(0.1), nn.Conv1d(ch, ch, 1),
                                 nn.LeakyReLU(0.1), nn.Conv1d(ch, cout, 1))

    def forward(self, x):
        h = self.inp(x)
        for c, r, d in zip(self.convs, self.res, self.dil):
            y = torch.nn.functional.pad(h, (2 * d, 0))
            a, b = c(y).chunk(2, dim=1)
            h = h + r(torch.tanh(a) * torch.sigmoid(b))
        return self.out(h)


def mel_of(x: torch.Tensor) -> torch.Tensor:
    m = librosa.feature.melspectrogram(y=x.numpy(), sr=R.SR, n_fft=2048,
                                       hop_length=R.HOP, n_mels=N_MEL, power=1.0)
    return torch.log(torch.tensor(m) + 1e-5)


def main() -> None:
    nf = R.NOISE_NFFT
    nh = nf // R.NOISE_HOPDIV
    items = []
    for uid in [p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]]:
        x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
        gt = torch.tensor(x[: R.SR * 8])
        y, h, nz, p = R.resynthesize(gt)
        tgt = torch.log(R.stft(gt - h, nf, nh).abs() + 1e-5)
        f0 = p["f0"]
        # condition = mel + log f0 + voicing, resampled to the breath grid
        mel = mel_of(gt)
        T = tgt.shape[-1]
        src_t = torch.linspace(0, 1, mel.shape[-1])
        dst_t = torch.linspace(0, 1, T)
        idx = torch.searchsorted(src_t, dst_t).clamp(0, mel.shape[-1] - 1)
        c = torch.cat([mel[:, idx],
                       torch.log(R.fill_f0(f0)[None] + 1.0)[:, (torch.arange(T) *
                            (len(f0) - 1) / max(T - 1, 1)).long()],
                       (f0 > 50).float()[None][:, (torch.arange(T) *
                            (len(f0) - 1) / max(T - 1, 1)).long()]], dim=0)
        items.append((uid, gt, h, p, c.float(), tgt.float()))
        print(f"[{uid}] frames {T}  cond {tuple(c.shape)}  target {tuple(tgt.shape)}", flush=True)

    net = CausalTCN(items[0][4].shape[0], nf // 2 + 1).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=STEPS)
    t0 = time.time()
    for step in range(STEPS):
        loss = 0.0
        for _, _, _, _, c, tgt in items:
            pred = net(c[None].to(DEV))[0]
            loss = loss + torch.nn.functional.l1_loss(pred, tgt.to(DEV))
        opt.zero_grad(set_to_none=True)
        (loss / len(items)).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 500 == 0 or step == STEPS - 1:
            print(f"  step {step:5d}  L1 {float(loss)/len(items):.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    net.eval()
    dsp, neu = [], []
    with torch.no_grad():
        for uid, gt, h, p, c, tgt in items:
            n = gt.shape[-1]
            y, _, _, _ = R.resynthesize(gt)
            dsp.append(score_one(gt, y))
            nm = torch.exp(net(c[None].to(DEV))[0].cpu()) - 1e-5
            nm = nm.clamp(min=0.0)
            g = torch.Generator().manual_seed(1)
            src = torch.randn(n, generator=g)
            yz = h + R._shaped_noise(nm, src, n, 0, nf, nh)
            neu.append(score_one(gt, yz))
            print(f"[{uid}] DSP {dsp[-1]:.3f} -> mel-predicted breath {neu[-1]:.3f}", flush=True)
    print(f"\nDSP core                     {float(np.mean(dsp)):.4f}")
    print(f"neural breath (overfit gate) {float(np.mean(neu)):.4f}")
    print(f"oracle-phase ceiling of DSP  3.8506")


if __name__ == "__main__":
    main()
