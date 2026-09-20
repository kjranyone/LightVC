"""Task A, second stage: the objective, not the data.

The multi-speaker run answered the question it was built for. Scale flipped the
sign -- 300 speakers gave +0.037 on unseen speakers with unseen noise seeds,
where one speaker gave zero or less -- but +0.037 against a +1.0 gap says the
limiter is not data. Multi-resolution STFT does not penalise phase directly, so
it is a weak driver for a branch whose whole job is to generate one.

So keep everything that made the multi-speaker run trustworthy and change only
the objective: add a multi-resolution STFT discriminator on the output waveform,
with feature matching. The DSP skeleton stays frozen and only the breath
branch's texture is learned, which is the one place CLAUDE.md allows a GAN.

Protocol unchanged and non-negotiable:
  * fresh noise realisation every step (memorising a draw is impossible)
  * absolute phase generation, zero-initialised so step 0 renders the DSP core
  * train and test speakers disjoint
  * evaluation on noise seeds never used in training
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import score_one
from rddsp_neural import CausalTCN
from rddsp_wave_net import mrstft, CROP, MARGIN, DEV
from rddsp_dspecies_honest import realise
from rddsp_dspecies_multi import pick, prep, N_TRAIN, N_TEST, EVAL_N

STEPS = 20000
ADV_W = 0.6          # adversarial weight
FM_W = 2.0           # feature matching weight
D_START = 1500       # let the generator settle before the discriminator bites
RES = [(512, 128), (1024, 256), (2048, 512)]


class SpecDisc(nn.Module):
    """One discriminator per STFT resolution, on the log-magnitude."""

    def __init__(self, ch: int = 48):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in RES:
            self.blocks.append(nn.ModuleList([
                nn.Conv2d(1, ch, (3, 9), padding=(1, 4)),
                nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4)),
                nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4)),
                nn.Conv2d(ch, ch, (3, 3), padding=(1, 1)),
                nn.Conv2d(ch, 1, (3, 3), padding=(1, 1)),
            ]))

    def forward(self, x):
        outs, feats = [], []
        for (nfft, hop), block in zip(RES, self.blocks):
            w = torch.hann_window(nfft, device=x.device)
            S = torch.stft(x, nfft, hop, nfft, w, center=True, return_complex=True)
            h = torch.log(S.abs() + 1e-5)[None, None]
            for i, c in enumerate(block):
                h = c(h)
                if i < len(block) - 1:
                    h = torch.nn.functional.leaky_relu(h, 0.1)
                    feats.append(h)
            outs.append(h)
        return outs, feats


def main() -> None:
    nf, nh = R.NOISE_NFFT, R.NOISE_NFFT // R.NOISE_HOPDIV
    NB = nf // 2 + 1
    tr_p, te_p = pick(N_TRAIN, N_TEST)
    print(f"analysing {len(tr_p)}+{len(te_p)} utterances, one per speaker ...", flush=True)
    t0 = time.time()
    train = [d for d in (prep(p, nf, nh) for p in tr_p) if d]
    test = [d for d in (prep(p, nf, nh) for p in te_p) if d]
    print(f"  train {len(train)} test {len(test)} in {time.time()-t0:.0f}s", flush=True)
    print(f"DSP core: test {np.mean([i['dsp'] for i in test]):.4f}", flush=True)

    G = CausalTCN(train[0]["cond"].shape[0], 3 * NB).to(DEV)
    nn.init.zeros_(G.out[-1].weight)
    nn.init.zeros_(G.out[-1].bias)
    D = SpecDisc().to(DEV)
    optG = torch.optim.AdamW(G.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=1e-4)
    optD = torch.optim.AdamW(D.parameters(), lr=2e-4, betas=(0.8, 0.99))
    schG = torch.optim.lr_scheduler.OneCycleLR(optG, 4e-4, total_steps=STEPS)
    gen = torch.Generator().manual_seed(0)

    def spectrum(out, mag, ph):
        d, cr, ci = out[:NB], out[NB:2 * NB], out[2 * NB:]
        nm = (mag + 1e-5) * torch.exp(d.clamp(-4, 4)) - 1e-5
        u = torch.complex(ph[0] + cr, ph[1] + ci)
        u = u / (u.abs() + 1e-6)
        return nm.clamp(min=0.0).to(torch.complex64) * u

    def evaluate(items, seeds=(101,)):
        G.eval()
        ss, s0 = [], []
        with torch.no_grad():
            for it in items:
                n = it["gt"].shape[-1]
                for sd in seeds:
                    mag, ph = realise(it["env"], n, nf, nh, sd)
                    T = mag.shape[-1]
                    out = G(it["cond"][:, :T][None].to(DEV))[0].cpu()
                    ss.append(score_one(it["gt"], it["h"] + R.istft(spectrum(out, mag, ph), n, nf, nh)))
                    S0 = mag.to(torch.complex64) * torch.complex(ph[0], ph[1])
                    s0.append(score_one(it["gt"], it["h"] + R.istft(S0, n, nf, nh)))
        G.train()
        return float(np.mean(ss)), float(np.mean(s0))

    def sample():
        it = train[int(torch.randint(len(train), (1,), generator=gen))]
        sd = int(torch.randint(10 ** 6, (1,), generator=gen))
        a0 = int(torch.randint(MARGIN, max(MARGIN + 1, it["T"] - CROP - MARGIN),
                               (1,), generator=gen))
        lo, hi = a0 - MARGIN, min(a0 + CROP + MARGIN, it["T"])
        if hi - lo < 2 * MARGIN + 8:
            return None
        s0 = lo * nh
        ncrop = (hi - lo - 1) * nh
        if s0 + ncrop > it["gt"].shape[-1]:
            return None
        g = torch.Generator().manual_seed(sd)
        pf = R._filtered_noise(it["env"][:, lo:hi], torch.randn(ncrop, generator=g),
                               ncrop, nf, nh)
        S = R.stft(pf, nf, nh)[:, : hi - lo]
        u = S / (S.abs() + R.EPS)
        out = G(it["cond"][:, :hi][None].to(DEV))[0][:, lo:hi]
        Sy = spectrum(out, S.abs().float().to(DEV),
                      torch.stack([u.real, u.imag]).float().to(DEV))
        nz = torch.istft(Sy, nf, nh, nf, torch.hann_window(nf, device=DEV),
                         center=True, length=ncrop)
        m0, m1 = MARGIN * nh, ncrop - MARGIN * nh
        return ((it["h"][s0:s0 + ncrop].to(DEV) + nz)[m0:m1],
                it["gt"][s0:s0 + ncrop].to(DEV)[m0:m1])

    a, b = evaluate(test[:EVAL_N])
    print(f"  step {0:5d}  TEST net {a:.4f}  no-net {b:.4f}  (zero-init)", flush=True)
    t0, best = time.time(), (-1.0, 0)
    for step in range(1, STEPS + 1):
        s = sample()
        if s is None:
            continue
        fake, real = s
        if step > D_START:
            with torch.no_grad():
                fk = fake.detach()
            dr, _ = D(real)
            df, _ = D(fk)
            lossD = sum(torch.relu(1 - o).mean() for o in dr) + \
                    sum(torch.relu(1 + o).mean() for o in df)
            optD.zero_grad(set_to_none=True)
            lossD.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            optD.step()

        lossG = mrstft(fake, real)
        if step > D_START:
            df, ff = D(fake)
            _, fr = D(real)
            adv = sum((-o).mean() for o in df) / len(df)
            fm = sum(torch.nn.functional.l1_loss(a_, b_.detach())
                     for a_, b_ in zip(ff, fr)) / len(ff)
            lossG = lossG + ADV_W * adv + FM_W * fm
        optG.zero_grad(set_to_none=True)
        lossG.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        optG.step()
        schG.step()

        if step % 2000 == 0 or step == STEPS:
            te, te0 = evaluate(test[:EVAL_N])
            if te > best[0]:
                best = (te, step)
            print(f"  step {step:5d}  G {float(lossG):.3f}  TEST {te:.4f} "
                  f"(no-net {te0:.4f})  ({time.time()-t0:.0f}s)", flush=True)

    out = Path("/home/kojirotanaka/kjranyone/LightVC/results/rddsp_gan")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"G": G.state_dict()}, out / "gan.pt")
    te, te0 = evaluate(test)
    print(f"\nDSP core on all test  {np.mean([i['dsp'] for i in test]):.4f}")
    print(f"same, unseen seed     {te0:.4f}")
    print(f"species D + GAN TEST  {te:.4f}   (best during training {best[0]:.4f} @ {best[1]})")


if __name__ == "__main__":
    main()
