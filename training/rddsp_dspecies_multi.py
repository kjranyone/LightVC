"""Task A, first stage: does SCALE move the flat curve, or is it the objective?

Under the four-point protocol the neural breath branch was flat at 16 and at 54
utterances of ONE speaker. Two readings survive that:

  (a) not enough data -- the net memorises what it can and never has to learn a
      mapping;
  (b) the objective/architecture cannot express the mapping at any scale.

They separate on scale, and the cheapest decisive move is more SPEAKERS rather
than more utterances: 300 utterances from 300 different speakers makes
per-speaker memorisation useless while keeping the analysis cost bounded.

If the held-out curve is still flat here, (a) is refuted at 10x the data and 300x
the speakers, and Task A should be redirected from "more data" to "a different
objective" (adversarial / perceptual), which is what the full-corpus stage would
have to build anyway.

Protocol unchanged: fresh noise realisation every step, absolute phase
generation, disjoint train/test, evaluation on seeds never used in training,
output layer zero-initialised so step 0 renders the DSP core exactly.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import score_one
from rddsp_neural import CausalTCN, mel_of
from rddsp_resid_net import resample_rows
from rddsp_wave_net import mrstft, CROP, MARGIN, DEV
from rddsp_dspecies_honest import realise

ROOT = Path(__file__).resolve().parent.parent / "female-dataset"
N_TRAIN, N_TEST = 300, 40
SECONDS = 6
STEPS = 20000
EVAL_SEEDS = (101,)
EVAL_N = 16                      # test utterances scored at each checkpoint


def pick(n_train: int, n_test: int, seed: int = 0):
    spk = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    rng = random.Random(seed)
    rng.shuffle(spk)
    out = []
    for s in spk[: n_train + n_test]:
        w = sorted((ROOT / s).glob("*.wav"))
        if w:
            out.append(w[rng.randrange(len(w))])
    return out[:n_train], out[n_train: n_train + n_test]


def prep(path: Path, nf: int, nh: int):
    x, _ = librosa.load(str(path), sr=R.SR, mono=True)
    if len(x) < R.SR * 2:
        return None
    gt = torch.tensor(x[: R.SR * SECONDS])
    try:
        y, h, nz, p = R.resynthesize(gt)
    except Exception:
        return None
    env = p["noisefine"]
    T = env.shape[-1]
    f0 = p["f0"]
    if not bool((f0 > 50).any()):
        return None
    cond = torch.cat([resample_rows(mel_of(gt), T),
                      resample_rows(torch.log(R.fill_f0(f0)[None] + 1.0), T),
                      resample_rows((f0 > 50).float()[None], T)], dim=0).float()
    return dict(uid=path.stem, gt=gt, h=h, env=env.float(), cond=cond,
                dsp=score_one(gt, y), T=T)


def main() -> None:
    nf, nh = R.NOISE_NFFT, R.NOISE_NFFT // R.NOISE_HOPDIV
    NB = nf // 2 + 1
    tr_p, te_p = pick(N_TRAIN, N_TEST)
    print(f"analysing {len(tr_p)} train + {len(te_p)} test utterances, "
          f"one per speaker, {SECONDS}s each ...", flush=True)
    t0 = time.time()
    train = [d for d in (prep(p, nf, nh) for p in tr_p) if d]
    test = [d for d in (prep(p, nf, nh) for p in te_p) if d]
    print(f"  prepared train {len(train)} test {len(test)} in {time.time()-t0:.0f}s",
          flush=True)
    print(f"DSP core: train {np.mean([i['dsp'] for i in train]):.4f}  "
          f"test {np.mean([i['dsp'] for i in test]):.4f}", flush=True)

    net = CausalTCN(train[0]["cond"].shape[0], 3 * NB).to(DEV)
    nn.init.zeros_(net.out[-1].weight)
    nn.init.zeros_(net.out[-1].bias)
    nsamp = sum(i["gt"].shape[-1] for i in train)
    print(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M  "
          f"train samples {nsamp/1e6:.2f}M  speakers {len(train)}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-4, total_steps=STEPS)
    gen = torch.Generator().manual_seed(0)

    def spectrum(out, mag, ph):
        d, cr, ci = out[:NB], out[NB:2 * NB], out[2 * NB:]
        nm = (mag + 1e-5) * torch.exp(d.clamp(-4, 4)) - 1e-5
        u = torch.complex(ph[0] + cr, ph[1] + ci)
        u = u / (u.abs() + 1e-6)
        return nm.clamp(min=0.0).to(torch.complex64) * u

    def evaluate(items):
        net.eval()
        ss, s0 = [], []
        with torch.no_grad():
            for it in items:
                n = it["gt"].shape[-1]
                for sd in EVAL_SEEDS:
                    mag, ph = realise(it["env"], n, nf, nh, sd)
                    T = mag.shape[-1]
                    out = net(it["cond"][:, :T][None].to(DEV))[0].cpu()
                    ss.append(score_one(it["gt"], it["h"] + R.istft(spectrum(out, mag, ph), n, nf, nh)))
                    S0 = mag.to(torch.complex64) * torch.complex(ph[0], ph[1])
                    s0.append(score_one(it["gt"], it["h"] + R.istft(S0, n, nf, nh)))
        net.train()
        return float(np.mean(ss)), float(np.mean(s0))

    a, b = evaluate(test[:EVAL_N])
    print(f"  step {0:5d}  TEST net {a:.4f}  no-net {b:.4f}  (zero-init)", flush=True)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        loss = 0.0
        for _ in range(3):
            it = train[int(torch.randint(len(train), (1,), generator=gen))]
            sd = int(torch.randint(10 ** 6, (1,), generator=gen))
            a0 = int(torch.randint(MARGIN, max(MARGIN + 1, it["T"] - CROP - MARGIN),
                                   (1,), generator=gen))
            lo, hi = a0 - MARGIN, min(a0 + CROP + MARGIN, it["T"])
            if hi - lo < 2 * MARGIN + 8:
                continue
            g = torch.Generator().manual_seed(sd)
            s0 = lo * nh
            ncrop = (hi - lo - 1) * nh
            if s0 + ncrop > it["gt"].shape[-1]:
                continue
            pf = R._filtered_noise(it["env"][:, lo:hi], torch.randn(ncrop, generator=g),
                                   ncrop, nf, nh)
            S = R.stft(pf, nf, nh)[:, : hi - lo]
            u = S / (S.abs() + R.EPS)
            mag = S.abs().float().to(DEV)
            ph = torch.stack([u.real, u.imag]).float().to(DEV)
            out = net(it["cond"][:, :hi][None].to(DEV))[0][:, lo:hi]
            nz = torch.istft(spectrum(out, mag, ph), nf, nh, nf,
                             torch.hann_window(nf, device=DEV), center=True, length=ncrop)
            m0, m1 = MARGIN * nh, ncrop - MARGIN * nh
            loss = loss + mrstft((it["h"][s0:s0 + ncrop].to(DEV) + nz)[m0:m1],
                                 it["gt"][s0:s0 + ncrop].to(DEV)[m0:m1])
        if not torch.is_tensor(loss):
            continue
        opt.zero_grad(set_to_none=True)
        (loss / 3).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 2000 == 0 or step == STEPS:
            te, te0 = evaluate(test[:EVAL_N])
            print(f"  step {step:5d}  MRSTFT {float(loss)/3:.4f}  "
                  f"TEST {te:.4f} (no-net {te0:.4f})  ({time.time()-t0:.0f}s)", flush=True)

    out = Path("/home/kojirotanaka/kjranyone/LightVC/results/rddsp_multi")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict()}, out / "multi.pt")
    te, te0 = evaluate(test)
    print(f"\nDSP core on all test  {np.mean([i['dsp'] for i in test]):.4f}")
    print(f"same, unseen seed     {te0:.4f}")
    print(f"species D on TEST     {te:.4f}")


if __name__ == "__main__":
    main()
