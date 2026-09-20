"""Species D under the four-point checklist, on the RE-TUNED DSP core.

The earlier scaled run sat on knobs fitted to three easy utterances; the core has
since been re-tuned on twenty and gained +0.275 on held-out speech, so the
residual the net has to model is a different signal. Same protocol: fresh noise
every step, absolute phase, disjoint train/test, unseen evaluation seeds.

Original header:

Species D under the four-point checklist, SCALED (54 train / 12 test).

The 16/6 run was flat, but it had seen about four minutes of audio -- calling
that a verdict on the method would be the exact error this project already has a
rule about. This is the same protocol with 3.4x the data and 4x the steps.

Original header follows.

Species D under the four-point checklist the third degeneracy produced.

What was wrong with the 4.174 gate, verified by re-running it myself:
  * the noise realisation was FIXED (seed 1), so the net memorised that one draw
    -- re-seeding cost 4.174 -> 3.14, WORSE than not running the net at all;
  * the phase head could only ROTATE that fixed random field, so the only way it
    could ever help was by memorising it;
  * train == eval, and the output interface was an unconstrained complex
    spectrogram (1026 reals/frame = 3.01x the waveform), inside which gt-h is
    reconstructible to 44 dB with no net at all.

This run fixes the experiment rather than the score:
  1. a FRESH noise realisation every step, so memorising a draw is impossible;
  2. the phase is generated ABSOLUTELY -- normalize(base_phase + (cr + j ci)) --
     because rotating a field the net cannot see is not a phase model. Zero
     output still reproduces the base phase exactly, so step 0 is the DSP core;
  3. train and test utterances disjoint, and both disjoint from the three the
     DSP knobs were tuned on;
  4. evaluation uses seeds never seen in training.

Whatever number comes out of this is about a mapping, not a memory.
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
from rddsp_wave_net import mrstft, CROP, MARGIN, DEV

STEPS = 30000
N_TRAIN, N_TEST = 54, 12
EVAL_SEEDS = (101, 202)          # never used in training


def prep(uid: str, nf: int, nh: int) -> dict:
    x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=R.SR, mono=True)
    gt = torch.tensor(x[: R.SR * 8])
    y, h, nz, p = R.resynthesize(gt)
    env = p["noisefine"]
    T = env.shape[-1]
    f0 = p["f0"]
    cond = torch.cat([resample_rows(mel_of(gt), T),
                      resample_rows(torch.log(R.fill_f0(f0)[None] + 1.0), T),
                      resample_rows((f0 > 50).float()[None], T)], dim=0).float()
    return dict(uid=uid, gt=gt, h=h, env=env.float(), cond=cond,
                dsp=score_one(gt, y), T=T)


def realise(env: torch.Tensor, n: int, nf: int, nh: int, seed: int):
    """One draw of the DSP breath branch, analysed back: magnitude and phase."""
    g = torch.Generator().manual_seed(seed)
    pf = R._filtered_noise(env, torch.randn(n, generator=g), n, nf, nh)
    S = R.stft(pf, nf, nh)[:, : env.shape[-1]]
    u = S / (S.abs() + R.EPS)
    return S.abs().float(), torch.stack([u.real, u.imag]).float()


def main() -> None:
    nf, nh = R.NOISE_NFFT, R.NOISE_NFFT // R.NOISE_HOPDIV
    NB = nf // 2 + 1
    uids = [p.stem for p in sorted(CACHE.glob("*.npz"))]
    tr_ids, te_ids = uids[: N_TRAIN], uids[N_TRAIN: N_TRAIN + N_TEST]
    print(f"preparing {len(tr_ids) + len(te_ids)} utterances (DSP analysis)...", flush=True)
    train = [prep(u, nf, nh) for u in tr_ids]
    test = [prep(u, nf, nh) for u in te_ids]
    print(f"train {len(train)}  test {len(test)}  (disjoint; both disjoint from the "
          f"3 tuning utterances)", flush=True)
    print(f"DSP core: train {np.mean([i['dsp'] for i in train]):.4f}  "
          f"test {np.mean([i['dsp'] for i in test]):.4f}", flush=True)

    net = CausalTCN(train[0]["cond"].shape[0], 3 * NB).to(DEV)
    nn.init.zeros_(net.out[-1].weight)
    nn.init.zeros_(net.out[-1].bias)
    print(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M  "
          f"train samples {sum(i['gt'].shape[-1] for i in train)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-4, total_steps=STEPS)
    gen = torch.Generator().manual_seed(0)

    def spectrum(out, mag, ph):
        """magnitude = env * exp(d); phase = normalize(base + (cr + j ci)).

        ABSOLUTE, not a rotation: with a fresh noise draw every step the base
        phase is unpredictable, so a rotation of it carries no information and
        the only way to reduce the loss is to output a phase of one's own. Zero
        output leaves the base phase untouched, so step 0 is still the DSP core.
        """
        d, cr, ci = out[:NB], out[NB:2 * NB], out[2 * NB:]
        nm = (mag + 1e-5) * torch.exp(d.clamp(-4, 4)) - 1e-5
        u = torch.complex(ph[0] + cr, ph[1] + ci)
        u = u / (u.abs() + 1e-6)
        return nm.clamp(min=0.0).to(torch.complex64) * u

    def evaluate(items, seeds=EVAL_SEEDS):
        net.eval()
        ss, s0 = [], []
        with torch.no_grad():
            for it in items:
                n = it["gt"].shape[-1]
                for sd in seeds:
                    mag, ph = realise(it["env"], n, nf, nh, sd)
                    T = mag.shape[-1]
                    out = net(it["cond"][:, :T][None].to(DEV))[0].cpu()
                    S = spectrum(out, mag, ph)
                    ss.append(score_one(it["gt"], it["h"] + R.istft(S, n, nf, nh)))
                    S0 = mag.to(torch.complex64) * torch.complex(ph[0], ph[1])
                    s0.append(score_one(it["gt"], it["h"] + R.istft(S0, n, nf, nh)))
        net.train()
        return float(np.mean(ss)), float(np.mean(s0))

    a, b = evaluate(test)
    print(f"  step {0:5d}  TEST net {a:.4f}  no-net {b:.4f}  (zero-init)", flush=True)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        loss = 0.0
        for _ in range(3):
            it = train[int(torch.randint(len(train), (1,), generator=gen))]
            n = it["gt"].shape[-1]
            sd = int(torch.randint(10 ** 6, (1,), generator=gen))       # FRESH draw
            a0 = int(torch.randint(MARGIN, it["T"] - CROP - MARGIN, (1,), generator=gen))
            lo, hi = a0 - MARGIN, a0 + CROP + MARGIN
            g = torch.Generator().manual_seed(sd)
            s0 = lo * nh
            ncrop = (hi - lo - 1) * nh
            src = torch.randn(ncrop, generator=g)
            pf = R._filtered_noise(it["env"][:, lo:hi], src, ncrop, nf, nh)
            S = R.stft(pf, nf, nh)[:, : hi - lo]
            u = S / (S.abs() + R.EPS)
            mag = S.abs().float().to(DEV)
            ph = torch.stack([u.real, u.imag]).float().to(DEV)
            out = net(it["cond"][:, :hi][None].to(DEV))[0][:, lo:hi]
            Sy = spectrum(out, mag, ph)
            nz = torch.istft(Sy, nf, nh, nf, torch.hann_window(nf, device=DEV),
                             center=True, length=ncrop)
            m0, m1 = MARGIN * nh, ncrop - MARGIN * nh
            loss = loss + mrstft((it["h"][s0:s0 + ncrop].to(DEV) + nz)[m0:m1],
                                 it["gt"][s0:s0 + ncrop].to(DEV)[m0:m1])
        opt.zero_grad(set_to_none=True)
        (loss / 3).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 2500 == 0 or step == STEPS:
            tr, _ = evaluate(train[:4], seeds=(EVAL_SEEDS[0],))
            te, te0 = evaluate(test)
            print(f"  step {step:5d}  MRSTFT {float(loss)/3:.4f}  train {tr:.4f}  "
                  f"TEST {te:.4f} (no-net {te0:.4f})  ({time.time()-t0:.0f}s)", flush=True)

    out = Path("/home/kojirotanaka/kjranyone/LightVC/results/rddsp_dspecies_honest")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict()}, out / "honest.pt")
    te, te0 = evaluate(test)
    print(f"\nDSP core on test        {np.mean([i['dsp'] for i in test]):.4f}")
    print(f"same, unseen seeds      {te0:.4f}")
    print(f"species D on TEST       {te:.4f}")


if __name__ == "__main__":
    main()
