"""Species D with a train/test split: does it GENERALISE, or did it memorise?

The overfit gate reached 4.174 with 5.5M parameters over 1.06M samples, so the
net had five times more parameters than data and could in principle have stored
the answer. Swapping the conditioning between utterances costs 0.9-1.4 PESQ, so
it is at least reading the mel -- but a memorised mel->fine-structure lookup
behaves the same way. Only held-out speech separates the two.

Same architecture, same zero-init (step 0 renders the DSP core exactly), same
mel+f0+voicing conditioning, same loss through the renderer. The only change is
that the training set and the evaluation set are disjoint.
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

STEPS = 14000
N_TRAIN, N_TEST = 16, 6


def prep(uid: str, nf: int, nh: int) -> dict:
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
    pf = R._filtered_noise(base, src, gt.shape[-1], nf, nh)
    S = R.stft(pf, nf, nh)[:, :T]
    mag = S.abs()
    S = S / (S.abs() + R.EPS)
    return dict(uid=uid, gt=gt, h=h, base=mag.float(), cond=cond,
                ph=torch.stack([S.real, S.imag]).float(),
                dsp=score_one(gt, y), T=T)


def main() -> None:
    nf, nh = R.NOISE_NFFT, R.NOISE_NFFT // R.NOISE_HOPDIV
    NB = nf // 2 + 1
    uids = [p.stem for p in sorted(CACHE.glob("*.npz"))]
    tr_ids = uids[-(N_TRAIN + N_TEST + 3): -(N_TEST + 3)]
    te_ids = uids[-(N_TEST + 3): -3]
    print(f"train {len(tr_ids)}  test {len(te_ids)} (disjoint, and both disjoint "
          f"from the 3 utterances every knob was tuned on)", flush=True)
    train = [prep(u, nf, nh) for u in tr_ids]
    test = [prep(u, nf, nh) for u in te_ids]
    print(f"DSP core: train {np.mean([i['dsp'] for i in train]):.4f}  "
          f"test {np.mean([i['dsp'] for i in test]):.4f}", flush=True)

    net = CausalTCN(train[0]["cond"].shape[0], 3 * NB).to(DEV)
    nn.init.zeros_(net.out[-1].weight)
    nn.init.zeros_(net.out[-1].bias)
    nparam = sum(p.numel() for p in net.parameters())
    nsamp = sum(i["gt"].shape[-1] for i in train)
    print(f"params {nparam/1e6:.2f}M  train samples {nsamp/1e6:.2f}M  "
          f"ratio {nparam/nsamp:.2f}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-4, total_steps=STEPS)
    gen = torch.Generator().manual_seed(0)

    def spectrum(out, base, ph):
        d, cr, ci = out[:NB], out[NB:2 * NB], out[2 * NB:]
        nm = (base + 1e-5) * torch.exp(d.clamp(-4, 4)) - 1e-5
        rot = torch.complex(1.0 + cr, ci)
        rot = rot / (rot.abs() + 1e-6)
        return nm.clamp(min=0.0).to(torch.complex64) * (torch.complex(ph[0], ph[1]) * rot)

    def evaluate(items):
        net.eval()
        ss = []
        with torch.no_grad():
            for it in items:
                out = net(it["cond"][None].to(DEV))[0].cpu()
                S = spectrum(out, it["base"], it["ph"])
                ss.append(score_one(it["gt"], it["h"] + R.istft(S, it["gt"].shape[-1], nf, nh)))
        net.train()
        return float(np.mean(ss))

    print(f"  step {0:5d}  train {evaluate(train):.4f}  test {evaluate(test):.4f}  "
          f"(zero-init == DSP core)", flush=True)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        loss = 0.0
        for _ in range(3):
            it = train[int(torch.randint(len(train), (1,), generator=gen))]
            a = int(torch.randint(MARGIN, it["T"] - CROP - MARGIN, (1,), generator=gen))
            lo, hi = a - MARGIN, a + CROP + MARGIN
            out = net(it["cond"][:, :hi][None].to(DEV))[0][:, lo:hi]
            S = spectrum(out, it["base"][:, lo:hi].to(DEV), it["ph"][:, :, lo:hi].to(DEV))
            n = (hi - lo - 1) * nh
            s0 = lo * nh
            nz = torch.istft(S, nf, nh, nf, torch.hann_window(nf, device=DEV),
                             center=True, length=n)
            m0, m1 = MARGIN * nh, n - MARGIN * nh
            loss = loss + mrstft((it["h"][s0:s0 + n].to(DEV) + nz)[m0:m1],
                                 it["gt"][s0:s0 + n].to(DEV)[m0:m1])
        opt.zero_grad(set_to_none=True)
        (loss / 3).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 1000 == 0 or step == STEPS:
            tr, te = evaluate(train), evaluate(test)
            print(f"  step {step:5d}  MRSTFT {float(loss)/3:.4f}  "
                  f"train {tr:.4f}  TEST {te:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    out = Path("/home/kojirotanaka/kjranyone/LightVC/results/rddsp_dspecies_gen")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict()}, out / "gen.pt")
    import soundfile as sf
    net.eval()
    with torch.no_grad():
        for it in test:
            o = net(it["cond"][None].to(DEV))[0].cpu()
            S = spectrum(o, it["base"], it["ph"])
            y = (it["h"] + R.istft(S, it["gt"].shape[-1], nf, nh)).numpy()
            g = it["gt"].numpy()
            y = y * float(np.sqrt((g ** 2).mean() / ((y ** 2).mean() + 1e-12)))
            pk = np.abs(y).max()
            sf.write(out / f"{it['uid']}_gt.wav", g, R.SR)
            sf.write(out / f"{it['uid']}_heldout.wav",
                     (y * (0.95 / pk) if pk > 0.95 else y).astype(np.float32), R.SR)
    print(f"\nDSP core   train {np.mean([i['dsp'] for i in train]):.4f}  "
          f"test {np.mean([i['dsp'] for i in test]):.4f}")
    print(f"species D  train {evaluate(train):.4f}  TEST {evaluate(test):.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
