"""Species D: DSP skeleton + neural spectrum, trained through the renderer.

The ladder that got here, all measured on the same three utterances:

  DSP core, honest interface                       3.439
  + residual net on log-magnitude L1               3.317  (L1 down 3.4x, PESQ DOWN)
  + residual net through the renderer, MR-STFT     3.501  (first legitimate gain)
  oracle-PHASE ceiling of the DSP core             3.851

The last line is the point. The breath branch has no phase model at all -- it
imposes a magnitude on a filtered-noise phase -- so no amount of magnitude
correction can pass 3.851, and the measured gap between 3.501 and that ceiling
is almost entirely phase. So let the net generate the phase too.

That is exactly what the design doc calls species D: the DSP skeleton keeps the
harmonic branch (glottal-phase harmonics below MVF, one node per period, 0.07x
of the sample rate) and the net supplies the breath branch's complex spectrum.
Conditioning stays mel + f0 + voicing = 0.31x, so the fine structure is
GENERATED from a low-rate description rather than transported from gt -- the
distinction that invalidated PESQ 4.070 and 4.024.

Zero-initialised on the magnitude head, and the phase head starts from the
renderer's own phase, so step 0 is still the DSP core exactly.
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


def main() -> None:
    nf, nh = R.NOISE_NFFT, R.NOISE_NFFT // R.NOISE_HOPDIV
    NB = nf // 2 + 1
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
        # the renderer's own phase field: what step 0 must reproduce
        # BASE = what the honest DSP breath branch actually produced, analysed
        # back. Its magnitude is the cepstrally truncated envelope the declared
        # interface carries, NOT the oracle full-resolution |STFT(x-h)| --
        # handing that to an istft is the magnitude-imposition leak that
        # invalidated 4.070 and 4.024, and it started this gate at 3.72 for
        # exactly that reason. Step 0 now reproduces the FIR render bit for bit.
        pf = R._filtered_noise(base, src, gt.shape[-1], nf, nh)
        S = R.stft(pf, nf, nh)[:, :T]
        mag = S.abs()
        S = S / (S.abs() + R.EPS)
        items.append(dict(uid=uid, gt=gt, h=h, base=mag.float(), cond=cond,
                          ph=torch.stack([S.real, S.imag]).float(),
                          dsp=score_one(gt, y), T=T))
        print(f"[{uid}] T {T}  DSP {items[-1]['dsp']:.4f}", flush=True)

    net = CausalTCN(items[0]["cond"].shape[0], 3 * NB).to(DEV)
    nn.init.zeros_(net.out[-1].weight)
    nn.init.zeros_(net.out[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-4, total_steps=STEPS)
    gen = torch.Generator().manual_seed(0)

    def spectrum(out, base, ph):
        """magnitude = base * exp(delta); phase = renderer phase rotated by the
        net's own unit vector, so a zero output leaves both untouched."""
        d, cr, ci = out[:NB], out[NB:2 * NB], out[2 * NB:]
        nm = (base + 1e-5) * torch.exp(d.clamp(-4, 4)) - 1e-5
        rot = torch.complex(1.0 + cr, ci)
        rot = rot / (rot.abs() + 1e-6)
        base_ph = torch.complex(ph[0], ph[1])
        return nm.clamp(min=0.0).to(torch.complex64) * (base_ph * rot)

    def evaluate():
        net.eval()
        ss = []
        with torch.no_grad():
            for it in items:
                out = net(it["cond"][None].to(DEV))[0].cpu()
                S = spectrum(out, it["base"], it["ph"])
                y = it["h"] + R.istft(S, it["gt"].shape[-1], nf, nh)
                ss.append(score_one(it["gt"], y))
        net.train()
        return float(np.mean(ss))

    print(f"  step {0:5d}  PESQ {evaluate():.4f}  (zero-init)", flush=True)
    t0, best = time.time(), (-1.0, None)
    for step in range(1, STEPS + 1):
        loss = 0.0
        for it in items:
            a = int(torch.randint(MARGIN, it["T"] - CROP - MARGIN, (1,), generator=gen))
            lo, hi = a - MARGIN, a + CROP + MARGIN
            out = net(it["cond"][:, :hi][None].to(DEV))[0][:, lo:hi]
            S = spectrum(out, it["base"][:, lo:hi].to(DEV), it["ph"][:, :, lo:hi].to(DEV))
            n = (hi - lo - 1) * nh
            s0 = lo * nh
            nz = torch.istft(S, nf, nh, nf, torch.hann_window(nf, device=DEV),
                             center=True, length=n)
            hh = it["h"][s0: s0 + n].to(DEV)
            tt = it["gt"][s0: s0 + n].to(DEV)
            m0, m1 = MARGIN * nh, n - MARGIN * nh
            loss = loss + mrstft((hh + nz)[m0:m1], tt[m0:m1])
        opt.zero_grad(set_to_none=True)
        (loss / len(items)).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 400 == 0 or step == STEPS:
            m = evaluate()
            if m > best[0]:
                best = (m, step)
            print(f"  step {step:5d}  MRSTFT {float(loss)/len(items):.4f}  "
                  f"PESQ {m:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # render the final state for the ear gate
    import soundfile as sf
    out = Path("/home/kojirotanaka/kjranyone/LightVC/results/rddsp_dspecies")
    out.mkdir(parents=True, exist_ok=True)
    net.eval()
    with torch.no_grad():
        for it in items:
            o = net(it["cond"][None].to(DEV))[0].cpu()
            S = spectrum(o, it["base"], it["ph"])
            y = it["h"] + R.istft(S, it["gt"].shape[-1], nf, nh)
            g = it["gt"].numpy()
            yy = y.numpy() * float(np.sqrt((g ** 2).mean() / ((y.numpy() ** 2).mean() + 1e-12)))
            pk = np.abs(yy).max()
            sf.write(out / f"{it['uid']}_gt.wav", g, R.SR)
            sf.write(out / f"{it['uid']}_dspeciesD.wav",
                     (yy * (0.95 / pk) if pk > 0.95 else yy).astype(np.float32), R.SR)
    torch.save({"net": net.state_dict(), "cond_ch": items[0]["cond"].shape[0],
                "nf": nf, "nh": nh}, out / "dspeciesD.pt")
    # PROBE: does the net USE the mel conditioning, or has it memorised 24 s of
    # audio into 5.5M parameters (five times more parameters than samples)?
    # Swap the conditioning between utterances and keep everything else.
    with torch.no_grad():
        for i, it in enumerate(items):
            other = items[(i + 1) % len(items)]
            L = min(it["cond"].shape[-1], other["cond"].shape[-1])
            o = net(other["cond"][:, :L][None].to(DEV))[0].cpu()
            S = spectrum(o, it["base"][:, :L], it["ph"][:, :, :L])
            y = it["h"] + R.istft(S, it["gt"].shape[-1], nf, nh)
            print(f"  [{it['uid']}] conditioning swapped with {other['uid'][-5:]}: "
                  f"PESQ {score_one(it['gt'], y):.4f}", flush=True)
    print(f"wrote {out}", flush=True)
    print(f"\nDSP core                        {float(np.mean([i['dsp'] for i in items])):.4f}")
    print(f"species D (mel -> breath spectrum) {best[0]:.4f} (step {best[1]})")
    print(f"oracle-phase ceiling of the DSP  3.8506")
    print(f"target                           4.0000")


if __name__ == "__main__":
    main()
