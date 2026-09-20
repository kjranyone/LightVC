"""H-F (docs/paper_benchmark_degeneracy.md 12.4): is the rddsp core a better
harmonic prior than f0 sinusoids?

The frontier derivation says the deficit is the MAP, not the rate: BigVGAN
reaches 4.228 in this harness from mel at 0.31x, while the hand-designed DSP map
caps at 3.851 with oracle phase at 0.5x. The SOTA shape for low latency is
Wavehax -- aliasing-free complex-spectrogram estimation in the TF domain,
conditioned on a harmonic prior, 0.623M parameters. Its prior is

    e[n] = sum_k sqrt(0.02/K) sin(2 pi k phi[n] + k psi_rand)  +  0.01 z[n]

i.e. f0 and nothing else; the per-harmonic phase is explicitly random. This
project has a prior that carries measured glottal-relative phase (glottal cycle
correlation 0.987), an MVF split and aperiodicity, at 0.07x of the sample rate.
No published vocoder uses that.

Two arms, identical network, identical training, identical held-out speakers,
ONLY the prior differs:

    arm A   Wavehax-style f0 sinusoids with random phase
    arm B   the rddsp harmonic branch h

This is a legitimate setup under 2.8: conditioning is mel + prior (both low
rate and declared), the output spectrogram is unconstrained, evaluation is on
speakers never trained on, and the prior contains no fixed random field that
could be memorised -- arm A's phase offsets are redrawn every step.
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
from rddsp_loop import score_one
from rddsp_neural import mel_of, N_MEL
from rddsp_dspecies_multi import pick

NFFT, HOP = 512, 128
NBIN = NFFT // 2 + 1
SECONDS = 5
STEPS = 1500
BATCH = 6
PRIOR_NOISE = 0.05
NONCAUSAL = False
CROP = 96                      # frames per training crop
# CUDA was missing from this chain, and this machine has no XPU -- so every run
# in the H-F series trained on six CPU threads while an idle RTX 2080 Ti sat
# next to it. That is what "the session's compute budget" meant in 12.9'''.
DEV = ("xpu" if torch.xpu.is_available() else
       "cuda" if torch.cuda.is_available() else "cpu")


class Wavehax2D(nn.Module):
    """Causal-in-time 2D ConvNeXt stack over (freq, time), TF domain only.

    No time-domain pointwise nonlinearity anywhere, so no aliasing can be
    generated -- the property Wavehax exists for, and the one this project
    independently diagnosed as the source of its own inter-harmonic noise.
    groups=1 throughout (XPU backward fails on depthwise).
    """

    def __init__(self, cin: int, ch: int = 48, layers: int = 8):
        super().__init__()
        self.inp = nn.Conv2d(cin, ch, 1)
        self.blocks = nn.ModuleList()
        # FREQUENCY DILATION. With kernel 7 and no dilation the receptive field
        # after 8 layers is 56 of 257 bins -- too local for a spectral envelope,
        # which is correlated across the whole band. Doubling the dilation every
        # other layer reaches the full band (7 + 6*(1+2+4+8+16+...) ) while
        # keeping the parameter count of a 7-tap kernel.
        self.dil = [1, 2, 4, 8, 1, 2, 4, 8][:layers]
        for df in self.dil:
            self.blocks.append(nn.ModuleList([
                nn.Conv2d(ch, ch, (7, 3), dilation=(df, 1)),
                nn.GroupNorm(1, ch),
                nn.Conv2d(ch, 3 * ch, 1),
                nn.Conv2d(3 * ch, ch, 1),
            ]))
        self.out = nn.Conv2d(ch, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):                            # [B, C, F, T]
        h = self.inp(x)
        for (c, n, u, d), df in zip(self.blocks, self.dil):
            # CAUSAL by default (left-only time padding). NONCAUSAL=True pads
            # symmetrically, which is the ONLY change -- same parameters, same
            # data, same schedule -- so the difference isolates what the
            # low-latency constraint costs. Without this number, "we are at 2.9
            # and SOTA is 4.2" cannot be attributed between causality, capacity
            # and training budget.
            pt = (1, 1) if NONCAUSAL else (2, 0)
            y = torch.nn.functional.pad(h, (pt[0], pt[1], 3 * df, 3 * df))
            y = d(torch.nn.functional.gelu(u(n(c(y)))))
            h = h + y
        return self.out(h)


def stft(x):
    return torch.stft(x, NFFT, HOP, NFFT, torch.hann_window(NFFT, device=x.device),
                      center=True, return_complex=True)


def istft(S, n):
    return torch.istft(S, NFFT, HOP, NFFT, torch.hann_window(NFFT, device=S.device),
                       center=True, length=n)


def wavehax_prior(f0: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """Eq. 16 of Wavehax: f0 sinusoids, random per-harmonic phase, plus noise."""
    f0u = R.frame_upsample(R.fill_f0(f0).double(), n).clamp(min=0.0)
    phi = 2 * math.pi * torch.cumsum(f0u, 0) / R.SR
    fm = float(f0[f0 > 50].median()) if (f0 > 50).any() else 200.0
    K = max(1, int((R.SR / 2) / fm))
    e = torch.zeros(n, dtype=torch.float64)
    off = torch.rand(K, generator=gen, dtype=torch.float64) * 2 * math.pi
    a = math.sqrt(0.02 / K)
    for k in range(1, K + 1):
        e = e + a * torch.sin(k * phi + k * off[k - 1])
    return (e + 0.01 * torch.randn(n, generator=gen, dtype=torch.float64)).float()


def prep(path: Path):
    x, _ = librosa.load(str(path), sr=R.SR, mono=True)
    if len(x) < R.SR * 2:
        return None
    gt = torch.tensor(x[: R.SR * SECONDS])
    try:
        y, h, nz, p = R.resynthesize(gt)
    except Exception:
        return None
    if not bool((p["f0"] > 50).any()):
        return None
    S = stft(gt)
    T = S.shape[-1]
    mel = mel_of(gt)
    idx = (torch.arange(T) * (mel.shape[-1] - 1) / max(T - 1, 1)).long()
    return dict(gt=gt, h=h, y=y, f0=p["f0"], mel=mel[:, idx].float(), T=T,
                dsp=score_one(gt, y))


def features(it, prior_wave, dev):
    """[C, F, T]: mel broadcast over frequency + the prior's complex spectrum."""
    P = stft(prior_wave.to(dev))[:, : it["T"]]
    T = P.shape[-1]
    mel = it["mel"][:, :T].to(dev)
    m = torch.nn.functional.interpolate(mel[None, None], size=(NBIN, T),
                                        mode="bilinear", align_corners=False)[0]
    return torch.cat([m, P.real[None], P.imag[None],
                      torch.log(P.abs()[None] + 1e-5)], 0), P


class WavLMConvLoss(torch.nn.Module):
    """L_MOS of FINALLY (arXiv:2410.05920 Eq. for L_MOS):

        100 * ||phi(y) - phi(g(x))||_2^2  +  || |STFT(y)| - |STFT(g)| ||_1
        phi = WavLM CONVOLUTIONAL encoder (not the transformer output)

    The frontier update in 12.7: multi-resolution STFT alone cannot drive a
    branch that has to invent stochastic structure -- it is a magnitude match,
    and every magnitude-matched breath sounds equally right to it. A perceptual
    feature space penalises the parts that make speech sound like speech. Their
    ablation puts WavLM-conv above Wav2Vec2, EnCodec and CDPAM, and CLAUDE.md
    permits auxiliary models for supervision as long as inference does not
    depend on them.

    16 kHz is WavLM's rate; the resampler is differentiable so gradients reach
    the 44.1 kHz generator."""

    def __init__(self, dev):
        super().__init__()
        from transformers import WavLMModel
        import torchaudio
        m = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.fe = m.feature_extractor.to(dev).eval()
        for q in self.fe.parameters():
            q.requires_grad_(False)
        self.rs = torchaudio.transforms.Resample(R.SR, 16000).to(dev)

    def forward(self, y, t):
        a = self.fe(self.rs(y)[None])[0]
        b = self.fe(self.rs(t)[None])[0]
        n = min(a.shape[-1], b.shape[-1])
        return ((a[..., :n] - b[..., :n]) ** 2).mean()


def main() -> None:
    arm = sys.argv[1] if len(sys.argv) > 1 else "B"
    use_lmos = "--lmos" in sys.argv
    global NONCAUSAL
    NONCAUSAL = "--noncausal" in sys.argv
    tr_p, te_p = pick(80, 12)
    names = {"A": "Wavehax f0 sinusoids", "B": "rddsp harmonic core",
             "C": "rddsp FULL core"}
    print(f"arm {arm}  ({names[arm]})  analysing ...", flush=True)
    t0 = time.time()
    train = [d for d in (prep(p) for p in tr_p) if d]
    test = [d for d in (prep(p) for p in te_p) if d]
    print(f"  train {len(train)} test {len(test)} in {time.time()-t0:.0f}s   "
          f"DSP core on test {np.mean([i['dsp'] for i in test]):.4f}", flush=True)

    net = Wavehax2D(cin=4).to(DEV)
    print(f"  net {sum(p.numel() for p in net.parameters())/1e6:.3f}M params "
          f"nfft {NFFT} hop {HOP} (STFT/iSTFT {1000*NFFT/R.SR:.1f} ms; "
          f"the PRIOR is whole-utterance non-causal, so this is NOT the system latency)", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS)
    g = torch.Generator().manual_seed(0)

    def prior_of(it, gen):
        if arm == "A":
            e = wavehax_prior(it["f0"], it["gt"].shape[-1], gen)
            # GAIN MATCH. Wavehax normalises its prior to a pseudo-constant power
            # of 0.02 because its net predicts the whole spectrogram; here the net
            # predicts a RESIDUAL on top of the prior, so a prior at a tenth of
            # the speech level forces the net to supply a large gain before it can
            # supply any structure. Left unmatched, arm A diverged (loss 56.6 at
            # step 500) and the A/B comparison measured amplitude, not
            # information. Match it to arm B's own level.
            return e * (float(it["h"].std()) / max(float(e.std()), 1e-8))
        # Wavehax's prior carries 0.01*z[n]. Arm B had NO stochastic term at all,
        # so the net had to invent every unvoiced sample out of a deterministic
        # input -- a TF-domain net cannot create noise from silence. Give it the
        # same seed of randomness, scaled to the harmonic branch's own level.
        if arm == "C":
            # arm C: the FULL DSP core (harmonic + breath), not just harmonics.
            # 12.8 set this aside because it measures the core rather than the
            # prior, which is right for the A/B question -- but wrong when the
            # goal is the best system. The net then refines a 2.97 starting point
            # instead of inventing the stochastic component from nothing, which
            # is what capped arms A and B near 1.5.
            return it["y"] + torch.randn(it["y"].shape[-1], generator=gen) * \
                   float(it["y"].std()) * PRIOR_NOISE * 0.2
        h = it["h"]
        z = torch.randn(h.shape[-1], generator=gen) * float(h.std()) * PRIOR_NOISE
        return h + z

    lmos = WavLMConvLoss(DEV) if use_lmos else None
    if lmos is not None:
        print("  L_MOS: WavLM-base-plus conv encoder (FINALLY), weight 100", flush=True)

    def mrstft(y, t):
        loss = 0.0
        for nf, hp in ((256, 64), (512, 128), (1024, 256)):
            w = torch.hann_window(nf, device=y.device)
            Y = torch.stft(y, nf, hp, nf, w, center=True, return_complex=True).abs()
            T_ = torch.stft(t, nf, hp, nf, w, center=True, return_complex=True).abs()
            loss = loss + (Y - T_).norm() / (T_.norm() + 1e-8)
            loss = loss + torch.nn.functional.l1_loss(torch.log(Y + 1e-5),
                                                      torch.log(T_ + 1e-5))
        return loss / 3

    def evaluate(items):
        net.eval()
        ss, pr = [], []
        ge = torch.Generator().manual_seed(999)
        with torch.no_grad():
            for it in items:
                n = it["gt"].shape[-1]
                pw = prior_of(it, ge)
                f, P = features(it, pw, DEV)
                o = net(f[None])[0]
                S = torch.complex(P.real + o[0], P.imag + o[1])
                ss.append(score_one(it["gt"], istft(S, n).cpu()))
                pr.append(score_one(it["gt"], pw[:n]))
        net.train()
        return float(np.mean(ss)), float(np.mean(pr))

    a, b = evaluate(test[:12])
    print(f"  step {0:5d}  TEST {a:.4f}   prior alone {b:.4f}", flush=True)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        # BATCH. The diagnostic in 12.7 showed the net could not fit its own
        # TRAINING data (1.32 against a prior of 1.89) with an unstable loss --
        # an optimisation problem, not a generalisation one. Batch size 1 over a
        # 257x192 complex regression is the obvious cause: every step is one
        # crop of one utterance, so the gradient is almost pure noise.
        CTX = 24
        loss = 0.0
        for _ in range(BATCH):
            it = train[int(torch.randint(len(train), (1,), generator=g))]
            pw = prior_of(it, g)
            f, P = features(it, pw, DEV)
            T = P.shape[-1]
            if T <= CROP + CTX + 8:
                continue
            s = int(torch.randint(CTX + 4, T - CROP - 4, (1,), generator=g))
            o = net(f[None, :, :, s - CTX: s + CROP])[0][:, :, CTX:]
            Pc = P[:, s: s + CROP]
            S = torch.complex(Pc.real + o[0], Pc.imag + o[1])
            nseg = (CROP - 1) * HOP
            y = istft(S, nseg)
            tgt = it["gt"][s * HOP: s * HOP + nseg].to(DEV)
            ys, ts = y[HOP * 2: -HOP * 2], tgt[HOP * 2: -HOP * 2]
            l = mrstft(ys, ts)
            if lmos is not None:
                l = l + 100.0 * lmos(ys, ts)
            loss = loss + l / BATCH
        if not torch.is_tensor(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 500 == 0 or step == STEPS:
            # TRAIN as well as TEST. Without it, "the net makes things worse"
            # cannot be attributed: a net that also fails on its own training
            # data has an optimisation or capacity problem, not a generalisation
            # one, and the fixes are disjoint.
            te, pr = evaluate(test[:8])
            tr, trp = evaluate(train[:8])
            # Report each set against ITS OWN prior. Comparing the train score to
            # the test set's prior is apples to oranges -- different utterances,
            # different difficulty -- and it made the net look like it was
            # failing to fit its training data when the baseline was simply
            # measured on other audio.
            print(f"  step {step:5d}  loss {float(loss):.4f}  "
                  f"TRAIN {tr:.4f} (prior {trp:.4f}, {tr-trp:+.4f})  "
                  f"TEST {te:.4f} (prior {pr:.4f}, {te-pr:+.4f})  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    te, pr = evaluate(test)
    print(f"\narm {arm}: TEST {te:.4f}   prior alone {pr:.4f}   "
          f"DSP core {np.mean([i['dsp'] for i in test]):.4f}")


if __name__ == "__main__":
    main()
