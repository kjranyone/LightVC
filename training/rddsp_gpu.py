"""GPU trainer for the TF-domain residual refiner (the H-F net of rddsp_hf.py).

Why this file exists at all: rddsp_hf.py picks its device with

    DEV = "xpu" if torch.xpu.is_available() else "cpu"

and this machine has no XPU -- it has an RTX 2080 Ti that torch sees as CUDA.
Every run in the H-F series therefore trained on six CPU threads: the 1000-step
capacity sweep took 3123 s at ch 48 and 7304 s at ch 96, and the ch 160 arm was
killed at 68 minutes without reaching its first checkpoint. That is why the
sweep concluded "capacity does not help" from runs that had not converged: the
OneCycle schedule ended while TEST was still rising (+0.2239 at 500, +0.2358 at
1000). The budget, not the architecture, set that boundary.

Same network, same loss, same data, same crop protocol as rddsp_hf.py arm C --
only the execution changes:

  * device cuda, with the prepared corpus resident on the GPU
  * the BATCH loop replaced by one stacked forward of [B, 4, F, CTX+CROP]
    (it was a Python loop of B single-crop forwards)
  * prep() cached to disk, so the 190-536 s of librosa + R.resynthesize is paid
    once per corpus size instead of once per run
  * checkpoints saved, which the sweep did not do -- every trained net from the
    CPU series was discarded at process exit and cannot be probed now

The degeneracy guards from 2.8 are preserved unchanged, because they are what
makes any number here admissible:

  * the prior's stochastic term is redrawn from a fresh generator every step
    and again at every evaluation, so nothing can be memorised from it
  * evaluation speakers never appear in training (pick() splits by speaker)
  * conditioning is mel + prior only, both declared and both low rate
  * TRAIN and TEST are each reported against THEIR OWN prior
"""
from __future__ import annotations

import argparse
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
from rddsp_hf import Wavehax2D, NFFT, HOP, NBIN, SECONDS, CROP

CACHE_DIR = Path("/home/kojirotanaka/kjranyone/LightVC/results/rddsp_gpu")
CTX = 24
# The stochastic term added to the prior. It was introduced for arm B, whose
# prior is the HARMONIC branch alone: a TF-domain net cannot create noise out of
# a deterministic input, so without a seed of randomness it could not synthesise
# any unvoiced sound at all. Arm C's prior is the FULL DSP core, which already
# carries a breath branch -- and the added noise costs 0.266 PESQ on the test set
# (clean core 2.9726, prior as fed to the net 2.7069). The whole "+0.237 neural
# gain" of 12.9 is measured against that self-inflicted handicap, and the trained
# net's 2.885 never gets back to the 2.973 it would have had by doing nothing.
# Kept as a flag rather than deleted, because 6.4's four-point check requires
# that ANY random field in the prior be reseeded every step; setting it to zero
# removes the field entirely, which satisfies the check vacuously.
PRIOR_NOISE = 0.05
LOG_FLOOR_DB = None   # None = the original absolute eps=1e-5; else dB below peak


def stft(x, dev=None):
    return torch.stft(x, NFFT, HOP, NFFT, torch.hann_window(NFFT, device=x.device),
                      center=True, return_complex=True)


def istft(S, n):
    return torch.istft(S, NFFT, HOP, NFFT, torch.hann_window(NFFT, device=S.device),
                       center=True, length=n)


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
    T = stft(gt).shape[-1]
    mel = mel_of(gt)
    idx = (torch.arange(T) * (mel.shape[-1] - 1) / max(T - 1, 1)).long()
    # The phase track is kept so a DEPLOYABLE prior can be built. The core prior
    # y carries glottal phase MEASURED from gt, which is unavailable at
    # conversion time (12.5), so anything trained on it has no product path.
    # f0 IS available at conversion time; wrapping the accumulated phase keeps
    # float32 storage honest (the raw cumsum reaches ~2*pi*1750 rad over 5 s,
    # where float32 would cost 1e-3 rad and 0.03 rad at the 30th harmonic).
    n = gt.shape[-1]
    f0u = R.frame_upsample(R.fill_f0(p["f0"]).double(), n).clamp(min=0.0)
    phi = torch.remainder(2 * math.pi * torch.cumsum(f0u, 0) / R.SR, 2 * math.pi)
    return dict(gt=gt, y=y, mel=mel[:, idx].float(), T=T, dsp=score_one(gt, y),
                f0=p["f0"], phi=phi.float())


def build(n_train: int, n_test: int):
    """Corpus prep is deterministic given (n_train, n_test) -- cache it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{SECONDS}_{NFFT}_{HOP}_{R.ENV_RATE}_{R.LS_PERIODS}_{R.MVF_SCALE}_v2"
    f = CACHE_DIR / f"prep_{n_train}_{n_test}_{key}.pt"
    if f.exists():
        d = torch.load(f, weights_only=False)
        print(f"  cached corpus {f.name}: train {len(d[0])} test {len(d[1])}", flush=True)
        return d
    t0 = time.time()
    tr_p, te_p = pick(n_train, n_test)
    tr = [d for d in (prep(p) for p in tr_p) if d]
    te = [d for d in (prep(p) for p in te_p) if d]
    torch.save((tr, te), f)
    print(f"  prepared train {len(tr)} test {len(te)} in {time.time()-t0:.0f}s", flush=True)
    return tr, te


def mel_to_linear(dev, nbin: int | None = None):
    # ⚠ `nbin` を必須の概念にする（2.1 の作るもの #6）。既定の NBIN 決め打ちだと
    #   2.4a が合成 nfft を動かしたときに黙って形が食い違う。
    """Place each mel band at ITS OWN frequency on the linear bin axis.

    The previous code did

        interpolate(mel[None, None], size=(NBIN, T), mode="bilinear")

    which stretches the 80 mel bands LINEARLY over 257 linear-frequency bins.
    Mel bands are not linearly spaced, so this puts the wrong band at almost
    every bin -- measured against the true centres:

        bin 128 (11025 Hz)  <-  mel band 40 (2876 Hz)    error -8149 Hz
        bin 160 (13781 Hz)  <-  mel band 49 (4548 Hz)    error -9233 Hz

    The frequency receptive field of the stack is 7 taps at dilations up to 8,
    about +-50 bins = +-4.3 kHz, so it cannot bridge a 9 kHz displacement. The
    net was therefore unable to read the target magnitude at the frequency it
    was correcting -- which is exactly the quantity a magnitude loss needs.

    Interpolating linearly between the true mel centres costs one fixed
    [NBIN, N_MEL] matrix and keeps the declared 80-band interface unchanged."""
    import librosa
    mc = torch.tensor(librosa.mel_frequencies(n_mels=N_MEL + 2, fmin=0.0,
                                              fmax=R.SR / 2)[1:-1],
                      dtype=torch.float32)
    nb = NBIN if nbin is None else int(nbin)
    nf = (nb - 1) * 2
    bf = torch.fft.rfftfreq(nf, 1 / R.SR).clamp(float(mc[0]), float(mc[-1]))
    i1 = torch.searchsorted(mc, bf).clamp(1, N_MEL - 1)
    i0 = i1 - 1
    w = ((bf - mc[i0]) / (mc[i1] - mc[i0])).clamp(0, 1)
    W = torch.zeros(nb, N_MEL)
    r = torch.arange(nb)
    W[r, i0] = 1 - w
    W[r, i1] = w
    return W.to(dev)


def to_dev(items, dev):
    """Resident on the GPU: mel already broadcast to the full frequency axis, and
    the prior's clean spectrum. The noise the prior carries is NOT baked in --
    STFT is linear, so it is added per step as STFT(z), keeping the reseeding
    guarantee that makes the memorisation control in 5 meaningful."""
    out = []
    W = mel_to_linear(dev)
    for it in items:
        T = it["T"]
        mel = it["mel"][:, :T].to(dev)
        m = (W @ mel)[None]
        d = dict(gt=it["gt"].to(dev), y=it["y"].to(dev), m=m, T=T, dsp=it["dsp"])
        if "bv" in it:
            d["bv"] = it["bv"].to(dev)
        if "phi" in it:
            d["phi"] = it["phi"].to(dev)
            fv = it["f0"][it["f0"] > 50]
            d["f0med"] = float(fv.median()) if fv.numel() else 200.0
        out.append(d)
    return out


def wavehax_prior(it, gen, dev):
    """Eq. 16 of Wavehax: f0 sinusoids with RANDOM per-harmonic phase, plus
    noise. Deployable: f0 and mel are both available at conversion time, and
    nothing here is measured from the target waveform.

    The per-harmonic offsets are redrawn every call, so there is no fixed random
    field to memorise (6.4 point 2). Level is matched to the core prior so that
    a residual net does not have to supply gain before it can supply structure --
    leaving that unmatched is what made arm A diverge in 12.8."""
    phi, n = it["phi"], it["gt"].shape[-1]
    K = max(1, int((R.SR / 2) / it["f0med"]))
    off = torch.rand(K, generator=gen, device=dev) * 2 * math.pi
    amp = math.sqrt(0.02 / K)
    e = torch.zeros(n, device=dev)
    # Chunked over harmonics: the full [K, n] outer product is 324 MB at a male
    # f0 of 60 Hz, and six of those per training step will not fit beside the
    # corpus.
    for s0 in range(0, K, 32):
        kk = torch.arange(s0 + 1, min(s0 + 33, K + 1), device=dev,
                          dtype=torch.float32)[:, None]
        e = e + (amp * torch.sin(kk * phi[None] + kk * off[s0: s0 + 32, None])).sum(0)
    e = e + 0.01 * torch.randn(n, generator=gen, device=dev)
    return e * (float(it["y"].std()) / e.std().clamp(min=1e-8))


def prior_wave(it, gen, dev, pn: float = PRIOR_NOISE, mode: str = "core"):
    if mode == "bigvgan":
        return it["bv"]
    if mode == "wavehax":
        return wavehax_prior(it, gen, dev)
    if pn <= 0:
        return it["y"]
    z = torch.randn(it["y"].shape[-1], generator=gen, device=dev)
    return it["y"] + z * float(it["y"].std()) * pn * 0.2


def feats(it, pw):
    P = stft(pw)[:, : it["T"]]
    T = P.shape[-1]
    return torch.cat([it["m"][:, :, :T], P.real[None], P.imag[None],
                      torch.log(P.abs()[None] + 1e-5)], 0), P


class WavLMConvLoss(nn.Module):
    """L_MOS of FINALLY: 100 * ||phi(y) - phi(t)||^2, phi = WavLM conv encoder."""

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
        """**相対距離**にする（2026-08-12 修正）。

        旧版は生の `((a-b)**2).mean()` を返しており、実測で:
          - `phi(gt)` の std が 0.0103 と極小 ⇒ 損失は O(1e-4)。重み 100 でも
            寄与は `mrstft` の **0.6%**＝目的関数に事実上存在しなかった。
          - gt と gt×0.5（−6 dB）で **0.000000**＝音量差に完全に盲目
            （`feat_extract_norm="group"` の GroupNorm がスケールを除去する）。
          - **無音 0.000115 < 別発話 0.000202**＝「無音のほうが gt に近い」。
            L2 の幾何上、原点は 2 つの独立点間距離の半分に来るので、
            この項は**無音へ寄せる**方向に効いていた。
        参照側のエネルギーで割ると O(1) になり、重みを素直に決められる。
        **無音が有利になる幾何そのものは L2 である限り消えない**ので、
        この項を単独で強くしない（`mrstft` と同格までにする）。
        """
        a = self.fe(self.rs(y))
        b = self.fe(self.rs(t))
        n = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :n], b[..., :n]
        return ((a - b) ** 2).mean() / (b ** 2).mean().clamp(min=1e-12)


def mrstft(y, t):
    """Multi-resolution STFT, reduced PER ITEM.

    Naively stacking the batch and calling .norm() gives the ratio of batch
    Frobenius norms, not the mean of per-item ratios -- which silently weights
    crops by energy. Peer review measured the tilt at up to 8x across an 18 dB
    level spread within a batch, and a factor ~2 on the loss value itself. Since
    the log-L1 term and L_MOS both reduce by mean, that also moved the relative
    weight of mrstft against 100*L_MOS. Reduce over (freq, time) and average
    over the batch, which is what the per-crop CPU loop computed."""
    loss = 0.0
    for nf, hp in ((256, 64), (512, 128), (1024, 256)):
        w = torch.hann_window(nf, device=y.device)
        Y = torch.stft(y, nf, hp, nf, w, center=True, return_complex=True).abs()
        T_ = torch.stft(t, nf, hp, nf, w, center=True, return_complex=True).abs()
        num = (Y - T_).flatten(-2).norm(dim=-1)
        den = T_.flatten(-2).norm(dim=-1) + 1e-8
        loss = loss + (num / den).mean()
        if LOG_FLOOR_DB is None:
            loss = loss + torch.nn.functional.l1_loss(torch.log(Y + 1e-5),
                                                      torch.log(T_ + 1e-5))
        else:
            # RELATIVE log floor. With an ABSOLUTE eps of 1e-5 the derivative of
            # log(x+eps) is 1/(x+eps), so a bin 100 dB below the utterance peak
            # gets a gradient ~1e5 times larger than a bin at the peak. Measured
            # on the held-out set (rddsp_lossmass.py): bins below -100 dB are
            # 31% of the bins and carry 86% of the gradient, while everything
            # above -60 dB -- the entire audible range -- carries 0.2%. The loss
            # was optimising silence. Clamping relative to the target's own peak
            # confines the gradient to the range a listener can hear.
            ref = T_.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            fl = 10.0 ** (LOG_FLOOR_DB / 20.0)
            loss = loss + torch.nn.functional.l1_loss(
                torch.log((Y / ref).clamp(min=fl)),
                torch.log((T_ / ref).clamp(min=fl)))
    return loss / 3


def safe_score(ref, deg):
    """PESQ raises on a silent degraded signal, and --direct starts from silence
    because the output layer is zero-initialised. 1.0 is PESQ's own floor, so
    returning it is the honest reading of "no signal", not a sentinel."""
    if not torch.isfinite(deg).all() or float(deg.abs().max()) < 1e-6:
        return 1.0
    try:
        return score_one(ref, deg)
    except ValueError:
        return 1.0


@torch.no_grad()
def evaluate(net, items, dev, seed=999, pn=PRIOR_NOISE, mode="core",
             direct=False):
    net.eval()
    ge = torch.Generator(device=dev).manual_seed(seed)
    ss, pr = [], []
    for it in items:
        n = it["gt"].shape[-1]
        pw = prior_wave(it, ge, dev, pn, mode)
        f, P = feats(it, pw)
        o = net(f[None])[0]
        S = (torch.complex(o[0], o[1]) if direct else
             torch.complex(P.real + o[0], P.imag + o[1]))
        ss.append(safe_score(it["gt"].cpu(), istft(S, n).cpu()))
        pr.append(safe_score(it["gt"].cpu(), pw[:n].cpu()))
    net.train()
    # Per-utterance scores travel with the means. The paired sd of (net - prior)
    # over utterances is the only honest error bar available here, and it costs
    # nothing -- the previous code threw the vector away and left the report
    # quoting differences with no CI at all.
    return float(np.mean(ss)), float(np.mean(pr)), np.array(ss) - np.array(pr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--ntrain", type=int, default=80)
    ap.add_argument("--ntest", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lmos", type=float, default=100.0)
    ap.add_argument("--pnoise", type=float, default=PRIOR_NOISE)
    # "bigvgan" puts the reference vocoder's own output in the prior slot. It has
    # NO product path (122M, non-causal) and its absolute score is BigVGAN's, not
    # this net's -- only the DELTA over the prior is attributable here. It answers
    # one question the DSP-core prior cannot: does this refiner help a system that
    # is already strong, or is its gain specific to repairing a weak one?
    ap.add_argument("--prior", type=str, default="core",
                    choices=["core", "wavehax", "bigvgan"])
    # RESIDUAL vs DIRECT. With the core prior a residual is right: the prior is
    # already at 2.97 and the net only has to correct it, and zero-init makes
    # step 0 exactly the prior. With the wavehax prior it is wrong: f0 sinusoids
    # have the wrong spectrum everywhere, so a residual net must first CANCEL
    # the prior before it can build anything. Wavehax itself predicts the output
    # spectrogram directly and uses the prior only as an input feature.
    ap.add_argument("--direct", action="store_true")
    # OVERFIT GATE (CLAUDE.md). If a net cannot improve the very utterances it
    # is trained on, "there is nothing to learn" and "the optimiser cannot find
    # it" are not distinguishable, and the fixes are disjoint. Train and test on
    # the same K utterances.
    ap.add_argument("--overfit", type=int, default=0)
    # HOLD THE EVALUATION SET FIXED when the training set size changes. pick(N, 12)
    # takes speakers N..N+11 as the test split, so pick(300,12) and pick(80,12)
    # evaluate on DIFFERENT speakers -- and 12.14 is a whole section about what
    # happens when a table's rows come from different audio. Evaluate every
    # training size on pick(80,12)'s test speakers.
    ap.add_argument("--evalfrom", type=int, default=0)
    # WARM START for --direct. 12.19 measures the generation parameterisation as
    # having a nearly consistent gradient (SNR 0.98 vs 0.26 for correction), but
    # it starts from silence while the residual starts AT the prior (2.9726). The
    # two advantages are separable: spend the first K steps teaching the direct
    # net to copy the prior -- which is trivial, since the prior's spectrum is
    # literally input channels 1 and 2 -- and only then switch to the real loss.
    # After that it has both the prior's starting point and generation's gradient.
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--logfloor", type=float, default=None,
                    help="dB below the target peak to clamp the log term")
    # CONSISTENCY. The net emits a complex spectrogram directly; nothing forces
    # it to be the STFT of any waveform. Measured off-subspace distance
    # ||STFT(iSTFT(S)) - S||/||S||: convfix 0.184, lf60 0.255 (+39%). Overlap-add
    # smears an inconsistent prediction, which is a classic metallic timbre, and
    # the ear reported exactly that on lf60. Penalise it explicitly.
    ap.add_argument("--consist", type=float, default=0.0)
    # COMPLEX (phase-aware) term. mrstft compares .abs() only, so nothing in the
    # objective has ever penalised phase directly. 12.10 measures the two halves
    # of the remaining gap on this harness as almost equal -- perfect magnitude
    # with our phase reaches 3.714, perfect phase with our magnitude 3.685 -- and
    # Zbig is already at 3.5548, i.e. near the "fix one side" ceiling. Going past
    # it requires both. This adds the complex residual directly:
    #     || S - STFT(target) ||_1 / || STFT(target) ||_1
    ap.add_argument("--cplx", type=float, default=0.0)
    # ADVERSARIAL. Every published low-latency vocoder (Wavehax, BigVGAN, HiFi-GAN)
    # trains with a discriminator; this project has one measurement of it (9.1)
    # showing PESQ falls monotonically, which 12.7 then corrected: that is the
    # KNOWN behaviour of adversarial training under PESQ, not a failure. Today's
    # blind tests settle how to judge it -- the ear, not PESQ. Sub-band CQT
    # discriminator (training/cqt_disc.py): log-frequency, so it sees harmonic
    # structure the way the ear does. CLAUDE.md permits GAN for texture only, on
    # top of a settled generator, hence --dstart.
    ap.add_argument("--gan", type=float, default=0.0)
    ap.add_argument("--fm", type=float, default=2.0)
    ap.add_argument("--dstart", type=int, default=2000)
    # The CQT discriminator reflect-pads by its longest window (16384
    # samples at 44.1 kHz), so the crop has to exceed that: 96 frames is
    # 11648 samples after the edge trim and the pad fails.
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--tag", type=str, default="run")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    global LOG_FLOOR_DB
    LOG_FLOOR_DB = a.logfloor
    torch.manual_seed(a.seed)
    tr_raw, te_raw = build(a.ntrain, a.ntest)
    if a.evalfrom:
        _, te_raw = build(a.evalfrom, 12)
        print(f"  eval set held fixed: pick({a.evalfrom}, 12) test split "
              f"({len(te_raw)} speakers)", flush=True)
    train = to_dev(tr_raw, dev)
    test = to_dev(te_raw, dev)
    if a.overfit:
        train = train[: a.overfit]
        test = train
        print(f"  OVERFIT GATE: train == test == {len(train)} utterances", flush=True)
    print(f"  device {dev}  train {len(train)} test {len(test)}  "
          f"DSP core on test {np.mean([i['dsp'] for i in test]):.4f}", flush=True)

    net = Wavehax2D(cin=4, ch=a.ch, layers=a.layers).to(dev)
    if a.direct:
        # Zero-init exists so that step 0 IS the prior -- meaningful only for a
        # residual. In direct mode it means step 0 is SILENCE, and the net has
        # to climb from nothing before any structure can be scored; 500 steps in,
        # the output was still below the raw f0 prior. Restore the default init.
        net.out.reset_parameters()
    npar = sum(p.numel() for p in net.parameters())
    print(f"  ch {a.ch} layers {a.layers}  {npar/1e6:.3f}M params  "
          f"batch {a.batch} steps {a.steps} lr {a.lr} lmos {a.lmos} "
          f"pnoise {a.pnoise} prior {a.prior} direct {a.direct} "
          f"logfloor {a.logfloor} consist {a.consist} cplx {a.cplx} "
          f"gan {a.gan} seed {a.seed}",
          flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    g = torch.Generator(device=dev).manual_seed(a.seed + 1)
    gc = torch.Generator().manual_seed(a.seed + 2)
    lmos = WavLMConvLoss(dev) if a.lmos > 0 else None
    disc = dopt = None
    if a.gan > 0:
        from cqt_disc import MSSubBandCQTDisc
        disc = MSSubBandCQTDisc(sr=R.SR).to(dev)
        dopt = torch.optim.AdamW(disc.parameters(), lr=a.lr, betas=(0.8, 0.99),
                                 weight_decay=1e-4)
        print(f"  discriminator: MS sub-band CQT, "
              f"{sum(p.numel() for p in disc.parameters())/1e6:.2f}M, "
              f"gan {a.gan} fm {a.fm} from step {a.dstart}", flush=True)

    te0, pr0, _ = evaluate(net, test, dev, pn=a.pnoise, mode=a.prior, direct=a.direct)
    print(f"  step {0:5d}  TEST {te0:.4f}  prior {pr0:.4f}", flush=True)
    crop = a.crop
    nseg = (crop - 1) * HOP
    t0 = time.time()
    for step in range(1, a.steps + 1):
        F, Pc, TG = [], [], []
        for _ in range(a.batch):
            it = train[int(torch.randint(len(train), (1,), generator=gc))]
            f, P = feats(it, prior_wave(it, g, dev, a.pnoise, a.prior))
            T = P.shape[-1]
            if T <= crop + CTX + 8:
                continue
            s = int(torch.randint(CTX + 4, T - crop - 4, (1,), generator=gc))
            F.append(f[:, :, s - CTX: s + crop])
            Pc.append(P[:, s: s + crop])
            TG.append(it["gt"][s * HOP: s * HOP + nseg])
        if not F:
            continue
        # ONE stacked forward. The CPU version ran B separate forwards inside a
        # Python loop and accumulated the loss; identical mathematically, but it
        # left the device idle between crops.
        o = net(torch.stack(F))[:, :, :, CTX:]
        Pb = torch.stack(Pc)
        S = (torch.complex(o[:, 0], o[:, 1]) if a.direct else
             torch.complex(Pb.real + o[:, 0], Pb.imag + o[:, 1]))
        y = istft(S, nseg)
        tgt = torch.stack(TG)
        ys, ts = y[:, HOP * 2: -HOP * 2], tgt[:, HOP * 2: -HOP * 2]
        cplx = None
        if a.cplx > 0:
            with torch.no_grad():
                Tc = stft(tgt)
            m = min(Tc.shape[-1], S.shape[-1])
            cplx = ((S[..., :m] - Tc[..., :m]).abs().flatten(-2).sum(-1)
                    / Tc[..., :m].abs().flatten(-2).sum(-1).clamp(min=1e-8)).mean()
        cons = None
        if a.consist > 0:
            S2 = stft(y)
            m = min(S2.shape[-1], S.shape[-1])
            cons = ((S2[..., :m] - S[..., :m]).abs().pow(2).flatten(-2).sum(-1).sqrt()
                    / S[..., :m].abs().pow(2).flatten(-2).sum(-1).sqrt().clamp(min=1e-8)
                    ).mean()
        if step <= a.warm:
            # copy the prior; no perceptual term, it is not a perceptual task
            loss = torch.nn.functional.mse_loss(torch.view_as_real(S),
                                                torch.view_as_real(Pb))
        else:
            loss = mrstft(ys, ts)
            if lmos is not None:
                loss = loss + a.lmos * lmos(ys, ts)
            if cons is not None:
                loss = loss + a.consist * cons
            if cplx is not None:
                loss = loss + a.cplx * cplx
            if disc is not None and step > a.dstart:
                # ---- D step (LSGAN) on the detached generator output ----
                dopt.zero_grad(set_to_none=True)
                yr, yg, _, _ = disc(ts[:, None], ys.detach()[:, None])
                dl = sum(((r - 1) ** 2).mean() + (g ** 2).mean()
                         for r, g in zip(yr, yg)) / len(yr)
                dl.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                dopt.step()
                # ---- G side: adversarial + feature matching ----
                _, yg2, fr, fg = disc(ts[:, None], ys[:, None])
                adv = sum(((g - 1) ** 2).mean() for g in yg2) / len(yg2)
                fmv = sum(torch.nn.functional.l1_loss(b, a_.detach())
                          for A, B in zip(fr, fg) for a_, b in zip(A, B))
                fmv = fmv / max(sum(len(A) for A in fr), 1)
                loss = loss + a.gan * adv + a.fm * fmv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % a.every == 0 or step == a.steps:
            te, pr, dte = evaluate(net, test[:8], dev, pn=a.pnoise, mode=a.prior, direct=a.direct)
            tr, trp, _ = evaluate(net, train[:8], dev, pn=a.pnoise, mode=a.prior, direct=a.direct)
            print(f"  step {step:5d}  loss {float(loss):.4f}  "
                  f"TRAIN {tr:.4f} ({tr-trp:+.4f})  TEST {te:.4f} ({te-pr:+.4f}"
                  f" +-{1.96*dte.std(ddof=1)/np.sqrt(len(dte)):.4f})  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            torch.save({"net": net.state_dict(), "args": vars(a), "step": step,
                        "test": te, "prior": pr},
                       CACHE_DIR / f"{a.tag}_ch{a.ch}_s{a.seed}.pt")

    te, pr, dte = evaluate(net, test, dev, pn=a.pnoise, mode=a.prior, direct=a.direct)
    ci = 1.96 * dte.std(ddof=1) / np.sqrt(len(dte))
    print(f"\n{a.tag} ch {a.ch} seed {a.seed}: TEST {te:.4f}  prior {pr:.4f}  "
          f"gain {te-pr:+.4f} +-{ci:.4f} (95% paired CI, n={len(dte)})  "
          f"({time.time()-t0:.0f}s)", flush=True)
    np.save(CACHE_DIR / f"{a.tag}_ch{a.ch}_s{a.seed}_gains.npy", dte)


if __name__ == "__main__":
    main()
