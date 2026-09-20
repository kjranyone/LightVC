"""Residual-DDSP synthesis core (survey-compliant).

Built to `current/hybrid_dsp_vocoder_survey.md` §10 rather than by hand:

  §10-2  filter = frequency-sampling FIR (always stable; no direct-form LPC,
         which diverges at pole radius >0.98 and has no XPU kernel).
  §10-3  phase = sine excitation supplies the reference, the net only ever adds
         a residual. Filter phase is selectable: min (causal, 0 extra latency),
         linear (diagnostic), or mixed (NHV-style complex cepstrum: negative
         quefrency = anticausal = glottal open phase, the part min-phase-only
         synthesis drops and hears as buzzy/robotic).
  §10-4  breath = MVF (one scalar per frame, harmonic below / noise above) plus
         band aperiodicity. Low MVF widens the noise side.
  §10-5  f0 = harmonic-sum (SHS/SWIPE'-style, prime harmonics only) so a weak
         fundamental is recovered from the harmonic series instead of being
         octave-flipped, and breathy frames are down-weighted by voicing.
  §10-7  no GPL lineage: SHS/SWIPE'/MVF/D4C are reimplemented from the papers,
         not lifted from Praat / COVAREP / c4dm-pyin.

Why MVF is load-bearing here (§5-4, §9-5): F0 error is only audible as harmonic
mis-placement *below* MVF. Lowering MVF shrinks the F0-dependent band, so the
weak-fundamental problem that has capped this project is absorbed structurally
rather than fought with a better detector.
"""
from __future__ import annotations

import math

import numpy as np
import torch

SR = 44100
NFFT = 2048
HOP = 256
NB = NFFT // 2 + 1
EPS = 1e-8
PRIMES = (1, 2, 3, 5, 7)

# Ablation switches. Everything added after the v2 render is gated here so each
# change can be scored against PESQ-vs-gt (the metric this project validated
# against the ear) instead of against LSD, which moved the wrong way.
MVF_SMOOTH = True      # median-9 + slew limit on the MVF contour
GCI_REFINE = False     # refractory epoch selection -- OFF, it HURTS.
# PESQ-vs-gt ablation (2026-07-29, 3 utterances): turning refinement off is
# +0.084, the largest single effect of anything added after v2. It was written
# to fix an audible crackle on the theory that 35.8% of raw ZFF epochs were
# spurious, but that statistic was measured against a global median period and
# counted unvoiced regions. The refractory removed 635 -> 473 epochs, and a
# MISSING epoch damages the phase track far more than an extra one: the harmonic
# offsets then interpolate across two real periods. Keep the raw epochs.
#   all on 2.753 | -GCI_REFINE 2.837 | -CYC_MEASURE 2.783 | -MVF_SMOOTH 2.761
#                | -UNVOICED_FULL 2.739 | -POWER_GATE 2.675 | ~v2 2.737
POWER_GATE = True      # sqrt (power-complementary) harmonic/noise crossfade.
# Glottal phase source. "measured" is the shipping path and is NOT deployable:
# the phase is read off the target waveform, which does not exist at conversion
# time (12.5, H-G). The other settings ask how much of it has to be measured:
#   utt    one pulse shape for the whole utterance (circular mean per harmonic)
#   random one fixed random offset per harmonic (Wavehax's prior)
#   zero   every harmonic in phase -- an impulse train, the textbook buzz
# If "utt" holds up, the phase is a slow, low-rate property and can be predicted
# from content; if it collapses toward "zero", it has to be tracked per frame.
GLOTTAL_PHASE = "measured"
# FLIPPED SIGN when the breath branch moved to its own 1024 grid: it was +0.078
# on the shared grid and is -0.026 now (3.353 without vs 3.327 with). The
# self-calibrating gain on the independent grid already matches the branch
# energy, so the sqrt double-counts. Every structural change in this loop has
# flipped at least one earlier winner -- nothing carries over untested.
UNVOICED_FULL = True   # unvoiced frames: noise carries the full |X|
CYC_MEASURE = True     # pick the cyclic/white mix per utterance
MVF_FLOOR = 2000.0     # lower bound on the harmonic/noise split, applied by
# RE-TUNED 2026-08-02 on TWENTY utterances. Every value in this file up to that
# point had been chosen on the same three, which turned out to be the easy ones
# (3.41 against 2.88 for the rest). Re-tuning moved the tuning set +0.209 and the
# HELD-OUT set +0.265 -- the held-out set gaining more is the signature of
# settings that were simply better, rather than fitted. Tune on the twenty.
# analyze(). What looked like a "MVF_SCALE 0.40 optimum" was this clamp: scaling
# the estimate by 0.40 drove almost every frame onto the 1 kHz floor, which is
# why 0.08 and 0.03 scored identically (2.9106). The estimator itself stays
# UNBIASED -- it has unit tests that check it recovers a known boundary, and
# biasing it inside broke them. Sweep the floor, not a scale.
MVF_SCALE = 0.3       # synthesis-side multiplier on the harmonic/noise split.
# Re-swept after the noise-grid change: 0.45 -> 3.362, 0.5 -> 3.351,
# 0.55 -> 3.353, 0.6 -> 3.353, 0.7 -> 3.303, 1.0 -> 3.262.
GLOBAL_GAIN = False    # SHIPPING DEFAULT: off. Matching the breath branch level
# with one scalar over the whole utterance is the last non-causal element --
# measured by perturbation, changing a parameter mid-utterance moved every
# output sample. Costs 0.025 PESQ (3.362 -> 3.337) to drop.
CAUSAL = False         # synthesis-side framing. False = centred, which is what
# ships: latency is the synthesis window half, nfft/2 = 11.6 ms, measured by
# perturbation at 9.93 ms. True places frame k at offset k*hop, which is NOT
# "causal" but a misalignment -- it puts the breath branch 11.6 ms behind the
# harmonic branch and cost 0.93 PESQ.
RTISI_LA = 2           # RTISI-LA look-ahead in frames (latency = LA*hop)
RTISI_IT = 5           # iterations inside the look-ahead buffer
NOISE_SMOOTH = 160       # REFUTED: smoothing the FIR noise target is monotonically
# worse (3.014 -> 2.970 / 2.765 / 2.457 / 2.227 at 160/80/40/20 coefficients),
# so the FIR gap is NOT "the target asks for structure a filter cannot make".
# Cause still unknown. cepstral coefficients kept in the FIR noise target
# (0 = off). A short min-phase FIR can only realise a SMOOTH envelope, but the
# target is |X|-derived and carries harmonic-fine structure the filter cannot
# produce -- asking for it is what caps the FIR path. In a correct DDSP split the
# fine structure belongs to the harmonic branch and the noise branch carries an
# envelope.
NOISE_PREFILTER = True   # SHIPPING DEFAULT. 3.337 -> 3.414 with no extra
# latency, no iteration and no whole-signal pass, which closes the gap to the
# offline Griffin-Lim reference (3.440) to 0.026. The FIR alone scores 3.04
# because a short filter can only make a smooth envelope; as an INITIALISER it
# is exactly right. FIR-filter the noise source before the magnitude
# projection, so the phase handed to the projection already belongs to a signal
# with roughly the target magnitude. The projection then has little
# inconsistency left to create, which is what the graininess is: frame-rate
# modulation from imposing a magnitude on a phase that never produced it. One
# extra single-pass filter, no iteration, no extra lookahead -- unlike Griffin-
# Lim, which fixes the same thing by iterating over the whole signal.
NOISE_MODE = "fir"   # "istft" imposes a magnitude on an arbitrary phase;
# "fir" actually FILTERS the noise signal, so |Y| = |N|*|H| holds by
# construction and there is no magnitude inconsistency to leak across frames.
NOISE_GL = 0           # SHIPPING DEFAULT: 0 = single pass.
# Griffin-Lim iterations buy quality (0: 3.337, 1: 3.386, 2: 3.414, 8: 3.423 at
# the same 11.6 ms window latency) but each is a whole-signal pass here. 1-2 are
# probably realisable with one extra frame of buffering; 8 is not. Latency is
# set by the synthesis window, nfft/2 = 11.6 ms, NOT by the iteration count --
# that confusion is what made this look unshippable for several rounds.
# Historical note, 8 CHOSEN BY EAR ("gl8 is clean"), over PESQ's preference for 4 (3.452 vs
# 3.423) -- the second time PESQ misranked this defect. The leak metric agreed
# with the ear (max overshoot 20.1 dB at 8 vs 23.9 at 4).
# NOT SHIPPABLE: eight whole-signal STFT round trips plus block latency.
# PESQ 3.362 -> 3.452 and the overshoot falls monotonically with iterations
# (mean 0.98 -> 0.68 / 0.65 / 0.55 / 0.53 dB at 0/1/2/4/8, max 33.3 -> 20.1).
# 4 is the PESQ optimum; 8 leaks least. Both are rendered for the listener to
# compare, because PESQ has already been wrong about this exact defect once.
# The excess energy in quiet frames is a PHASE problem: measured, the target
# noisemag matches gt to within 0-8 dB but the render overshoots the target by
# up to +11 dB, because overlap-add with a phase that is inconsistent across
# frames lets a loud frame bleed into its quiet neighbour. GL reduces exactly
# that inconsistency without touching the magnitudes, unlike NOISE_ITER which
# scaled the magnitude down and pumped.
NOISE_ITER = 0         # per-frame magnitude correction on the breath branch.
# REFUTED BY EAR, and this one matters: PESQ went UP (3.361 -> 3.393) and the
# level error at the flagged spot went +4.50 dB -> -2.00 dB, but the listener
# reported v10 (ITER 0) has LESS noise than v11 (ITER 3). Matching energy per
# frame against a phase-inconsistent render divides down a quiet frame's own
# content by the leakage that arrived from its loud neighbour -- the leak stays,
# the signal shrinks, and the result pumps. Correct on average, worse to listen
# to. PESQ is the best proxy this project has and it still missed this.
FLOOR_K = 0.25          # noise-floor min-pool kernel, in pitch periods
VOI_THR = 0.3         # voicing threshold in harmonic_sum_f0
HARM_AP = False        # per-harmonic aperiodicity from IHPC instead of the
# 3 kHz D4C bands. Below MVF (600-1000 Hz here) the band version is effectively a
# single scalar for the whole harmonic range, so no harmonic can be more or less
# periodic than its neighbours.
NOISE_NFFT = 1024      # breath branch rendered on its own grid (hop = nfft/NOISE_HOPDIV)
NOISE_HOPDIV = 3       # => 1024/171 = 6x overlap.
# Chosen against GRAININESS, not PESQ. Overlap-add inconsistency does not show
# up as broadband roughness -- measured that way the synthesis is SMOOTHER than
# the reference -- it shows up as a LINE in the envelope modulation spectrum at
# exactly SR/hop. At hop 256 that line sits at 172 Hz, dead in the roughness
# band, +1.48 dB above the reference. Six-fold overlap moves it to 259 Hz and
# pushes it 1.69 dB BELOW the reference, for 0.008 PESQ. Latency is unaffected:
# it is set by nfft/2, not by the hop.
#   4x: line 172 Hz +1.48 / PESQ 3.414   6x: 259 Hz -1.69 / 3.406
#   8x: line 345 Hz -1.28 / PESQ 3.404
# +0.368 PESQ, the single largest gain in the whole loop (2.959 -> 3.327).
# It is the WINDOW LENGTH, not the overlap: 1024 scores 3.284/3.306/3.327/3.321
# at 2x/3x/4x/8x, while 2048 scores 2.768/2.956 at 2x/4x and 512 and 256 are
# worse again (2.901 / 2.170). 2048 samples is 46 ms at 44.1 kHz -- longer than
# most speech events -- so the breath magnitude gets averaged across phonemes.
# 1024 is 23 ms, which is the right timescale. The harmonic branch keeps the
# long window because it needs the frequency resolution.
# The harmonic branch wants a long window (frequency resolution); the breath
# branch wants a short one (a 2048 window smears a fricative onset over 46 ms).
# 0 = share the main grid.
# PESQ is monotonic in this: 0.70 -> 2.963, 0.85 -> 2.936, 1.00 -> 2.921,
# 1.15 -> 2.900, 1.40 -> 2.864, 2.00 -> 2.796, and it keeps improving down to
# 0.40 (2.986) before TURNING OVER: 0.25 -> 2.918, 0.15 -> 2.912, 0.03 -> 2.911.
# That turnover matters -- it is the check that PESQ is not simply rewarding the
# degenerate all-noise solution the way LSD did (LSD preferred rendering the
# whole signal as noise, which is whispering). PESQ does not, so 0.40 is a real
# optimum rather than a slide toward degeneracy. The estimator is unbiased for
# "where do harmonics stop", but synthesis wants the split lower: a harmonic
# rendered from a slightly wrong amplitude/phase is worse than the same band
# rendered as shaped noise. Consistent with survey 5-4 -- this is a breathy
# voice and low MVF is its natural regime.
PULSE_MIX = 0.0        # amplitude of the glottal pulse train in the breath
# branch's source, against sqrt(1-mix^2) of white noise (power-complementary).
# Worth +0.004, and that is the finding: mixed excitation is the textbook fix for
# "no oscillator above MVF", and it does almost nothing here. Swept against
# envelope smoothness (where it should matter most, because a smooth envelope
# cannot make harmonic lines and the pulse must): ceps 30/45/60/90/160 gains
# +0.003/+0.006/+0.022/+0.031/+0.004. PESQ-wb resamples to 16 kHz and weights the
# low band, which the harmonic branch already owns -- replacing our output with
# gt ABOVE 2 kHz was worth +0.009 -- so harmonic structure up there buys almost
# nothing on this objective. It may still matter to the EAR; that is a separate
# gate and this number must not be used to close it.
PH_MODEL_HZ = 0        # above this frequency the harmonic phase comes from the
# amplitudes (minimum phase) instead of the measurement. 0 = measured everywhere.
NOISE_IRLEN = 0        # time-varying FIR length (0 = nfft//4). Longer realises
# a sharper spectral envelope but costs latency only through the STFT grid, since
# the filter is minimum phase.
GL_LA = 0              # Griffin-Lim lookahead in FRAMES (0 = whole-signal GL,
GL_CTX = 24            # which is not shippable). Latency = GL_LA*hop on top of
# the synthesis window, independent of NOISE_GL.
PHASE_SMOOTH = 1       # causal moving average, in NODES, applied to the unit
# phasor of each harmonic's complex envelope (its amplitude is left alone).
# Band attribution at the legitimate operating point puts the dominant remaining
# error in PHASE, not magnitude (+0.247 vs +0.170 for voiced frames over the full
# band), and the glottal offset measured at successive epochs jitters by 1.0-1.4
# rad against pi/2 = 1.571 for pure noise -- flat in k, so it is measurement
# noise rather than an f0 error. Injected into the phase track it is a random FM
# of tens of Hz on every harmonic at once.
NOISE_BANDS = 0        # reduce the breath-branch magnitude to this many mel
# bands before rendering (0 = keep full STFT resolution). This is the honesty
# check on the noise target: the network will predict a SPECTRAL ENVELOPE, not a
# 513-bin magnitude, so any gain that disappears under band reduction was fine
# structure copied from gt rather than something a vocoder can be given.
RESID_NOISE = False    # breath-branch target = |STFT(x - h)| (see resynthesize)
RESID_MIX = 0.7        # blend toward the min-pooled floor target
CAUSAL_ONLY = True     # remove every SYNTHESIS-side dependency on the future.
# The analysis half of this file stands in for the network that will predict
# these parameters, so its lookahead is the predictor's problem, not the
# vocoder's. The synthesis half is the vocoder, and it had four futures in it
# that the "11.6 ms" figure never counted: centred median filters on f0 and MVF
# (+-11.6 and +-23.2 ms), fill_f0 interpolating toward the NEXT voiced frame,
# the centred voicing ramp (+-VOI_RAMP frames), and choose_cyc sweeping the
# whole utterance. This flag replaces each with a causal equivalent so the
# quality can be quoted honestly.
F0_FILL = True         # continuous f0 through unvoiced gaps (see fill_f0).
VOI_RAMP = 3           # half-width, in frames, of the raised-cosine that opens
# the harmonic branch at a voiced onset. 0 = the hard sample-rate step that
# ships. A ramp costs VOI_RAMP frames of lookahead (5.8 ms each), so it is only
# worth taking if it pays.
ENV_LS = True          # least-squares harmonic amplitudes instead of the Hann
# demodulator (harmonic_ls). The demodulator needs ENV_PERIODS=4 -- 18 ms -- to
# separate neighbouring harmonics at one node per period, because it relies on
# the window's nulls. A joint solve does not, so it can use a short window and
# still separate them.
LS_PERIODS = 3.0       # least-squares window, in periods of median f0.
# NOT gated by rddsp_bankcheck any more -- that gate was wrong here. At one node
# per period a harmonic model spends 2*K*f0 = 2*MVF reals/s, which is the Nyquist
# DOF of the band it covers, INDEPENDENT of the window length. So a rate-1 bank
# is inherently near-critical inside its own band and the white-noise test flags
# it whatever the window; shortening the window buys time resolution without
# buying any parameters. The gate that actually binds is the RATE LEDGER, and
# there the harmonic branch costs 0.07x the sample rate while the breath branch
# was costing 1.88x. Window length was never the leak.
# LOCKED TO ENV_RATE the same way ENV_PERIODS is, and for the same reason: a
# window of P periods gives each harmonic channel a bandwidth of 2*f0/P, so at
# P <= 2 the channels TILE the band and no harmonic/noise decomposition happens
# at all -- the branch just returns the waveform. Measured with
# rddsp_bankcheck.py: white-noise reconstruction 8.45 dB at P=2 against 4.30 dB
# at P=4, and the short window scores 3.614 against 3.378 purely on that.
# Changing the estimator does NOT buy time resolution: least squares and the
# Hann demodulator agree to 0.002 PESQ at P=4. What least squares does buy is a
# 10x faster analysis, which is why it ships.
LS_RIDGE = 1e-6        # ridge, as a fraction of the mean diagonal of A'WA
LS_CHUNK = 192         # nodes solved per batch (memory: chunk*L*2K doubles)
ENV_DEMOD = True       # harmonic amplitudes from heterodyne demodulation against
# the SYNTHESIS phase track, instead of a windowed DFT at a per-GCI constant f0.
# Mathematically the same filter, but the basis follows the f0 contour through
# the window instead of freezing it, and the envelope is then sampled from a
# per-sample signal rather than reconstructed from one measurement per epoch.
ENV_RATE = 1           # envelope nodes per glottal period. THE INTERFACE.
# A harmonic's complex envelope is physically band-limited to +-f0/2: it is
# modulated by the vocal tract (well under 50 Hz) and by cycle-to-cycle jitter
# and shimmer, which by definition cannot exceed one cycle. One node per period
# is therefore exactly its Nyquist rate, and the branch spends 2*K*f0 = 2*MVF
# reals/s -- exactly the Nyquist DOF of the band [0, MVF] it covers.
#
# ENV_RATE=2 measures BETTER (3.725 vs 3.435 at MVF 3000, and raising MVF starts
# helping instead of hurting), and it is REFUTED anyway: at two nodes per period
# the envelope carries up to +-f0, and the only thing between +-f0/2 and +-f0 is
# the neighbouring harmonic's beat -- inter-harmonic content, which is not a
# harmonic amplitude. The rate ledger says the same thing: 0.27x of the sample
# rate for a band whose DOF is 0.14x, i.e. 2x overcomplete, i.e. able to
# transport that band rather than model it. This is the third form of the same
# degeneracy (see current/residual_ddsp.md 0''''.0b4); it survives the
# system-level probes (h=0 costs 2.09, detuning f0 costs 0.63) precisely because
# transporting through the carrier still needs the carrier.
# CORRECTED 2026-08-01. The earlier note here said the demodulator band-limits
# A_k to +-f0/2 and that the GCI rate is therefore exactly Nyquist. WRONG by a
# factor of two: a Hann of ENV_PERIODS periods has main-lobe half-width
# 2*f0/ENV_PERIODS, so the critical node rate is 4*f0/ENV_PERIODS and the GCI
# rate is HALF of it at ENV_PERIODS=2. Worse, the missing factor is exactly the
# span that contains the NEIGHBOURING HARMONIC, so raising ENV_RATE past the
# critical rate stops modelling harmonics and starts copying the waveform --
# see ENV_PERIODS below. The measured ladder is real; the explanation was not.
# 2 -> 3.930, 3 -> 3.961, 4 -> 3.912, 6 -> 3.927.
# Measured ladder (same noise branch, 3 utterances): GCI rate linear 3.657,
# GCI rate CUBIC 3.658, 2x rate linear 3.912, per-sample 3.976. Cubic buying
# nothing while doubling the rate buys +0.25 is the signature of undersampling:
# no interpolator can recover what the sampling threw away.
ENV_PERIODS = 4.0      # demodulator low-pass length, in periods of median f0.
# (ENV_PERIODS, ENV_RATE) = (4.0, 1) is the PARAMETER INTERFACE, not a tuning
# choice, and the two numbers are locked to each other by ENV_RATE >= 4/EP.
# Below that line the envelope is undersampled; ABOVE it the harmonic bank stops
# being a model at all. Sum_k W(f - k*f0) for a Hann of EP periods is EXACTLY
# constant whenever EP <= 2 (Poisson: the only surviving term is w[0], because
# w[+-SR/f0] falls outside a window shorter than two periods), so the "harmonic
# bank" becomes a perfect-reconstruction filter bank and h is the gt WAVEFORM
# below the gate. Measured: at EP=1.75 it reconstructs WHITE NOISE -- no
# harmonic structure whatever -- at 17.97 dB SNR, and detuning f0 by 31% costs
# only 0.48 PESQ, against 1.26 at EP=4.0. An earlier configuration of this file
# (EP=1.75, RATE=5) scored 4.070 that way. That number is WITHDRAWN: it measured
# how invertible the filter bank is, not how good the vocoder is.
ENV_POLAR = False      # interpolate |A| and arg(A) separately (the pre-2026-08
# behaviour) instead of interpolating the complex envelope. A complex envelope
# is what the demodulator produces and what is band-limited; splitting it into
# polar coordinates makes two signals that are not.
GATE_SOFT = 0.08       # width of the harmonic/noise crossover, as a fraction of
# MVF. It must be the SAME number in analyze() and in synthesize_v2() or the two
# branches stop being complementary and the band is double-counted or lost.
# The crossover is amplitude-complementary (POWER_GATE is refuted), so inside it
# the two uncorrelated branches sum to g^2 + (1-g)^2 of the power -- a hole
# reaching -3 dB at MVF. At 0.15 that hole is +-150 Hz wide around 1 kHz, and
# per-harmonic heterodyne measures exactly it: the 837 Hz harmonic comes out
# 2.1-2.8 dB low. Narrowing the crossover shrinks the hole without moving the
# split or giving the phase-blind branch more of the peak (which is what
# POWER_GATE did, and why it lost).
SUBF0_FULL = 0.0       # hand the breath branch the FULL |X| below this multiple
# of f0 (0 = off). See rddsp_fullband.py.
FLOOR_GAIN = 1.0       # level of the INTER-HARMONIC floor the breath branch adds
# BELOW MVF, where the harmonic bank already supplies the peaks. Band-wise oracle
# substitution (rddsp_attrib.py) says the entire remaining PESQ gap lives under
# 2 kHz -- replacing our output with gt ABOVE 2 kHz is worth +0.009, below 1 kHz
# +0.860 -- while the harmonic bank already cancels 24.4 dB of the 0-500 Hz band
# in voiced frames. So what is left down there is largely what we ADD: a
# min-pooled valley rendered with an arbitrary phase, on top of harmonics that
# were already right.
POWER_AP = False       # REFUTED (-0.023 PESQ). sqrt on the aperiodicity weight (same power-complementary
                       # argument as POWER_GATE: ap mixes harmonic and noise
                       # POWER, so the amplitude weight must be its square root)


def _win(n: int = NFFT) -> torch.Tensor:
    return torch.hann_window(n)


def stft(x: torch.Tensor, nfft: int = NFFT, hop: int = HOP) -> torch.Tensor:
    # ANALYSIS stays centred even when CAUSAL is set. In the product the analysis
    # is replaced by the frame network, so its framing is not a latency term;
    # making it causal here only misaligns the GCIs, f0 and harmonic branch
    # against each other and cost 0.95 PESQ when it was wired to the same flag.
    return torch.stft(x, nfft, hop, nfft, _win(nfft), center=True, return_complex=True)


def istft(S: torch.Tensor, n: int, nfft: int = NFFT, hop: int = HOP) -> torch.Tensor:
    if CAUSAL:
        y = ola_istft_causal(S, nfft, hop)
        return torch.nn.functional.pad(y, (0, max(0, n - y.shape[-1])))[:n]
    return torch.istft(S, nfft, hop, nfft, _win(nfft), center=True, length=n)


def ola_istft_causal(S: torch.Tensor, nfft: int, hop: int) -> torch.Tensor:
    """center=False overlap-add. Output sample t depends only on frames whose
    window STARTS at or before t, so the synthesis contributes no lookahead
    whatever nfft is -- the property this project already established on the
    neural side and then forgot when it wrote the DDSP renderer with
    center=True."""
    w = _win(nfft)
    fr = torch.fft.irfft(S, n=nfft, dim=0) * w[:, None]
    T = fr.shape[-1]
    L = (T - 1) * hop + nfft
    out = torch.nn.functional.fold(fr[None], (1, L), (1, nfft), stride=(1, hop))[0, 0, 0]
    wn = torch.nn.functional.fold((w ** 2)[:, None].expand(nfft, T)[None],
                                  (1, L), (1, nfft), stride=(1, hop))[0, 0, 0]
    # NO trim. Frame k's audio is placed at output offset k*hop, so parameters
    # for frame k can only affect samples from k*hop onward: lookahead is
    # exactly 0. Trimming to line the output up with a centre-framed reference
    # is an offline-comparison convenience and puts the latency straight back.
    return out / (wn + 1e-8)


def frame_upsample(v: torch.Tensor, n: int, hop: int = HOP) -> torch.Tensor:
    """Linear interpolation from frame rate to sample rate. [..., T] -> [..., n]."""
    t = torch.arange(n, dtype=torch.float64, device=v.device) / hop
    i = t.long().clamp(0, v.shape[-1] - 2)
    fr = (t - i).to(v.dtype)
    return v[..., i] * (1 - fr) + v[..., i + 1] * fr


# ---------------------------------------------------------------- f0 (§9-4)

def harmonic_sum_f0(x: torch.Tensor, fmin: float = 60.0, fmax: float = 600.0,
                    cents: float = 10.0, kmax: int = 20):
    """SWIPE'-style harmonic-sum F0 (§9-4). Two lobes per candidate:

      + at k*f0        (weight 1/sqrt(k))  -- reward explained harmonics
      - at (k+0.5)*f0                      -- PENALISE energy the candidate
                                              cannot explain

    Both lobes are required. Rewards alone pick the octave-too-high candidate
    (2*f0 scores the strong 2nd harmonic at full weight while the true f0 scores
    a missing fundamental at full weight), which is exactly the failure this
    voice triggers. The negative lobes kill it: at 2*f0 the odd harmonics land
    mid-lobe. The 1/sqrt(k) decay handles the other direction (f0/2 explains
    everything but only at high k, so it scores lower).

    Returns (f0 [T], voicing [T] in 0..1)."""
    mag = stft(x).abs()                                          # [NB, T]
    nb, T = mag.shape
    binhz = SR / NFFT
    ncand = int(math.log2(fmax / fmin) * 1200 / cents) + 1
    cand = fmin * 2 ** (torch.arange(ncand, dtype=torch.float32) * cents / 1200)

    def sample(f: torch.Tensor) -> torch.Tensor:
        ok = (f < SR / 2 - binhz) & (f > 0)
        b = (f / binhz).clamp(0, nb - 2)
        lo = b.long()
        fr = (b - lo)[:, None]
        a = mag[lo] * (1 - fr) + mag[lo + 1] * fr
        return a * ok[:, None].float()

    score = torch.zeros(ncand, T)
    for k in range(1, kmax + 1):
        w = 1.0 / math.sqrt(k)
        score += w * sample(cand * k)
        score -= 0.5 * w * sample(cand * (k + 0.5))
    idx = score.argmax(0)
    f0 = cand[idx]
    peak = score.gather(0, idx[None]).squeeze(0)
    voi = (peak / (mag.sum(0) / math.sqrt(nb) + EPS)).clamp(min=0)
    voi = (voi / (voi.median() + EPS) * 0.5).clamp(0, 1)
    f0 = _median_filter(f0, 5)
    f0 = torch.where(voi > VOI_THR, f0, torch.zeros_like(f0))
    return f0, voi


def _median_filter(v: torch.Tensor, k: int) -> torch.Tensor:
    if CAUSAL_ONLY:
        # Same window LENGTH, all of it in the past. A centred median of 9 frames
        # is +-23.2 ms of lookahead, which is twice the whole synthesis latency
        # budget and was never counted in it.
        pad = torch.nn.functional.pad(v[None, None], (k - 1, 0), mode="replicate")[0, 0]
        return pad.unfold(0, k, 1).median(-1).values
    p = k // 2
    return torch.nn.functional.pad(v[None, None], (p, p), mode="replicate")[0, 0].unfold(0, k, 1).median(-1).values


def median_f0(f0: torch.Tensor) -> float:
    """The scalar f0 that sets the cyclic-noise burst length and the demodulator
    window. Whole-utterance normally; in CAUSAL_ONLY it is a startup calibration
    over the first second, which a streaming implementation can actually do."""
    v = f0[f0 > 50]
    if v.numel() == 0:
        return 200.0
    if CAUSAL_ONLY:
        cal = f0[: int(SR / HOP)]
        cal = cal[cal > 50]
        if cal.numel() >= 8:
            return float(cal.median())
    return float(v.median())


# ---------------------------------------------------------------- MVF (§5-2)

def harmonic_analysis(x: torch.Tensor, f0: torch.Tensor, kmax: int = 400,
                      periods: int = 4, hop: int = HOP):
    """Exact complex amplitude of every harmonic, evaluated AT k*f0 rather than
    at an FFT bin (§5-2 peak-picking done properly).

    Reading an FFT bin costs up to 1.4 dB of scalloping loss and, worse, gives a
    phase that belongs to the bin frequency, not to k*f0 -- which destroys the
    inter-harmonic phase coherence that MVF is measured from. A direct DFT at
    the harmonic frequency has neither problem.

    Returns X [kmax, T] complex, already scaled so |X_k| is the amplitude of
    that harmonic."""
    T = f0.shape[-1]
    X = torch.zeros(kmax, T, dtype=torch.complex64)
    xp = torch.nn.functional.pad(x, (NFFT, NFFT))
    ks = torch.arange(1, kmax + 1, dtype=torch.float64)[:, None]
    for t in range(T):
        f = float(f0[t])
        if f <= 50.0:
            continue
        L = min(int(round(periods * SR / f)) | 1, NFFT)
        c = t * hop + NFFT
        seg = xp[c - L // 2: c - L // 2 + L].double()
        if seg.shape[-1] < L:
            break
        w = torch.hann_window(L, dtype=torch.float64)
        nn = torch.arange(L, dtype=torch.float64)[None, :] - L // 2
        kk = int(min(kmax, (SR / 2 - 1) // f))
        ph = -2 * math.pi * ks[:kk] * f * nn / SR
        e = torch.cos(ph) + 1j * torch.sin(ph)
        X[:kk, t] = (e @ (seg * w).to(torch.complex128) * (2.0 / w.sum())).to(torch.complex64)
    return X


def inter_harmonic_coherence(X: torch.Tensor, smooth: int = 3) -> torch.Tensor:
    """IHPC (§5-2, "IHPC alone is best"): coherence of the SECOND DIFFERENCE of
    phase across harmonic index.

    A periodic frame has phi_k = k*theta + tract(k): both the constant phase and
    the linear (delay) term vanish under phi_{k+1} - 2 phi_k + phi_{k-1}, and the
    vocal-tract term varies smoothly, so cos(.) -> 1. Noise has independent
    phases, so cos(.) averages to 0. Being phase-based it does not depend on
    resolving the harmonic peak, which is what defeated the magnitude feature
    (AS) at every window length."""
    p = torch.angle(X)
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]
    coh = torch.cos(d2)
    alive = (X[2:].abs() > 0) & (X[1:-1].abs() > 0) & (X[:-2].abs() > 0)
    coh = torch.where(alive, coh, torch.full_like(coh, -1.0))
    if smooth > 1:
        coh = torch.nn.functional.avg_pool1d(coh.T[None], smooth, 1, smooth // 2)[0].T[: coh.shape[0]]
    return coh


def pitch_sync_mag(x: torch.Tensor, f0: torch.Tensor, periods: int = 4,
                   nfft: int = NFFT, hop: int = HOP) -> torch.Tensor:
    """Pitch-adaptive (4-period Hann) magnitude spectra, as §5-2 specifies.

    A fixed 46 ms window smears the upper harmonics of real, non-stationary
    speech, so harmonic contrast decays smoothly instead of showing the knee
    that MVF is defined by. Four periods (~19 ms at 209 Hz) tracks the pitch.
    Matches this project's own earlier finding that a pitch-adaptive envelope
    was the winning recipe."""
    T = f0.shape[-1]
    nb = nfft // 2 + 1
    out = torch.zeros(nb, T)
    xp = torch.nn.functional.pad(x, (nfft, nfft))
    for t in range(T):
        f = float(f0[t]) if float(f0[t]) > 50.0 else 200.0
        L = min(int(round(periods * SR / f)) | 1, nfft)
        c = t * hop + nfft
        seg = xp[c - L // 2: c - L // 2 + L]
        if seg.shape[-1] < L:
            break
        w = torch.hann_window(L)
        out[:, t] = torch.fft.rfft(seg * w, n=nfft).abs() * (2.0 / w.sum())
    return out


def estimate_mvf(x: torch.Tensor, f0: torch.Tensor, floor: float = 1000.0,
                 thr: float | None = None, kmax: int = 400,
                 X: torch.Tensor | None = None, run: int = 3):
    """Maximum Voiced Frequency (§5-1/5-2): the boundary above which the signal
    stops being harmonic. Decided from IHPC by Drugman's ML rule

        MVF = argmax_m [PI_{k<=m} p(x_k|H1) * PI_{l>m} p(x_l|H0)]

    which for a log-likelihood ratio s_k = coherence - thr is the argmax of the
    PREFIX SUM of s_k -- tolerant to an isolated harmonic dying in a formant
    null, unlike an unbroken run."""
    T = f0.shape[-1]
    mvf = torch.full((T,), floor)
    voiced = f0 > 50
    if not voiced.any():
        return mvf
    if X is None:
        X = harmonic_analysis(x, f0, kmax)
    coh = inter_harmonic_coherence(X, smooth=5)             # [kmax-2, T]
    K = coh.shape[0]
    # Per-frame ML crossing (Drugman fits H1/H0 distributions rather than using
    # a fixed value: a clean tone sits near 1.0 while real speech, whose tract
    # phase swings across every formant, sits near 0.1 where it is still fully
    # harmonic). Reference the frame's own low harmonics against its own top
    # band, which is above MVF by construction.
    # References must be taken over LIVE harmonics only. kmax is a fixed array
    # bound (400), but a 209 Hz frame only has ~105 harmonics below Nyquist; the
    # rest are dead entries. Averaging those as "the top band" makes the
    # threshold depend on kmax instead of on the signal.
    kk = torch.floor((SR / 2 - 1) / f0.clamp(min=1.0)).clamp(3, K)
    pos = torch.arange(K, dtype=torch.float32)[:, None] / kk[None]
    lo_m = (pos <= 0.10) | (torch.arange(K)[:, None] < 4)
    hi_m = (pos > 0.80) & (pos <= 1.0)
    lo_ref = (coh * lo_m).sum(0) / lo_m.sum(0).clamp(min=1)
    hi_ref = (coh * hi_m).sum(0) / hi_m.sum(0).clamp(min=1)
    level = thr if thr is not None else (lo_ref + hi_ref) / 2
    bad = (coh <= level[None]).float()
    # First run of 3 consecutive incoherent harmonics -- NOT a prefix-sum argmax,
    # which integrates any threshold bias and drifts to Nyquist.
    runn = torch.nn.functional.avg_pool1d(bad.T[None], run, 1)[0].T >= 0.99
    ar = torch.arange(runn.shape[0], dtype=torch.float32)[:, None].expand_as(runn)
    first = torch.where(runn, ar, torch.full_like(ar, float(K))).min(0).values
    best = first + 2.0
    mvf = torch.where(voiced, (best * f0).clamp(min=floor, max=SR / 2 - 1), mvf)
    # MVF is a physical boundary and must move smoothly. Unsmoothed it jumped up
    # to 5.4 kHz between adjacent 5.8 ms frames (p99), which re-splits harmonic
    # and noise violently and clicks. Median then a slew limit of one octave per
    # 10 frames.
    if not MVF_SMOOTH:
        return _median_filter(mvf, 5)
    mvf = _median_filter(mvf, 9)
    lim = mvf.clone()
    for i in range(1, len(lim)):
        hi, lo = lim[i - 1] * 1.07, lim[i - 1] / 1.07
        lim[i] = mvf[i].clamp(lo, hi)
    return lim


# ------------------------------------------------------- aperiodicity (§6-1)

def band_edges(sr: int = SR, step: float = 3000.0, cap: float = 15000.0):
    top = min(cap, sr / 2 - 3000.0)
    n = max(1, int(top // step))
    return [(i * step, (i + 1) * step) for i in range(n)] + [(n * step, sr / 2)]


def estimate_bap(x: torch.Tensor, f0: torch.Tensor):
    """Band aperiodicity (D4C-style ratio, §6-1): noise power / total power per
    3 kHz band, in 0..1. The noise floor is the inter-harmonic minimum, so this
    measures how much of each band is NOT explained by the harmonic series."""
    mag = stft(x).abs()
    nb, T = mag.shape
    binhz = SR / NFFT
    bands = band_edges()
    ap = torch.ones(len(bands), T)
    voiced = f0 > 50
    if not voiced.any():
        return ap, bands
    p0 = int((f0[voiced].median() / binhz).clamp(min=2).item())
    pw = mag ** 2
    floor = -torch.nn.functional.max_pool1d(-pw.T[None], kernel_size=2 * p0 + 1,
                                            stride=1, padding=p0)[0].T
    for i, (lo, hi) in enumerate(bands):
        a, b = int(lo / binhz), min(int(hi / binhz), nb)
        if b <= a:
            continue
        ap[i] = (floor[a:b].sum(0) / (pw[a:b].sum(0) + EPS)).clamp(0.001, 1.0)
    ap = torch.where(voiced[None].expand_as(ap), ap, torch.ones_like(ap))
    return ap, bands


def bap_to_bins(ap: torch.Tensor, bands, nb: int = NB) -> torch.Tensor:
    """Piecewise-constant band aperiodicity -> per-bin curve [nb, T]."""
    binhz = SR / NFFT
    out = torch.ones(nb, ap.shape[-1])
    for i, (lo, hi) in enumerate(bands):
        a, b = int(lo / binhz), min(int(hi / binhz), nb)
        if b > a:
            out[a:b] = ap[i][None]
    return out


# ------------------------------------------------------------- filter (§4-5)

def _min_phase_ir(logmag: torch.Tensor, nfft: int) -> torch.Tensor:
    """Minimum-phase IR from log-magnitude via the real cepstrum (causal
    lifter). Causal by construction => no added algorithmic latency."""
    c = torch.fft.irfft(logmag.transpose(-1, -2).to(torch.complex64), n=nfft, dim=-1).real
    n = nfft
    lift = torch.zeros(n, device=c.device)
    lift[0] = 1.0
    lift[1:n // 2] = 2.0
    lift[n // 2] = 1.0
    C = torch.fft.rfft(c * lift, n=n, dim=-1)
    return torch.fft.irfft(torch.exp(C), n=n, dim=-1)


def _linear_phase_ir(logmag: torch.Tensor, nfft: int) -> torch.Tensor:
    ir = torch.fft.irfft(torch.exp(logmag).transpose(-1, -2).to(torch.complex64), n=nfft, dim=-1).real
    return torch.roll(ir, nfft // 2, dims=-1)


def mixed_phase_ir(cceps: torch.Tensor, nfft: int, quef: int) -> torch.Tensor:
    """NHV-style mixed-phase IR (§7-4/7-5) from a low-quefrency complex cepstrum.
    Positive quefrency = causal = vocal tract (minimum phase); NEGATIVE quefrency
    = anticausal = glottal open phase. Min-phase-only synthesis drops the latter,
    which is exactly the buzzy/robotic residue."""
    n = cceps.shape[-1]
    keep = torch.zeros(n, device=cceps.device)
    keep[: quef + 1] = 1.0
    keep[-quef:] = 1.0
    C = torch.fft.fft(cceps * keep, n=n, dim=-1)
    return torch.fft.ifft(torch.exp(C), n=n, dim=-1).real[..., :nfft]


def complex_cepstrum(x: torch.Tensor, nfft: int = NFFT, hop: int = HOP) -> torch.Tensor:
    """Per-frame complex cepstrum (oracle mixed-phase source, §7-4/7-5).

    The LINEAR PHASE MUST BE REMOVED FIRST. Framing puts a large pure delay on
    every frame (unwrapped phase reaches ~500 rad here); a delay is not
    representable in the low-quefrency region, it swamps the cepstrum and breaks
    the Hermitian extension, which makes min-phase and max-phase systems look
    identical. Subtracting phi[0] and the ramp to phi[nyq] leaves the zero-delay
    system, whose negative quefrency is the genuine anticausal (glottal open
    phase) part."""
    X = stft(x, nfft, hop)
    nb = X.shape[0]
    ph = _unwrap(torch.angle(X))
    k = torch.arange(nb, device=X.device, dtype=ph.dtype)[:, None] / (nb - 1)
    ph = ph - ph[:1] - k * (ph[-1:] - ph[:1])                 # phi[0]=phi[nyq]=0
    logX = torch.log(X.abs() + EPS) + 1j * ph
    full = torch.cat([logX, logX[1:-1].flip(0).conj()], dim=0)
    c = torch.fft.ifft(full.transpose(0, 1), dim=-1)
    assert float(c.imag.abs().max()) < 1e-3, "hermitian extension broken"
    return c.real


def _unwrap(p: torch.Tensor) -> torch.Tensor:
    d = torch.diff(p, dim=0)
    d = d - 2 * math.pi * torch.round(d / (2 * math.pi))
    return torch.cat([p[:1], p[:1] + torch.cumsum(d, dim=0)], dim=0)


def fir_filter(x: torch.Tensor, logmag: torch.Tensor, phase_mode: str = "min",
               cceps: torch.Tensor | None = None, quef: int = 80,
               nfft: int = NFFT, hop: int = HOP) -> torch.Tensor:
    """Frequency-sampling FIR applied by STFT overlap-add (§4-5). logmag is
    [nb, T]; the IR is windowed to `nfft` and convolved frame-wise."""
    n = x.shape[-1]
    if phase_mode == "mixed":
        raise NotImplementedError(
            "mixed-phase is NOT VERIFIED and is disabled. rddsp_verify.py shows "
            "the complex cepstrum does not yet separate min- from max-phase "
            "(0.909 vs 1.117; a working estimator would give <0.5 vs >2.0), so "
            "the anticausal/glottal-open-phase part it is supposed to recover is "
            "not actually being recovered. It is the OPTIONAL fallback of "
            "survey 10-3, only needed if the sine-excitation phase reference "
            "still sounds robotic. Do not enable it until the test passes.")
    elif phase_mode == "linear":
        ir = _linear_phase_ir(logmag, nfft)
    else:
        ir = _min_phase_ir(logmag, nfft)
    T = min(ir.shape[0], logmag.shape[-1])
    H = torch.fft.rfft(ir[:T], n=2 * nfft, dim=-1)
    X = stft(x, nfft, hop)[:, :T]
    xf = torch.fft.irfft(X.transpose(0, 1), n=nfft, dim=-1)
    XF = torch.fft.rfft(xf * _win(nfft), n=2 * nfft, dim=-1)
    y = torch.fft.irfft(XF * H, n=2 * nfft, dim=-1)[..., :nfft]
    Y = torch.fft.rfft(y, n=nfft, dim=-1).transpose(0, 1)
    return istft(Y, n, nfft, hop)


def cyclic_envelope(gci: torch.Tensor, n: int, f0_mean: float,
                    decay: float = 4.0) -> torch.Tensor:
    """Glottal-synchronous AMPLITUDE envelope, mean-normalised to 1.

    The cyclic-noise SOURCE is comb-structured in frequency, which is fine when
    only its phase is used but wrong as the input to a filter: filtering it with
    H = nm gives |N_src|*nm, i.e. the comb lands in the output twice. Carry the
    glottal synchrony as a time envelope on white noise instead -- flat in
    frequency, so H = nm/const is exact."""
    T0 = SR / max(f0_mean, 50.0)
    L = int(4 * T0)
    burst = torch.exp(-decay * torch.arange(L) / T0)
    pulses = torch.zeros(n + L)
    idx = gci[(gci >= 0) & (gci < n)]
    pulses[idx] = 1.0
    e = torch.nn.functional.conv1d(pulses[None, None], burst.flip(0)[None, None],
                                   padding=L - 1)[0, 0][:n]
    # Normalise LOCALLY, not globally. A global mean makes the envelope ~0
    # wherever there are no epochs, i.e. every unvoiced region, so the white
    # source there collapses to (1-cyc) of its level -- 20 dB down at cyc=0.9,
    # which is exactly the 18 dB dip measured at every IR length. Local
    # normalisation modulates around 1 everywhere and falls back to a flat 1
    # where there is nothing to synchronise to.
    k = int(4 * T0) | 1
    loc = torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(e[None, None], (k // 2, k // 2), mode="replicate"),
        k, 1)[0, 0][:n]
    return torch.where(loc > EPS, e / loc.clamp(min=EPS), torch.ones_like(e))


def _filtered_noise(nm: torch.Tensor, src: torch.Tensor, n: int,
                    nfft: int, hop: int, irlen: int = 512) -> torch.Tensor:
    """Time-varying FIR filtering of white noise (survey 4-5).

    The only shippable form of the breath branch: one causal pass, no
    iteration. Imposing a target magnitude on an arbitrary phase does not
    produce a signal with that magnitude (measured +11 dB over target in quiet
    frames = the hiss burst in near-silence), and the fix for that, Griffin-Lim,
    needs eight whole-signal round trips plus block latency, which a sub-30 ms
    budget cannot pay. Filtering has no inconsistency to correct: the output
    spectrum IS the input spectrum times the response.

    Correctness details that each cost a rebuild when missed:
      * the filter is nm / E|white|, NOT nm -- otherwise the source's own
        spectrum multiplies into the output;
      * the analysis window satisfies COLA at this hop so the normaliser is a
        constant; dividing by sum(w^2) is only valid when the frame content is
        unchanged, and convolution changes it;
      * the convolution is kept at its full length nfft+L-1 instead of being
        truncated back to nfft, which discarded the filter tail;
      * frame 0 of a center=True grid sits at sample -nfft/2, so the fold output
        is trimmed by that much (12 ms of pure delay when it was not)."""
    L = irlen or (NOISE_IRLEN or (nfft // 4))
    T = nm.shape[-1]
    lg = torch.log(nm / _white_gain(nfft, hop) + EPS)
    if NOISE_SMOOTH:
        c = torch.fft.irfft(lg.T.to(torch.complex64), n=nfft, dim=-1).real
        keep = torch.zeros(nfft)
        keep[: NOISE_SMOOTH] = 1.0
        keep[-NOISE_SMOOTH + 1:] = 1.0
        lg = torch.fft.rfft(c * keep, n=nfft, dim=-1).real.T
    ir = _min_phase_ir(lg, nfft)[:T, :L]
    # Taper only the LAST eighth. hann(2L)[L:] decays from 1 to 0 across the
    # whole IR, which attenuated its entire second half -- measured as a 13-14 dB
    # deficit on quiet broadband frames, and it survived even at L = nfft where
    # nothing is truncated at all. The taper only has to kill the discontinuity
    # at the cut.
    tp = max(8, L // 8)
    ir = ir.clone()
    ir[:, -tp:] = ir[:, -tp:] * torch.hann_window(2 * tp)[tp:][None, :]
    w = torch.hann_window(nfft)
    xp = torch.nn.functional.pad(src, (nfft, nfft))
    idx = torch.arange(nfft)[None, :] + torch.arange(T)[:, None] * hop
    fr = xp[idx + nfft - nfft // 2] * w
    M = nfft + L - 1
    Y = torch.fft.rfft(fr, n=M, dim=-1) * torch.fft.rfft(ir, n=M, dim=-1)
    yf = torch.fft.irfft(Y, n=M, dim=-1)
    out = torch.nn.functional.fold(yf.T[None], (1, (T - 1) * hop + M),
                                   (1, M), stride=(1, hop))[0, 0, 0]
    cola = float(torch.stack([w[i::hop].sum() for i in range(hop)]).mean())
    out = out[nfft // 2:] / max(cola, EPS)
    # No whole-signal gain here. In the shipping path this output is used only
    # as a PHASE source (S = stft(y); S /= |S|), so any scalar cancels -- removing
    # it changed PESQ by 0.0000 on every utterance -- and it was one more
    # non-causal term on a ledger that is supposed to be empty.
    return torch.nn.functional.pad(out, (0, max(0, n - out.shape[-1])))[:n]


_WG: dict = {}


def _white_gain(nfft: int, hop: int) -> float:
    """Expected |STFT| of unit-variance white noise on this grid."""
    k = (nfft, hop)
    if k not in _WG:
        g = torch.Generator().manual_seed(7)
        wn = torch.randn(SR, generator=g)
        _WG[k] = float(stft(wn, nfft, hop).abs().mean())
    return _WG[k]


def rtisi_la(nm: torch.Tensor, n: int, nfft: int, hop: int,
             lookahead: int = 2, maxit: int = 5,
             init: torch.Tensor | None = None) -> torch.Tensor:
    """RTISI-LA (Zhu, Beauregard & Wyse, ICME 2006) -- phase reconstruction with
    a BOUNDED look-ahead, reimplemented from the algorithm description.

    Offline Griffin-Lim fixes the overlap-add magnitude inconsistency but needs
    the whole signal, so it cannot ship under a 30 ms budget. RTISI-LA gets the
    same effect frame by frame: keep a buffer of (lookahead+1) frames, iterate
    the magnitude/phase projection inside it, then COMMIT ONLY THE OLDEST frame
    and slide. Algorithmic latency is exactly lookahead*hop -- 5.8 ms per frame
    at 1024/256 -- and the published result is that it beats Griffin-Lim at
    equal compute.

    Not the LTFAT implementation (GPL); written from the paper's description."""
    T = nm.shape[-1]
    w = _win(nfft)
    total = (T - 1) * hop + nfft
    acc = torch.zeros(total)
    wsq = torch.zeros(total)
    cola = float(torch.stack([(w ** 2)[i::hop].sum() for i in range(hop)]).mean())
    C = [None] * T
    ph0 = init if init is not None else None
    for m in range(T):
        hi = min(T - 1, m + lookahead)
        for j in range(m, hi + 1):
            if C[j] is None:
                if ph0 is not None:
                    C[j] = nm[:, j] * ph0[:, j] / (ph0[:, j].abs() + EPS)
                else:
                    g = torch.Generator().manual_seed(1234 + j)
                    a = torch.rand(nm.shape[0], generator=g) * 2 * math.pi
                    C[j] = nm[:, j] * (torch.cos(a) + 1j * torch.sin(a))
        for _ in range(maxit):
            for j in range(m, hi + 1):
                seg = acc.clone()
                sw = wsq.clone()
                for k in range(m, hi + 1):
                    fr = torch.fft.irfft(C[k], n=nfft) * w
                    seg[k * hop: k * hop + nfft] += fr
                    sw[k * hop: k * hop + nfft] += w ** 2
                # Normalise by the FULL-overlap constant, not by the partial
                # window sum. Frames at the edge of the look-ahead buffer have
                # incomplete support, so dividing by what has accumulated so far
                # inflates them; the paper's implementation carries dedicated
                # edge windows for exactly this reason.
                loc = seg[j * hop: j * hop + nfft] / cola
                P = torch.fft.rfft(loc * w, n=nfft)
                C[j] = nm[:, j] * P / (P.abs() + EPS)
        fr = torch.fft.irfft(C[m], n=nfft) * w
        acc[m * hop: m * hop + nfft] += fr
        wsq[m * hop: m * hop + nfft] += w ** 2
    y = acc / (wsq + 1e-8)
    return torch.nn.functional.pad(y, (0, max(0, n - y.shape[-1])))[:n]


def gl_bounded(nm: torch.Tensor, S0: torch.Tensor, n: int, nfft: int, hop: int,
               iters: int, la: int, ctx: int) -> torch.Tensor:
    """Griffin-Lim with a BOUNDED lookahead, so the iteration count stops being a
    latency term.

    Plain Griffin-Lim is a whole-signal STFT round trip per iteration: output
    sample t is touched by every frame, so it can never ship. The projection is
    local though -- one iteration mixes a frame only with the frames its window
    overlaps -- so committing frame by frame with `la` frames of lookahead and
    `ctx` frames of already-committed history reproduces it up to the truncation.
    Latency is (2*la - 1)*hop on top of the synthesis window, NOT la*hop: the
    commit step is `la` frames, so the FIRST frame of a block has to wait for the
    whole block plus its lookahead. Perturbation-measured 840 samples (19.0 ms)
    at the block head rising to 1093 (24.8 ms) at its tail. It does NOT grow with
    `iters`, which is the property that makes it shippable at all. Committed frames are re-imposed after every iteration; without
    that the past keeps moving and the block boundaries click."""
    T = nm.shape[-1]
    Y = nm * S0
    out = Y.clone()
    w = _win(nfft)
    b, step = 0, max(1, la)
    while b < T:
        e = min(T, b + step)
        lo, hi = max(0, b - ctx), min(T, e + la)
        F = hi - lo
        loc = Y[:, lo:hi].clone()
        loc[:, : b - lo] = out[:, lo:b]
        for _ in range(iters):
            seg = torch.istft(loc, nfft, hop, nfft, w, center=True, length=(F - 1) * hop)
            Z = torch.stft(seg, nfft, hop, nfft, w, center=True, return_complex=True)
            Z = Z[:, :F]
            loc = nm[:, lo:hi] * Z / (Z.abs() + EPS)
            loc[:, : b - lo] = out[:, lo:b]
        out[:, b:e] = loc[:, b - lo: e - lo]
        b = e
    return istft(out, n, nfft, hop)


_MELB: dict = {}


def band_reduce(nm: torch.Tensor, nfft: int, bands: int) -> torch.Tensor:
    """Project a magnitude spectrogram onto `bands` mel bands and back.

    Energy-preserving per band: the band POWER is kept and redistributed over the
    band's bins in proportion to the filter, so the result is the same envelope
    with the within-band fine structure removed."""
    key = (nfft, bands)
    if key not in _MELB:
        import librosa as _lb
        _MELB[key] = torch.tensor(_lb.filters.mel(sr=SR, n_fft=nfft, n_mels=bands,
                                                  fmin=0.0, fmax=SR / 2), dtype=torch.float32)
    M = _MELB[key]
    p = nm ** 2
    bp = M @ p
    norm = (M.sum(1, keepdim=True).clamp(min=EPS))
    back = M.T @ (bp / norm)
    den = (M.T @ (M.sum(1, keepdim=True) / norm)).clamp(min=EPS)
    return (back / den).clamp(min=0.0).sqrt()


def _shaped_noise(nm: torch.Tensor, phase_src: torch.Tensor | None, n: int,
                  seed: int = 0, nfft: int = NFFT, hop: int = HOP) -> torch.Tensor:
    """Render noise with magnitude `nm`, self-calibrating the overlap-add gain.

    A fixed correction constant is wrong here: independent per-bin random phase
    adds INCOHERENTLY across the 8x overlap (about -9 dB), while a cyclic-noise
    source is coherent frame to frame and loses nothing. Using the random-phase
    constant for a coherent source overshoots by exactly that 9 dB (measured).
    Match the achieved STFT energy to the target instead, which holds for any
    source coherence."""
    if NOISE_BANDS:
        nm = band_reduce(nm, nfft, NOISE_BANDS)
    if NOISE_MODE == "rtisi":
        S = None
        if phase_src is not None:
            S = stft(phase_src, nfft, hop)[:, : nm.shape[-1]]
        y = rtisi_la(nm, n, nfft, hop, RTISI_LA, RTISI_IT, S)
        got = stft(y, nfft, hop).abs()
        T = min(got.shape[-1], nm.shape[-1])
        return y * torch.sqrt((nm[:, :T] ** 2).sum() / ((got[:, :T] ** 2).sum() + EPS))
    if NOISE_MODE == "fir" and phase_src is not None:
        return _filtered_noise(nm, phase_src, n, nfft, hop)
    if NOISE_PREFILTER and phase_src is not None:
        phase_src = _filtered_noise(nm, phase_src, n, nfft, hop)
    if phase_src is None:
        g = torch.Generator().manual_seed(seed)
        ph = torch.rand(nm.shape, generator=g) * 2 * math.pi
        S = torch.cos(ph) + 1j * torch.sin(ph)
    else:
        S = stft(phase_src, nfft, hop)[:, : nm.shape[-1]]
        S = S / (S.abs() + EPS)
        nm = nm[:, : S.shape[-1]]
    if NOISE_GL and GL_LA:
        return gl_bounded(nm, S, n, nfft, hop, NOISE_GL, GL_LA, GL_CTX)
    y = istft(nm * S, n, nfft, hop)
    for _ in range(NOISE_GL):
        Y = stft(y, nfft, hop)
        T2 = min(Y.shape[-1], nm.shape[-1])
        Y = nm[:, :T2] * Y[:, :T2] / (Y[:, :T2].abs() + EPS)
        y = istft(Y, n, nfft, hop)
    if not GLOBAL_GAIN:
        return y / _white_gain(nfft, hop) if phase_src is None else y
    got = stft(y, nfft, hop).abs()
    T = min(got.shape[-1], nm.shape[-1])
    g = torch.sqrt((nm[:, :T] ** 2).sum() / ((got[:, :T] ** 2).sum() + EPS))
    if not NOISE_ITER:
        return y * g
    # One global scalar cannot fix a LOCAL error. Overlap-add with an
    # inconsistent phase source leaks energy from loud frames into their quiet
    # neighbours: measured +28 dB on the quietest frames while loud frames are
    # accurate to -0.9 dB. That is a burst of hiss where the recording is nearly
    # silent, which is exactly where mouth sounds sit. Correct the target
    # per frame and re-render.
    tgt = nm[:, :T]
    cur = nm.clone()
    for _ in range(NOISE_ITER):
        et = tgt.pow(2).sum(0, keepdim=True).sqrt()
        eg = (got[:, :T] * g).pow(2).sum(0, keepdim=True).sqrt()
        cur = cur.clone()
        cur[:, :T] = cur[:, :T] * (et / (eg + EPS)).clamp(0.1, 10.0)
        y = istft(cur * S, n, nfft, hop)
        got = stft(y, nfft, hop).abs()
        g = torch.sqrt((tgt ** 2).sum() / ((got[:, :T] ** 2).sum() + EPS))
    return y * g


# ------------------------------------------------------------- renderer (Psi)

_RPG: dict = {}


def rand_phase_gain(nfft: int = NFFT, hop: int = HOP) -> float:
    """Gain of istft(M * e^{j*random}) relative to M.

    istft normalises for COHERENT overlap-add. Random-phase frames add
    incoherently, so the reconstructed magnitude comes out systematically low
    (about 9 dB at 8x overlap). The factor depends only on (window, nfft, hop),
    so measure it once and divide it out -- otherwise the whole noise branch,
    i.e. every breath in the signal, is quietly attenuated."""
    key = (nfft, hop)
    if key not in _RPG:
        T = 200
        M = torch.ones(nfft // 2 + 1, T)
        g = torch.Generator().manual_seed(12345)
        ph = torch.rand(M.shape, generator=g) * 2 * math.pi
        y = istft(M * (torch.cos(ph) + 1j * torch.sin(ph)), (T - 1) * hop, nfft, hop)
        m = stft(y, nfft, hop).abs()
        a, b = 10, m.shape[-1] - 10
        _RPG[key] = float(m[:, a:b].mean())
    return _RPG[key]



def _sample_curve(curve: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
    """Read a per-bin frame curve [nb, T] at arbitrary sample-rate frequencies
    [n] (curve already upsampled to [nb, n])."""
    binhz = SR / NFFT
    b = (freq / binhz).clamp(0, curve.shape[0] - 2)
    lo = b.long()
    fr = b - lo
    idx = torch.arange(freq.shape[-1], device=freq.device)
    return curve[lo, idx] * (1 - fr) + curve[lo + 1, idx] * fr


def synthesize(f0: torch.Tensor, mvf: torch.Tensor, apbins: torch.Tensor,
               amp: torch.Tensor, noisemag: torch.Tensor, n: int,
               cceps: torch.Tensor | None = None, phase_mode: str = "min",
               kmax: int = 400, soft: float = 0.15, seed: int = 0):
    """Psi: harmonic bank (MVF-gated, aperiodicity-weighted) + shaped noise.

    amp      [nb, T]  calibrated linear magnitude, read at k*f0 -> A_k
    noisemag [nb, T]  linear magnitude of the noise floor
    mvf      [T]      harmonic/noise boundary in Hz (soft transition)
    apbins   [nb, T]  aperiodicity 0..1 (1 = fully noise)
    Phase comes from cumsum of the instantaneous harmonic frequency (§7-6 (a)),
    so no phase is ever regressed."""
    dev = f0.device
    f0u = frame_upsample(f0.double(), n).clamp(min=0.0)
    mvfu = frame_upsample(mvf.double(), n)
    ampu = frame_upsample(amp, n)
    apu = frame_upsample(apbins, n)
    phase = 2 * math.pi * torch.cumsum(f0u, dim=-1) / SR
    h = torch.zeros(n, dtype=torch.float64, device=dev)
    voiced = (f0u > 50).double()
    for k in range(1, kmax + 1):
        fk = k * f0u
        if (fk < SR / 2).sum() == 0:
            break
        alive = (fk < SR / 2 - SR / NFFT).double()
        gate = torch.sigmoid((mvfu - fk) / (soft * mvfu.clamp(min=1.0)))
        a = _sample_curve(ampu, fk.float()).double()
        ap = _sample_curve(apu, fk.float()).double()
        h = h + a * (1.0 - ap) * gate * alive * voiced * torch.sin(k * phase)
    binhz = SR / NFFT
    fb = (torch.arange(NB, device=dev, dtype=torch.float32) * binhz)[:, None]
    mvff = mvf[None].clamp(min=1.0)
    above = torch.sigmoid((fb - mvff) / (soft * mvff))
    nm = noisemag                       # already MVF-shaped by analyze()
    noise = _shaped_noise(nm, None, n, seed)
    y = h.float() + noise
    if phase_mode == "mixed":
        y = fir_filter(y, torch.zeros_like(noisemag), "mixed", cceps=cceps)   # disabled; raises
    return y, h.float(), noise


def hopratio(nh: int) -> float:
    return nh / HOP


def analyze(x: torch.Tensor, kmax: int = 400):
    """Oracle parameter extraction: everything Psi needs, measured from gt."""
    X = stft(x)
    w = _win()
    cal = 2.0 / w.sum()
    amp = X.abs() * cal
    f0, voi = harmonic_sum_f0(x)
    mvf = estimate_mvf(x, f0)
    ap, bands = estimate_bap(x, f0)
    apbins = bap_to_bins(ap, bands)
    binhz = SR / NFFT
    p0 = int(max(2, FLOOR_K * ((f0[f0 > 50].median() / binhz).item() if (f0 > 50).any() else 4)))
    pw = X.abs()
    floor = -torch.nn.functional.max_pool1d(-pw.T[None], kernel_size=2 * p0 + 1,
                                            stride=1, padding=p0)[0].T
    # The noise branch must carry the FULL magnitude above MVF (nothing else
    # renders it there) and only the inter-harmonic floor below MVF (where the
    # harmonic bank supplies the peaks). Handing it the floor everywhere loses
    # ~20 dB above MVF.
    fb = (torch.arange(NB, dtype=torch.float32) * binhz)[:, None]
    mvf = (mvf * MVF_SCALE).clamp(min=MVF_FLOOR, max=SR / 2 - 1)
    mv = mvf[None].clamp(min=1.0)
    above = torch.sigmoid((fb - mv) / (GATE_SOFT * mv))
    # An UNVOICED frame has no harmonics, so "the inter-harmonic floor" is not a
    # meaningful quantity there -- the whole spectrum is noise and the noise
    # branch must carry all of it. Leaving MVF at its 1 kHz floor made 38% of
    # frames synthesise everything below 1 kHz from the min-pooled valley, far
    # too quiet: 0-500 Hz was the worst band in the signal (LSD 1.185, worse
    # even than random phase).
    if UNVOICED_FULL:
        above = torch.where((f0 > 50)[None], above, torch.ones_like(above))
    if SUBF0_FULL:
        # BELOW the fundamental there is no harmonic either, so the
        # inter-harmonic floor is not a meaningful target there any more than it
        # is in an unvoiced frame -- and with f0 at 170-280 Hz that is the whole
        # 0-250 Hz octave. The full-band gate (rddsp_fullband.py) measures a
        # consistent -1.3 dB there in EVERY configuration, the largest single
        # band error left and the one no knob moved.
        sub = torch.sigmoid((f0.clamp(min=1.0)[None] * SUBF0_FULL - fb) / (0.2 * mv))
        above = torch.clamp(above + sub * (f0 > 50)[None].float(), max=1.0)
    # blend the two noise targets in POWER, for the same reason as the harmonic
    # gate above.
    # RAW |X| here, not `amp`: `amp` is calibrated to a sinusoid's peak
    # amplitude (x2/sum(w)), while the noise branch is rendered through istft
    # and needs the STFT magnitude itself. Mixing the two scales costs ~54 dB.
    fg = FLOOR_GAIN
    noisemag = (torch.sqrt((fg * floor) ** 2 * (1.0 - above) + X.abs() ** 2 * above)
                if POWER_GATE else fg * floor * (1.0 - above) + X.abs() * above)
    fine = None
    if NOISE_NFFT:
        nf, nh = NOISE_NFFT, NOISE_NFFT // NOISE_HOPDIV
        Xf = stft(x, nf, nh)
        nbf = nf // 2 + 1
        bhf = SR / nf
        pf = int(max(2, FLOOR_K * ((f0[f0 > 50].median() / bhf).item() if (f0 > 50).any() else 4)))
        flf = -torch.nn.functional.max_pool1d(-Xf.abs().T[None], kernel_size=2 * pf + 1,
                                              stride=1, padding=pf)[0].T
        Tf = Xf.shape[-1]
        idx = (torch.arange(Tf) * hopratio(nh)).clamp(0, mvf.shape[-1] - 1).long()
        mvff2 = mvf[idx]
        vf = (f0[idx] > 50)
        fbf = (torch.arange(nbf, dtype=torch.float32) * bhf)[:, None]
        ab = torch.sigmoid((fbf - mvff2[None].clamp(min=1.0)) / (GATE_SOFT * mvff2[None].clamp(min=1.0)))
        if UNVOICED_FULL:
            ab = torch.where(vf[None], ab, torch.ones_like(ab))
        if SUBF0_FULL:
            sb = torch.sigmoid((f0[idx].clamp(min=1.0)[None] * SUBF0_FULL - fbf)
                               / (0.2 * mvff2[None].clamp(min=1.0)))
            ab = torch.clamp(ab + sb * vf[None].float(), max=1.0)
        fine = (torch.sqrt((fg * flf) ** 2 * (1.0 - ab) + Xf.abs() ** 2 * ab)
                if POWER_GATE else fg * flf * (1.0 - ab) + Xf.abs() * ab)
    return dict(f0=f0, voi=voi, mvf=mvf, noisefine=fine, ap=ap, apbins=apbins, bands=bands,
                amp=amp, noisemag=noisemag, floor=floor, cceps=complex_cepstrum(x))


# ------------------------------------------------- GCI / mixed phase (§9-6, §7-5)

def zff_gci(x: torch.Tensor, f0_mean: float = 200.0, passes: int = 3):
    """Zero-Frequency Filtering GCI detection (§9-6, Murty & Yegnanarayana).

    Needs NO f0 estimator and no GPL code: difference the signal, pass it twice
    through a 0 Hz resonator (double pole at z=1), remove the polynomial trend
    with a window of about one pitch period, and read the positive zero
    crossings as glottal closure instants. Robust to the weak fundamental that
    defeats autocorrelation. Verified: 223.9 Hz vs the detector's 208.9 Hz."""
    from scipy.signal import lfilter
    s = x.double().numpy()
    d = np.diff(s, prepend=s[:1])
    y = lfilter([1.0], [1.0, -2.0, 1.0], d)
    y = lfilter([1.0], [1.0, -2.0, 1.0], y)
    N = max(3, int(round(SR / max(f0_mean, 50.0))))
    k = 2 * N + 1
    for _ in range(passes):
        pad = np.pad(y, (N, N), mode="edge")
        trend = np.convolve(pad, np.ones(k) / k, mode="valid")
        y = y - trend[: len(y)]
    z = np.where((y[:-1] <= 0) & (y[1:] > 0))[0] + 1
    soe = np.abs(np.diff(y, prepend=y[:1]))[z]
    return torch.tensor(z, dtype=torch.long), torch.tensor(soe, dtype=torch.float32)


def refine_gci(gci: torch.Tensor, soe: torch.Tensor, f0_at: torch.Tensor,
               tol: float = 0.62):
    """Pitch-guided epoch selection.

    Raw ZFF zero crossings are noisy: measured 35.8% spurious (interval < 0.6 T0)
    and 21.8% missed (> 1.5 T0) on real speech. Every spurious epoch injects a
    wrong glottal phase measurement, and interpolating between a good and a bad
    epoch sweeps the harmonic phase -- audible as crackle. Keep the strongest
    candidate per period (refractory selection by strength of excitation), then
    fill gaps at the expected position so the phase track never spans two
    periods."""
    if len(gci) == 0:
        return gci, soe
    T0 = SR / f0_at.clamp(min=50.0)
    order = torch.argsort(soe, descending=True)
    keep = torch.zeros(len(gci), dtype=torch.bool)
    acc: list[int] = []
    acc_t = torch.empty(0)
    for i in order.tolist():
        t = float(gci[i])
        if len(acc) and float((acc_t - t).abs().min()) < tol * float(T0[i]):
            continue
        acc.append(i)
        acc_t = torch.cat([acc_t, torch.tensor([t])])
        keep[i] = True
    idx = torch.argsort(gci[keep])
    return gci[keep][idx], soe[keep][idx]



def _unwrap_1d(p: torch.Tensor) -> torch.Tensor:
    d = torch.diff(p)
    d = d - 2 * math.pi * torch.round(d / (2 * math.pi))
    return torch.cat([p[:1], p[:1] + torch.cumsum(d, dim=0)])


def gci_complex_cepstrum(x: torch.Tensor, gci: torch.Tensor, f0_mean: float = 200.0,
                         nfft: int = 4096, tilt: float = 3000.0, periods: int = 2):
    """Complex cepstrum on GCI-CENTRED, exponentially weighted frames (§7-5).

    Three things make this work, all absent from a plain STFT frame: the window
    is centred on the glottal closure so the anticausal open-phase part lands on
    the negative-quefrency side; the frame is tilted so the zeros come off the
    unit circle and phase unwrapping stops breaking at spectral nulls; and FFT
    4096 gives the quefrency resolution.

    On the weighting: the survey quotes "alpha=0.72" without saying what it
    weights, and read as an exponential taper that is far too strong -- it drags
    every zero inside the unit circle and INVERTS the decomposition (measured:
    min-phase 1.037, max-phase 0.370). What is actually needed is only to get
    the zeros OFF the circle, in either direction. Calibrated against known
    minimum- and maximum-phase systems: separation 0.0005 vs 365.9."""
    L = max(16, int(round(periods * SR / max(f0_mean, 50.0))) | 1)
    xp = torch.nn.functional.pad(x, (nfft, nfft)).double()
    w = torch.hann_window(L, dtype=torch.float64)
    a = tilt ** (torch.arange(L, dtype=torch.float64) / L)
    a = a / a.mean()
    out = torch.zeros(len(gci), nfft, dtype=torch.float64)
    for i, g in enumerate(gci.tolist()):
        c = g + nfft
        seg = xp[c - L // 2: c - L // 2 + L]
        if seg.shape[-1] < L:
            break
        f = torch.fft.fft(torch.nn.functional.pad(seg * w * a, (0, nfft - L)), n=nfft)
        mag = torch.log(f.abs() + EPS)
        ph = _unwrap_1d(torch.angle(f))
        kk = torch.arange(nfft, dtype=torch.float64) / nfft
        ph = ph - ph[0] - kk * (ph[nfft // 2] - ph[0]) * 2.0
        out[i] = torch.fft.ifft(mag + 1j * ph).real
    return out


def causal_anticausal(cc: torch.Tensor, quef: int = 40):
    """Split a complex cepstrum into minimum-phase (positive quefrency, vocal
    tract) and maximum-phase (negative quefrency, glottal open phase) impulse
    responses."""
    n = cc.shape[-1]
    kc = torch.zeros(n, dtype=cc.dtype)
    ka = torch.zeros(n, dtype=cc.dtype)
    kc[0] = ka[0] = 0.5
    kc[1: quef + 1] = 1.0
    ka[-quef:] = 1.0
    ir_c = torch.fft.ifft(torch.exp(torch.fft.fft(cc * kc, dim=-1)), dim=-1).real
    ir_a = torch.fft.ifft(torch.exp(torch.fft.fft(cc * ka, dim=-1)), dim=-1).real
    return ir_c, ir_a


# ---------------------------------- glottal-phase harmonics & cyclic noise

def harmonic_analysis_at(x: torch.Tensor, pos: torch.Tensor, f0_at: torch.Tensor,
                         kmax: int = 400, periods: int = 2):
    """harmonic_analysis() evaluated at ARBITRARY sample positions (we use GCIs).

    TWO periods, not the four the MVF estimator uses: PESQ is monotonic in this
    window length (2: 2.898, 3: 2.876, 4: 2.814, 6: 2.729, 8: 2.662). A longer
    window averages the amplitude and phase over cycles that genuinely differ,
    so the resynthesised pulse train is smoother than the target. MVF still
    wants four, because there the job is resolving a harmonic peak rather than
    tracking its change.

    Anchoring to the glottal closure is what makes the measured harmonic phases
    meaningful: relative to the GCI they encode the glottal pulse shape -- the
    mixed-phase structure, causal vocal tract plus ANTICAUSAL open phase -- and
    are stable cycle to cycle. Relative to an arbitrary frame centre they are
    just a delay and carry nothing."""
    P = len(pos)
    X = torch.zeros(kmax, P, dtype=torch.complex64)
    xp = torch.nn.functional.pad(x, (NFFT, NFFT)).double()
    ks = torch.arange(1, kmax + 1, dtype=torch.float64)[:, None]
    for i in range(P):
        f = float(f0_at[i])
        if f <= 50.0:
            continue
        L = min(int(round(periods * SR / f)) | 1, NFFT)
        c = int(pos[i]) + NFFT
        seg = xp[c - L // 2: c - L // 2 + L]
        if seg.shape[-1] < L:
            break
        w = torch.hann_window(L, dtype=torch.float64)
        nn = torch.arange(L, dtype=torch.float64)[None, :] - L // 2
        kk = int(min(kmax, (SR / 2 - 1) // f))
        ph = -2 * math.pi * ks[:kk] * f * nn / SR
        e = torch.cos(ph) + 1j * torch.sin(ph)
        X[:kk, i] = (e @ (seg * w).to(torch.complex128) * (2.0 / w.sum())).to(torch.complex64)
    return X


def fill_f0(f0: torch.Tensor) -> torch.Tensor:
    """Log-linear interpolation of f0 across unvoiced gaps.

    harmonic_sum_f0() reports 0 where it finds no pitch, and the phase track is
    the integral of the UPSAMPLED f0, so every voiced onset makes the whole
    harmonic bank sweep from 50*k Hz up to f0*k inside one hop -- an audible
    chirp, and a frequency error exactly where the amplitude is also stepping.
    The oscillator wants a continuous frequency; whether a frame is voiced is a
    separate question, answered by the amplitude mask."""
    v = f0 > 50
    nv = int(v.sum())
    if nv == 0:
        return torch.full_like(f0, 200.0)
    if nv == 1:
        return torch.full_like(f0, float(f0[v][0]))
    idx = torch.arange(len(f0), dtype=torch.float64)
    vi = idx[v]
    lg = torch.log(f0[v].double())
    j = torch.searchsorted(vi, idx).clamp(1, nv - 1)
    if CAUSAL_ONLY:
        # Hold the last voiced f0. Interpolating toward the NEXT one needs the
        # end of the gap before the gap can be rendered.
        out = torch.exp(torch.where(idx < vi[0], lg[0], lg[(j - 1).clamp(0, nv - 1)]))
    else:
        t = ((idx - vi[j - 1]) / (vi[j] - vi[j - 1]).clamp(min=1.0)).clamp(0, 1)
        out = torch.exp(lg[j - 1] * (1 - t) + lg[j] * t)
    return torch.where(v, f0.double(), out).to(f0.dtype)


def phase_track(f0: torch.Tensor, n: int):
    """The carrier both analysis and synthesis must agree on."""
    f0u = frame_upsample((fill_f0(f0) if F0_FILL else f0).double(), n).clamp(min=0.0)
    return 2 * math.pi * torch.cumsum(f0u, dim=-1) / SR, f0u


def voicing(f0: torch.Tensor, n: int) -> torch.Tensor:
    v = (f0 > 50).double()
    if VOI_RAMP > 0:
        k = torch.hann_window(2 * VOI_RAMP + 1, periodic=False, dtype=torch.double)
        if CAUSAL_ONLY:
            k = k[: VOI_RAMP + 1]
            # NOT flipped. conv1d cross-correlates, so with a left pad of
            # VOI_RAMP the last kernel tap lands on the CURRENT frame; flipping
            # put hann's leading zero there instead, giving the current frame
            # zero weight and turning a causal ramp into a 2-frame delay on the
            # harmonic branch alone.
            v = torch.nn.functional.conv1d(
                torch.nn.functional.pad(v[None, None], (VOI_RAMP, 0), mode="replicate"),
                (k / k.sum())[None, None])[0, 0][: len(f0)]
        else:
            v = torch.nn.functional.conv1d(v[None, None], (k / k.sum())[None, None],
                                           padding=VOI_RAMP)[0, 0][: len(f0)]
    return frame_upsample(v, n).clamp(0.0, 1.0)


def model_phase(Xg: torch.Tensor, Phi_g: torch.Tensor, f0_at: torch.Tensor,
                cut: float, soft: float = 0.5) -> torch.Tensor:
    """Replace the MEASURED harmonic phase above `cut` Hz with the minimum phase
    implied by the harmonic amplitudes.

    Raising MVF always lost, and this is why: the glottal-relative phase measured
    at successive epochs is near-random above about 1.5 kHz (the epoch-to-epoch
    offset jitters by 1.0-1.4 rad against pi/2 = 1.571 for pure noise), so a
    harmonic rendered up there is a sinusoid with a random phase per cycle --
    worse than shaped noise, which is exactly what the MVF sweep kept saying.

    The amplitude is still good up there. Minimum phase reconstructs the vocal
    tract's dispersion from it through the Hilbert relation, so the upper
    harmonics become a coherent pulse anchored at the GCI instead of noise: the
    source-filter half of the mixed-phase model, which this file has always had
    on the ANALYSIS side (gci_complex_cepstrum, causal_anticausal) and never
    wired into the renderer. Below `cut` the measured phase is kept, because
    there it carries the anticausal open phase that minimum phase cannot."""
    K, P = Xg.shape
    ph = torch.remainder(torch.arange(1, K + 1, dtype=torch.float64)[:, None]
                         * Phi_g.double()[None, :], 2 * math.pi)
    Om = Xg * (torch.cos(ph) - 1j * torch.sin(ph)).to(torch.complex64)
    a_ = Om.abs()
    lg = torch.log(a_.double() + 1e-7)
    # cepstral minimum phase along the HARMONIC INDEX axis; mirror so the
    # transform sees a real even sequence.
    N = 2 * K
    sym = torch.cat([lg, lg.flip(0)], dim=0)
    c = torch.fft.ifft(sym.to(torch.complex128), dim=0).real
    fold = torch.zeros_like(c)
    fold[0] = c[0]
    fold[1:K] = 2 * c[1:K]
    fold[K] = c[K]
    mp = torch.angle(torch.exp(torch.fft.fft(fold.to(torch.complex128), dim=0)))[:K]
    fk = torch.arange(1, K + 1, dtype=torch.float64)[:, None] * f0_at.double()[None, :]
    w = torch.sigmoid((fk - cut) / (soft * max(cut, 1.0)))
    meas = torch.angle(Om).double()
    new = meas * (1 - w) + (meas + torch.remainder(mp - meas + math.pi, 2 * math.pi) - math.pi) * w
    out = (a_.double() * torch.exp(1j * (new + ph))).to(torch.complex64)
    return out


def smooth_phasor(v: torch.Tensor, width: int) -> torch.Tensor:
    """Causal moving average of a unit phasor along the node axis, renormalised.

    Averaging the PHASOR rather than the phase avoids unwrapping, and leaving the
    magnitude alone keeps the envelope's time resolution -- only the offset that
    jitters gets smoothed."""
    if width <= 1:
        return v
    k = torch.ones(1, 1, width, dtype=torch.float64) / width
    def pad(t):
        return torch.nn.functional.pad(t[None, None].double(), (width - 1, 0), mode="replicate")
    r = torch.nn.functional.conv1d(pad(v.real), k)[0, 0]
    i = torch.nn.functional.conv1d(pad(v.imag), k)[0, 0]
    o = torch.complex(r, i)
    return (o / (o.abs() + EPS)).to(v.dtype)


def _lp_centred(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    L = w.shape[-1]
    zr = torch.nn.functional.conv1d(z.real[None, None], w[None, None].flip(-1),
                                    padding=L // 2)[0, 0]
    zi = torch.nn.functional.conv1d(z.imag[None, None], w[None, None].flip(-1),
                                    padding=L // 2)[0, 0]
    return (zr + 1j * zi)[: z.shape[-1]]


def env_nodes(gci: torch.Tensor, rate: int = 0) -> torch.Tensor:
    """Envelope sampling grid: `rate` nodes per glottal period, GCIs included."""
    rate = rate or ENV_RATE
    if rate <= 1 or len(gci) < 2:
        return gci
    out = [gci]
    for m in range(1, rate):
        out.append(gci[:-1] + (torch.diff(gci) * m) // rate)
    # unique, not sort: a short epoch interval can put two subdivisions on the
    # same sample, and a zero-length node interval makes the interpolation
    # weight blow up before it is clamped.
    return torch.unique(torch.cat(out))


def harmonic_envelopes(x: torch.Tensor, f0: torch.Tensor, pos: torch.Tensor,
                       fmax: float, kmax: int = 400) -> torch.Tensor:
    """Complex envelope of each harmonic, sampled at `pos`.

    Demodulate by the SAME carrier synthesis will use -- exp(j k Phi) with Phi
    the integral of the f0 track -- and low-pass with a Hann of ENV_PERIODS
    glottal periods. Returned in harmonic_analysis_at()'s convention, i.e. with
    the carrier phase at the node folded back in, so synthesize_v2() removes it
    exactly as before.

    Only harmonics below `fmax` are computed. The crossover gate is
    sigmoid((MVF - f)/(GATE_SOFT*MVF)), which is 5e-5 at 2.5*MVF, so harmonics
    far above the split cost time and contribute nothing."""
    n = x.shape[-1]
    if ENV_LS:
        return harmonic_ls(x, f0, pos, fmax, kmax)
    Phi, f0u = phase_track(f0, n)
    fm = median_f0(f0)
    L = int(ENV_PERIODS * SR / fm) | 1
    w = torch.hann_window(L, periodic=False, dtype=torch.float64)
    w = w / w.sum()
    idx = pos.clamp(0, n - 1).long()
    X = torch.zeros(kmax, len(pos), dtype=torch.complex64)
    xd = x.double()
    # Bound k by the LOWEST voiced f0. The old test was per-sample on the FILLED
    # f0, which dips toward 60 Hz inside unvoiced gaps, so the loop ran to k=104
    # to render 14 audible harmonics -- seven times the work.
    fv = f0[f0 > 50]
    fmin = float(fv.min()) if fv.numel() else 200.0
    kcap = min(kmax, int(min(fmax, SR / 2) / max(fmin, 50.0)) + 1)
    for k in range(1, kcap + 1):
        fk = k * f0u
        car = torch.exp(1j * k * Phi)
        A = 2.0 * _lp_centred(xd * car.conj(), w)
        X[k - 1] = (A[idx] * car[idx]).to(torch.complex64)
    return X


def harmonic_ls(x: torch.Tensor, f0: torch.Tensor, pos: torch.Tensor,
                fmax: float, kmax: int = 400) -> torch.Tensor:
    """Least-squares harmonic amplitudes (Stylianou HNM), one solve per node.

    The Hann demodulator separates harmonic k from k+-1 only because the window
    is long enough to put its nulls on them, which is why ENV_PERIODS has to be
    4 once the envelope is sampled once per period -- and four periods is 18 ms,
    so every amplitude is averaged over eighteen milliseconds of speech. Solving
    for all harmonics JOINTLY removes that constraint: the basis vectors are
    made orthogonal by the solve rather than by the window, so two periods
    separates them exactly.

    It is also far harder to abuse. The fit spans 2K real dimensions out of L
    samples -- about 5% here -- so a signal with no harmonic structure has
    nowhere to hide, where a Hann bank with ENV_PERIODS < 2 reconstructs white
    noise at 18 dB. rddsp_bankcheck.py measures exactly this.

    Returns harmonic_analysis_at()'s convention: the carrier phase at the node
    folded back in."""
    n = x.shape[-1]
    Phi, _ = phase_track(f0, n)
    fm = median_f0(f0)
    L = int(LS_PERIODS * SR / fm) | 1
    half = L // 2
    idx = pos.clamp(0, n - 1).long()
    # An UNVOICED node reports f0 = 0. Clamping that to 50 Hz sizes the basis at
    # K = fmax/50 and marks every column live, so A'WA goes singular and the
    # solve returns garbage that the voicing gate then multiplies by zero -- but
    # only after it has polluted the batch's ridge. Park those nodes at the
    # utterance median instead; the gate discards them either way.
    f0_at = f0[(idx // HOP).clamp(0, len(f0) - 1)]
    f0_at = torch.where(f0_at > 50, f0_at, torch.full_like(f0_at, median_f0(f0)))
    w = torch.hann_window(L, periodic=False, dtype=torch.float64)
    off = torch.arange(-half, half + 1)
    xd = x.double()
    out = torch.zeros(kmax, len(pos), dtype=torch.complex64)
    for s in range(0, len(idx), LS_CHUNK):
        # K from the LOWEST f0 IN THIS CHUNK, not in the whole utterance. The
        # global minimum is an unvoiced-gap value (60 Hz measured), which sized
        # the basis at 144 columns when 30 were live: the dead columns are zero,
        # so they drag down mean(diag(A'WA)) and dilute the ridge fivefold, and
        # the solve costs (144/30)^3.
        fl = f0_at[s: s + LS_CHUNK].double().clamp(min=50.0)[:, None]
        K = max(1, min(kmax, int(min(fmax, SR / 2 - SR / NFFT) / float(fl.min()))))
        ks = torch.arange(1, K + 1, dtype=torch.float64)
        ii = (idx[s: s + LS_CHUNK, None] + off[None, :]).clamp(0, n - 1)
        ph = Phi[ii][:, :, None] * ks[None, None, :]                # [P,L,K]
        A = torch.cat([torch.cos(ph), -torch.sin(ph)], dim=-1)      # [P,L,2K]
        # K comes from the LOWEST voiced f0 in the utterance, so at a node whose
        # own f0 is higher, the top columns sit above Nyquist. Those basis
        # vectors alias onto lower ones and make A'WA singular -- the fit then
        # spreads a harmonic's energy across the degenerate pair. Drop them per
        # node; the ridge sends their coefficients to zero.
        live = (ks[None, :] * fl < min(fmax, SR / 2 - SR / NFFT)).double()
        A = A * torch.cat([live, live], dim=-1)[:, None, :]
        Aw = A * w[None, :, None]
        G = Aw.transpose(1, 2) @ A
        r = Aw.transpose(1, 2) @ xd[ii][:, :, None]
        d = torch.diagonal(G, dim1=1, dim2=2).mean(-1)[:, None, None]
        G = G + LS_RIDGE * d * torch.eye(2 * K, dtype=torch.float64)[None]
        th = torch.linalg.solve(G, r)[:, :, 0]                      # [P,2K]
        c = torch.complex(th[:, :K], th[:, K:])
        car = torch.exp(1j * (Phi[idx[s: s + LS_CHUNK]][:, None] * ks[None, :]))
        out[:K, s: s + LS_CHUNK] = (c * car).T.to(torch.complex64)
    return out


def cyclic_noise(gci: torch.Tensor, n: int, f0_mean: float, decay: float = 4.0,
                 seed: int = 0) -> torch.Tensor:
    """Cyclic noise (Wang & Yamagishi, arXiv:2004.02191): the convolution of a
    pulse train with a decaying static random noise.

    Independent random phase per STFT bin -- what the v1 noise branch did --
    is hiss with no relationship to the glottal cycle. Real breath is
    re-excited at every closure, so its envelope is modulated at f0. The paper
    reaches for cyclic noise exactly when the target is LESS periodic (breathy,
    whispered), which is the ASMR case. Measured envelope modulation at f0:
    3.55 vs 2.79 for random phase, ground truth 2.92."""
    g = torch.Generator().manual_seed(seed)
    T0 = SR / max(f0_mean, 50.0)
    L = int(4 * T0)
    burst = torch.randn(L, generator=g) * torch.exp(-decay * torch.arange(L) / T0)
    burst = burst / burst.norm().clamp(min=EPS)
    pulses = torch.zeros(n + L)
    idx = gci[(gci >= 0) & (gci < n)]
    pulses[idx] = 1.0
    return torch.nn.functional.conv1d(pulses[None, None], burst.flip(0)[None, None],
                                      padding=L - 1)[0, 0][:n]


def _env_mod(sig: torch.Tensor, f0_mean: float) -> float:
    from scipy.signal import hilbert
    e = np.abs(hilbert(sig.detach().numpy().astype(np.float64)))
    e = e - e.mean()
    E = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    fr = np.fft.rfftfreq(len(e), 1 / SR)
    band = (fr > f0_mean * 0.85) & (fr < f0_mean * 1.15)
    ref = (fr > 20) & (fr < f0_mean * 3)
    return float(E[band].max() / (np.median(E[ref]) + 1e-12))


def choose_cyc(x: torch.Tensor, h: torch.Tensor, gci: torch.Tensor,
               noisemag: torch.Tensor, f0_mean: float, n: int,
               cands=(0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)):
    """Measure the cyclic/white mix from the signal instead of fixing it.

    How glottal-synchronous the breath is varies utterance to utterance (ground
    truth envelope modulation at f0 spans 3.3 to 4.8 across the eval set), so a
    constant mix over-modulated some and under-modulated others in the same run.
    Compare the FULL rendered output against the full reference: matching the
    noise branch alone to all of gt is not a comparison, because gt's harmonics
    dominate its modulation at f0. The harmonic part does not depend on the mix,
    so it is rendered once and reused."""
    tgt = _env_mod(x, f0_mean)
    cn = cyclic_noise(gci, n, f0_mean)
    g = torch.Generator().manual_seed(1)
    wn = torch.randn(n, generator=g)
    best, err = cands[0], float("inf")
    for c in cands:
        src = c * cn / cn.std().clamp(min=EPS) + (1 - c) * wn
        y = h + _shaped_noise(noisemag, src, n, 0)
        e = abs(_env_mod(y, f0_mean) - tgt)
        if e < err:
            best, err = c, e
    return best


def synthesize_v2(f0: torch.Tensor, mvf: torch.Tensor, apbins: torch.Tensor,
                  noisemag: torch.Tensor, n: int, gci: torch.Tensor,
                  Xg: torch.Tensor, f0_at: torch.Tensor, kmax: int = 400,
                  soft: float | None = None, cyc: float = 0.0, seed: int = 0,
                  noisefine: torch.Tensor | None = None,
                  harm_ap: torch.Tensor | None = None,
                  pos: torch.Tensor | None = None):
    """Psi v2: harmonic bank carrying the MEASURED glottal phase, plus a breath
    branch that is cyclic-noise (glottal-synchronous) rather than per-bin random
    phase.

    v1 synthesised sum A_k sin(k*Phi): every harmonic at zero relative phase,
    which IS an impulse train and is the textbook cause of buzz. Here each
    harmonic carries exp(j*psi_k) measured at the GCI, so the real glottal pulse
    shape -- including the anticausal open phase that minimum-phase-only
    synthesis discards -- is reproduced. Glottal cycle correlation 0.986 vs
    0.369 for v1."""
    dev = f0.device
    soft = GATE_SOFT if soft is None else soft
    Phi, f0u = phase_track(f0, n)
    mvfu = frame_upsample(mvf.double(), n)
    apu = frame_upsample(apbins, n)
    voiced = voicing(f0, n)

    # The envelope grid and the GCI grid are NOT the same thing any more: the
    # glottal epochs still trigger the cyclic noise, but the harmonic envelope is
    # sampled ENV_RATE times per period because at one node per period it is
    # sampled at its own Nyquist rate.
    gi = (gci if pos is None else pos).clamp(0, n - 1)
    t_g = gi.double()
    tt = torch.arange(n, dtype=torch.double)
    j = torch.searchsorted(t_g, tt).clamp(1, len(t_g) - 1)
    t0, t1 = t_g[j - 1], t_g[j]
    fr = ((tt - t0) / (t1 - t0).clamp(min=1.0)).clamp(0, 1)
    frc = fr.to(torch.complex64)

    # NEGATIVE RESULT (2026-07-29): advancing the phase exactly 2*pi per GCI --
    # true pitch-synchronous synthesis, which would reproduce natural jitter for
    # free -- is WORSE at the current epoch accuracy: LSD 0.582 -> 0.650, band
    # 1.09 -> 1.56 dB, cycle correlation 0.978 -> 0.946. About 10% of epochs are
    # missing (473 found vs 519 expected), and forcing one period across a
    # two-period span halves the instantaneous frequency there. Integrating the
    # smoothed f0 tolerates a missed epoch. Do not revisit without a complete
    # epoch track.
    Phi_g = Phi[gi]

    h = torch.zeros(n, dtype=torch.float64, device=dev)
    for k in range(1, kmax + 1):
        fk = k * f0u
        if (fk < SR / 2).sum() == 0:
            break
        # Take the modulo in float64 BEFORE the complex64 cast. Phi_g reaches
        # ~1e5 rad in eight seconds, so casting it to float32 first quantises it
        # to ~1.2e-3 rad; multiplied by k that is up to 0.25 rad at the top of
        # the band, and it gets WORSE the longer the file.
        ph = torch.remainder(k * Phi_g, 2 * math.pi)
        Ok = Xg[k - 1] * (torch.cos(ph) - 1j * torch.sin(ph)).to(torch.complex64)
        a = Ok.abs()
        if float(a.max()) <= 0:
            continue
        if PHASE_SMOOTH > 1:
            Ok = a * smooth_phasor(Ok / (a + EPS), PHASE_SMOOTH)
        if ENV_POLAR:
            o = Ok / (a + EPS)
            oi = o[j - 1] * (1 - frc) + o[j] * frc
            oi = oi / (oi.abs() + EPS)
            ai = (a[j - 1] * (1 - fr) + a[j] * fr).double()
        else:
            # Interpolate the COMPLEX envelope. |A| and arg(A) are not the
            # band-limited quantities -- A is -- and splitting them makes the
            # phase rotate at a constant rate between nodes while the amplitude
            # tracks a straight line through a curve.
            ci = Ok[j - 1] * (1 - frc) + Ok[j] * frc
            ai, oi = ci.abs().double(), ci
        # POWER-complementary gate. The harmonic and noise branches are
        # uncorrelated, so their powers add; making the amplitudes complementary
        # gives g^2 + (1-g)^2 = 0.5 at the crossover, a -3 dB hole exactly at
        # MVF. Measured as -1 to -2.2 dB across 1-4 kHz before this sqrt.
        gate = torch.sigmoid((mvfu - fk) / (soft * mvfu.clamp(min=1.0)))
        if POWER_GATE:
            gate = torch.sqrt(gate)
        if harm_ap is not None and k - 1 < harm_ap.shape[0]:
            ah = harm_ap[k - 1]
            ap = (ah[j - 1] * (1 - fr.to(ah.dtype)) + ah[j] * fr.to(ah.dtype)).double()
        else:
            ap = _sample_curve(apu, fk.float()).double()
        alive = (fk < SR / 2 - SR / NFFT).double()
        # cos, not sin: the analysis DFT measures phase in a COSINE convention.
        # sin adds a constant -pi/2 to every harmonic, and a constant offset
        # across harmonics is not a time shift (that would scale with k) -- it is
        # a different pulse shape. Measured: cycle correlation 0.005 vs 0.986.
        apw = torch.sqrt(1.0 - ap) if POWER_AP else (1.0 - ap)
        ph = torch.angle(oi).double()
        if GLOTTAL_PHASE == "zero":
            ph = torch.zeros_like(ph)
        elif GLOTTAL_PHASE == "random":
            ph = torch.full_like(ph, float(torch.rand(1, generator=torch.Generator().manual_seed(seed + 7919 * k)) * 6.283185))
        elif GLOTTAL_PHASE == "utt":
            # ONE glottal pulse shape for the whole utterance: the circular mean
            # of the measured per-harmonic phase. If this retains the quality,
            # the phase is a slowly-varying speaker/utterance property -- low
            # rate, and therefore predictable from content. If it collapses to
            # the zero-phase (buzzy) case, the phase has to be tracked per frame
            # and 12.5's H-G is hard.
            w = ai.double() * voiced.double()
            c = (w * torch.exp(1j * ph)).sum() / w.sum().clamp(min=1e-12)
            ph = torch.full_like(ph, float(torch.angle(c)))
        h = h + ai * apw * gate * alive * voiced * torch.cos(k * Phi + ph)

    fm = median_f0(f0)
    g2 = torch.Generator().manual_seed(seed + 1)
    wn = torch.randn(n, generator=g2)
    if NOISE_MODE == "fir":
        if PULSE_MIX > 0.0:
            # MIXED EXCITATION (WORLD/MELP). Above MVF there is no oscillator at
            # all in this design, so voiced speech up there has no harmonic
            # structure -- it is rendered as filtered noise, the residual keeps
            # the harmonics the harmonic branch never got, and every attempt to
            # put them back through the noise MAGNITUDE has been a transport
            # degeneracy. A pulse train at the glottal epochs generates those
            # lines instead, from f0, at no parameter cost, and the SMOOTH
            # envelope (NOISE_SMOOTH) shapes them -- which is the whole point of
            # source-filter: the fine structure belongs to the source.
            pt = torch.zeros(n)
            gi = gci[(gci >= 0) & (gci < n)]
            if len(gi):
                pt[gi] = 1.0
            pt = pt - pt.mean()
            pt = pt / pt.std().clamp(min=EPS)
            src = PULSE_MIX * pt + math.sqrt(max(0.0, 1.0 - PULSE_MIX ** 2)) * wn
        else:
            # White source, glottal synchrony carried as a TIME envelope.
            src = wn * (1.0 + cyc * (cyclic_envelope(gci, n, fm) - 1.0))
    else:
        cn = cyclic_noise(gci, n, fm, seed=seed)
        src = cyc * cn / cn.std().clamp(min=EPS) + (1 - cyc) * wn
    if noisefine is not None:
        noise = _shaped_noise(noisefine, src, n, seed, NOISE_NFFT, NOISE_NFFT // NOISE_HOPDIV)
    else:
        noise = _shaped_noise(noisemag, src, n, seed)
    return h.float() + noise, h.float(), noise


def resynthesize(x: torch.Tensor, v1: bool = False, **kw):
    """Analyse and resynthesise. Defaults to Psi v2 (glottal-phase harmonics +
    cyclic noise), which beats v1 on every measured axis: glottal cycle
    correlation 0.986 vs 0.369, LSD 0.618 vs 0.745, crest 19.7 vs 18.5 against
    a ground truth of 19.0. v1 is kept only for ablation."""
    p = analyze(x)
    n = x.shape[-1]
    if v1:
        y, h, nz = synthesize(p["f0"], p["mvf"], p["apbins"], p["amp"],
                              p["noisemag"], n, **kw)
        return y, h, nz, p
    f0 = p["f0"]
    fm = median_f0(f0)
    gci, soe = zff_gci(x, fm)
    # Keep only VOICED epochs. There is no glottal pulse in an unvoiced frame,
    # so every zero crossing ZFF reports there is spurious -- and 38% of frames
    # are unvoiced, which is where most of the measured 36% "spurious" rate came
    # from. Feeding those to the phase track and to the cyclic-noise trigger is
    # what crackles.
    f0_at = f0[(gci // HOP).clamp(0, len(f0) - 1)]
    vm = f0_at > 50
    gci, soe, f0_at = gci[vm], soe[vm], f0_at[vm]
    if GCI_REFINE:
        gci, soe = refine_gci(gci, soe, f0_at)
    f0_at = f0[(gci // HOP).clamp(0, len(f0) - 1)]
    if ENV_DEMOD:
        pos = env_nodes(gci)
        kw["pos"] = pos
        Xg = harmonic_envelopes(x, f0, pos, 2.5 * float(p["mvf"].max()))
    else:
        Xg = harmonic_analysis_at(x, gci, f0_at)
    if PH_MODEL_HZ:
        Phi_all, _ = phase_track(f0, x.shape[-1])
        Xg = model_phase(Xg, Phi_all[pos.clamp(0, x.shape[-1] - 1).long()],
                         f0[(pos // HOP).clamp(0, len(f0) - 1)], PH_MODEL_HZ)
    if HARM_AP:
        # coherence in 0..1 -> aperiodicity: an incoherent harmonic is noise.
        coh = inter_harmonic_coherence(Xg, smooth=5).clamp(-1, 1)
        aph = ((1.0 - coh) / 2.0).clamp(0.0, 1.0)
        kw["harm_ap"] = torch.cat([aph[:1], aph, aph[-1:]], dim=0)[: Xg.shape[0]]
    y, h, nz = synthesize_v2(f0, p["mvf"], p["apbins"], p["noisemag"], n,
                             gci, Xg, f0_at, noisefine=p.get("noisefine"), **kw)
    if "cyc" not in kw and CYC_MEASURE and not CAUSAL_ONLY:
        kw["cyc"] = choose_cyc(x, h, gci, p["noisemag"], fm, n)
        y, h, nz = synthesize_v2(f0, p["mvf"], p["apbins"], p["noisemag"], n,
                                 gci, Xg, f0_at, noisefine=p.get("noisefine"), **kw)
    if RESID_NOISE:
        # HNM proper: the breath branch carries what the harmonic branch did NOT
        # explain, |STFT(x - h)|, instead of a min-pooled guess at the
        # inter-harmonic floor. This only becomes the right thing once the
        # harmonic branch is a genuine band-limited harmonic model -- tried at
        # the old operating point it LOST (3.406 -> 3.258), because back then h
        # was wrong and its error went straight into the noise target.
        nf, nh = (NOISE_NFFT, NOISE_NFFT // NOISE_HOPDIV) if NOISE_NFFT else (NFFT, HOP)
        rm = stft(x - h, nf, nh).abs()
        if RESID_MIX < 1.0:
            base = p.get("noisefine") if NOISE_NFFT else p["noisemag"]
            T = min(rm.shape[-1], base.shape[-1])
            rm = rm[:, :T] * RESID_MIX + base[:, :T] * (1.0 - RESID_MIX)
        y, h, nz = synthesize_v2(f0, p["mvf"], p["apbins"], rm, n,
                                 gci, Xg, f0_at, noisefine=rm, **kw)
    p["cyc"] = kw.get("cyc")
    p["gci"], p["soe"], p["Xg"] = gci, soe, Xg
    return y, h, nz, p
