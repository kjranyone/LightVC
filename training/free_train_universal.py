"""R-proto-A universal: train FreeVocoder as an OWN universal (multi-speaker)
vocoder. Diagnosis (band-MAE, ear): free's unseen-utt かすれ is a generalization/
data-scale deficit, not architecture — single-speaker 63 utts can't match a
universal vocoder (bigvgan, the ceiling, is universally pretrained). Fix = train
free on the full female corpus so it generalizes to unseen utts like bigvgan,
while staying 100% own weights (no chimera).

af1ad5575a3fa383 is held OUT entirely (train excludes it) -> eval on its eval-3 +
unseen-6 = the かすれ test set, overwriting *_free.wav so the listen server shows
universal-free vs bigvgan(ceiling) on the same set. Judge = human ear only.

Usage: cd training && HF_HUB_OFFLINE=1 uv run python free_train_universal.py \
         --steps 200000 --buf 4000
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import bigvgan
from bigvgan.env import AttrDict
from bigvgan.meldataset import get_mel_spectrogram
from causal_mel import causal_mel, mel_la
from bigvgan.discriminators import (MultiPeriodDiscriminator,
                                    MultiResolutionDiscriminator)
from bigvgan.loss import (MultiScaleMelSpectrogramLoss, discriminator_loss,
                          feature_loss, generator_loss)
from free_vocoder import FreeVocoder, FreeVocoderIF, FreeVocoderGCI
from vocoder_a import FreeVocoderA
from kansei_train import mrstft

_CWIN = {}


def _win(nfft, dev):
    k = (nfft, dev)
    if k not in _CWIN:
        _CWIN[k] = torch.hann_window(nfft, device=dev)
    return _CWIN[k]


def sharpness_loss(y, y_hat, nfft=2048, hop=512, fl=400, fh=200):
    """Image-sharpness (定位感) loss, phase-isolating. Whiten each signal by its
    OWN magnitude (unit-magnitude spectrum -> phase-only signal), then match the
    per-frame l4/l2 peakiness (temporal energy concentration). Whitening removes
    all magnitude info so this term is purely phase-coherence (cross-frequency
    GCI alignment -> pulse compactness), unlike autocorrelation which is
    phase-blind (Wiener-Khinchin)."""
    w = _win(nfft, y.device)

    def whiten(s):
        S = torch.stft(s.squeeze(1), nfft, hop, nfft, w, return_complex=True)
        floor = 1e-3 * S.abs().amax(dim=(1, 2), keepdim=True)
        U = S / (S.abs() + floor + 1e-9)
        return torch.istft(U, nfft, hop, nfft, w)

    def sharp(e):
        fr = e.unfold(-1, fl, fh)                       # [B, nframes, fl]
        l4 = (fr.pow(4).sum(-1) + 1e-9).pow(0.25)
        l2 = (fr.pow(2).sum(-1) + 1e-9).sqrt()
        return (l4 / l2).mean()

    with torch.no_grad():
        s_t = sharp(whiten(y))
    return torch.abs(sharp(whiten(y_hat)) - s_t)


_FBANDS = [(100, 400), (400, 1000), (1000, 2500), (2500, 6000), (6000, 12000)]


def env_stab_loss(y, y_hat, nfft=2048, hop=512, smooth=86):
    """Envelope-stability loss = the VALIDATED 定位感のブレ proxy, made trainable.
    Per freq band, the log-envelope ERROR (y_hat vs y), smoothed over ~1s
    (86 frames @86fps), squared -> penalizes slow (~0.5-2Hz) drift of the
    spectral envelope away from truth = the wobble. Stable (no whitening/division,
    unlike the broken sharpness loss). Needs long segments to see slow rates."""
    w = _win(nfft, y.device)
    Sy = torch.stft(y.squeeze(1), nfft, hop, nfft, w, return_complex=True).abs()
    Syh = torch.stft(y_hat.squeeze(1), nfft, hop, nfft, w, return_complex=True).abs()
    f = torch.linspace(0, 22050, Sy.shape[1], device=y.device)
    kernel = torch.ones(1, 1, smooth, device=y.device) / smooth
    loss = 0.0
    for lo, hi in _FBANDS:
        m = (f >= lo) & (f < hi)
        eg = torch.log(Sy[:, m].mean(1) + 1e-6)
        eh = torch.log(Syh[:, m].mean(1) + 1e-6)
        err = eh - eg
        err = err - err.mean(dim=1, keepdim=True)
        s = F.conv1d(err.unsqueeze(1), kernel, padding=smooth // 2).squeeze(1)
        loss = loss + (s ** 2).mean()
    return loss / len(_FBANDS)


def crest_loss(y_hat, target=15.0):
    """Penalize crest factor (peak/rms) above target. Free-phase iSTFT-heads
    develop spikes (clicks) as they sharpen -> 'ひどすぎ'. Keeps output bounded
    (natural speech crest ~15) while the GAN/CQT-disc sharpen harmonics."""
    yf = y_hat.squeeze(1) if y_hat.dim() == 3 else y_hat
    rms = yf.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6
    peak = yf.abs().amax(dim=-1, keepdim=True)
    return F.relu(peak / rms - target).mean()


def hf_loss(y, y_hat, nfft=1024, hop=256, sr=44100, lo=6000):
    """High-band (>=6kHz) log-magnitude match. Direct-waveform is HF-deficient
    (must GENERATE HF vs iSTFT-head outputting spectrum bins for free) -> muffled.
    This term reallocates capacity to HF to fix the muffling."""
    w = _win(nfft, y.device)
    Y = torch.stft(y.squeeze(1), nfft, hop, nfft, w, return_complex=True).abs()
    Yh = torch.stft(y_hat.squeeze(1), nfft, hop, nfft, w, return_complex=True).abs()
    f = torch.linspace(0, sr / 2, Y.shape[1], device=y.device)
    m = f >= lo
    return (torch.log(Y[:, m] + 1e-5) - torch.log(Yh[:, m] + 1e-5)).abs().mean()


def gd_loss(y, y_hat, nfft=2048, hop=512):
    """Group-delay matching (cross-frequency phase coherence = common GCI).
    Match the frequency-difference of phase (group delay) of y_hat to y,
    anti-wrapped, weighted by target magnitude (high-energy bins only)."""
    w = _win(nfft, y.device)
    Y = torch.stft(y.squeeze(1), nfft, hop, nfft, w, return_complex=True)
    Yh = torch.stft(y_hat.squeeze(1), nfft, hop, nfft, w, return_complex=True)
    P, Ph = torch.angle(Y), torch.angle(Yh)
    gdY = P[:, 1:] - P[:, :-1]
    gdH = Ph[:, 1:] - Ph[:, :-1]
    d = gdH - gdY
    aw = torch.abs(d - 2 * torch.pi * torch.round(d / (2 * torch.pi)))
    wt = Y.abs()[:, 1:]
    return (aw * wt).sum() / (wt.sum() + 1e-9)

SNAP = Path("/home/kojirotanaka/.cache/huggingface/hub/models--nvidia--"
            "bigvgan_v2_44khz_128band_512x/snapshots/"
            "95a9d1dcb12906c03edd938d77b9333d6ded7dfb")
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "female-dataset"
HOLDOUT = "af1ad5575a3fa383"
DATA = CORPUS / HOLDOUT
CACHE = ROOT / "data/e2_ltv_cache/af1ad5575a3fa383"
TRIAGE = ROOT / "results/e2_triage"
SR = 44100
HOP = 512
EPS = 1e-8
DEV = "cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu")
UNSEEN = ["af1ad5575a3fa383_00027042", "af1ad5575a3fa383_00035042",
          "af1ad5575a3fa383_00035936", "af1ad5575a3fa383_00036578",
          "af1ad5575a3fa383_00036850", "af1ad5575a3fa383_00040870"]


_ALLPATHS = None
_EXTRA_CORPUS = []          # extra corpus roots (e.g. irodori-tts) for FULL-DATA training


def _corpus_paths():
    global _ALLPATHS
    if _ALLPATHS is None:
        ps = [p for p in CORPUS.rglob("*.wav") if HOLDOUT not in str(p)]
        for root in _EXTRA_CORPUS:
            ps += [p for p in Path(root).rglob("*.wav") if HOLDOUT not in str(p)]
        np.random.default_rng(0).shuffle(ps)
        _ALLPATHS = ps
        extra = f" + {len(_EXTRA_CORPUS)} extra root(s)" if _EXTRA_CORPUS else ""
        print(f"  corpus: {len(ps)} wavs (female-dataset{extra}, holdout excluded)", flush=True)
    return _ALLPATHS


def build_buffer(n, frames, chunk=0, include_target=False, target_repeat=1):
    """Rolling buffer: successive chunks walk the whole shuffled corpus so over
    a run every wav is used as teacher (RAM holds only n at a time)."""
    paths = _corpus_paths()
    buf, seg = [], frames * HOP
    idx, tried = (chunk * n) % len(paths), 0
    while len(buf) < n and tried < len(paths):
        p = paths[idx % len(paths)]
        idx += 1
        tried += 1
        try:
            x, _ = librosa.load(str(p), sr=SR, mono=True)
        except Exception:
            continue
        if len(x) >= seg + HOP:
            buf.append(x.astype(np.float32))
    if include_target:
        # add the target voice's utts (except the 9 held-out eval/unseen uids) so
        # the model is trained ON af1ad's distribution = the real product setup.
        # Diagnostic (2026-07-16): the 定位感 wobble is a generalization gap
        # (in-dist ~2.9 vs af1ad holdout 6.54); including the target closes it.
        heldout = set(UNSEEN) | {p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]}
        added = 0
        for p in sorted(DATA.glob("*.wav")):
            if p.stem in heldout:
                continue
            try:
                x, _ = librosa.load(str(p), sr=SR, mono=True)
            except Exception:
                continue
            if len(x) >= seg + HOP:
                for _ in range(target_repeat):
                    buf.append(x.astype(np.float32))
                added += 1
        print(f"  +{added} target ({HOLDOUT}) utts x{target_repeat} "
              f"= {added*target_repeat} slots ({added*target_repeat*100//max(len(buf),1)}% of buffer)",
              flush=True)
    return buf


def gain_match(y, gt):
    g = np.sqrt((gt ** 2).mean() / ((y ** 2).mean() + EPS))
    y = y * g
    peak = np.abs(y).max()
    if peak > 0.95:
        y = y * (0.95 / peak)
    return y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--buf", type=int, default=4000)
    ap.add_argument("--refresh-every", type=int, default=8000)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--nfft", type=int, default=2048)   # vocoder synthesis grid
    ap.add_argument("--win", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=512)     # = mel hop (frame align)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--tag", default="free")
    ap.add_argument("--init", default="")
    ap.add_argument("--init-free", default="")
    ap.add_argument("--ifhead", action="store_true")
    ap.add_argument("--gcihead", action="store_true")
    ap.add_argument("--archa", action="store_true",
                    help="Vocoder A: F0-non-slaved tiny-iSTFT + anti-aliased upsample (vocoder_a.py)")
    ap.add_argument("--archw", action="store_true",
                    help="Vocoder A-wave: F0-non-slaved causal direct-waveform (no phase head, escapes free-phase ceiling)")
    ap.add_argument("--archw-aa", type=int, default=0, help="anti-aliased leaky in wave head (heavier)")
    ap.add_argument("--archw-lean", type=int, default=1, help="drop MRF on last N (highest-rate) stages; 0=full MRF (HiFiGAN, fixes imaging)")
    ap.add_argument("--archw-act", default="leaky", choices=["leaky", "snake"], help="wave head activation; snake=BigVGAN SnakeBeta (HF/liveliness)")
    ap.add_argument("--archc", action="store_true", help="Vocoder A-C: LARGE-nfft causal iSTFT-head (0 lookahead, fine resolution, RTF~0.1)")
    ap.add_argument("--archc-nfft", type=int, default=2048, help="synthesis nfft for --archc (2048=22Hz/bin)")
    ap.add_argument("--archh", action="store_true", help="Vocoder A-H: heterodyned-phase iSTFT-head (coherent harmonics + aperiodic gate)")
    ap.add_argument("--archm", action="store_true", help="Vocoder A-M: heterodyne + internal-f0 harmonic-comb magnitude prior")
    ap.add_argument("--archcomb", action="store_true", help="Vocoder A-c11: PLAIN free phase + internal-f0 comb magnitude prior only (no heterodyne, no random-phase gate). Warm-starts from a c10 --archc ckpt")
    ap.add_argument("--comb-warm", type=int, default=20000, help="--archcomb: steps to blend a forced svoi (1->0) so internal f0 gets gradient before the net may switch the prior off; 0=off")
    ap.add_argument("--cqt-disc", action="store_true", help="add MS-SB-CQT discriminator (fixes over-smoothed/non-sharp harmonics, arXiv:2311.14957)")
    ap.add_argument("--lam-crest", type=float, default=0.0, help="crest-factor penalty (suppress free-phase spikes/clicks)")
    ap.add_argument("--gen-clip", type=float, default=100.0, help="generator grad-norm clip")
    ap.add_argument("--ema", type=float, default=0.0, help="generator EMA decay (0=off, e.g. 0.999); eval/save use EMA")
    ap.add_argument("--r1", type=float, default=0.0, help="R1 discriminator grad penalty weight (anti disc-overpowering)")
    ap.add_argument("--r1-every", type=int, default=16, help="apply R1 every N steps (lazy regularization)")
    ap.add_argument("--lam-mel", type=float, default=15.0, help="mel-loss weight (lower to ease LF-collapse)")
    ap.add_argument("--disc-lr", type=float, default=0.0, help="discriminator LR (0=use --lr; lower to weaken disc)")
    ap.add_argument("--extra-corpus", nargs="+", default=[],
                    help="extra corpus roots for FULL-DATA training, e.g. ../data/female_tts_corpus")
    ap.add_argument("--lam-mrstft", type=float, default=2.0)
    ap.add_argument("--lam-hf", type=float, default=0.0, help="high-band (>=6kHz) log-mag loss (fixes muffled/HF-deficient direct-waveform)")
    ap.add_argument("--lam-sharp", type=float, default=0.0)
    ap.add_argument("--lam-gd", type=float, default=0.0)
    ap.add_argument("--lam-envstab", type=float, default=0.0)
    ap.add_argument("--include-target", action="store_true")
    ap.add_argument("--target-repeat", type=int, default=1)
    ap.add_argument("--mel-framing", choices=["centered", "causal"], default="centered",
                    help="Z0-B: 'causal' = left-aligned 0-lookahead mel (path B, freec_B)")
    ap.add_argument("--mel-nfft", type=int, default=2048,
                    help="Z0-B mel analysis n_fft(=win); sweep {512,1024,2048}")
    ap.add_argument("--mel-look", type=int, default=0,
                    help="Z0-B framing lookahead in samples (0=causal, (nfft-hop)//2=centered)")
    args = ap.parse_args()
    global _EXTRA_CORPUS, HOP
    _EXTRA_CORPUS = [r for r in args.extra_corpus if Path(r).exists()]
    HOP = args.hop          # mel/frame hop == vocoder frame rate (archa upsamples x HOP)

    h = AttrDict(json.loads((SNAP / "config.json").read_text()))
    h["resolutions"] = [[1024, 256, 1024], [2048, 512, 2048], [512, 128, 512]]
    # config C asymmetric: mel analysis stays long (n_fft/win_size=2048 for
    # resolution) but mel hop matches the vocoder frame rate so short-window
    # synthesis stays aligned.
    h["hop_size"] = args.hop
    if args.archm:
        from vocoder_a import HarmonicISTFTHead
        gen = HarmonicISTFTHead(dim=args.dim, n_blocks=args.layers, nfft=args.archc_nfft,
                                hop=args.hop, causal=bool(args.causal)).to(DEV)
    elif args.archh:
        from vocoder_a import HeteroISTFTHead
        gen = HeteroISTFTHead(dim=args.dim, n_blocks=args.layers, nfft=args.archc_nfft,
                              hop=args.hop, causal=bool(args.causal)).to(DEV)
    elif args.archcomb:
        from vocoder_a import CombISTFTHead
        gen = CombISTFTHead(dim=args.dim, n_blocks=args.layers, nfft=args.archc_nfft,
                            hop=args.hop, causal=bool(args.causal)).to(DEV)
    elif args.archc:
        from vocoder_a import CausalISTFTHead
        gen = CausalISTFTHead(dim=args.dim, n_blocks=args.layers, nfft=args.archc_nfft,
                              hop=args.hop, causal=bool(args.causal)).to(DEV)
    elif args.archw:
        from vocoder_a import WaveVocoderA
        gen = WaveVocoderA(dim=args.dim, causal=bool(args.causal), aa=bool(args.archw_aa),
                           lean_last=args.archw_lean, act=args.archw_act).to(DEV)
    elif args.archa:
        gen = FreeVocoderA(dim=args.dim, n_blocks=args.layers, causal=bool(args.causal)).to(DEV)
    elif args.gcihead:
        gen = FreeVocoderGCI(dim=args.dim, n_layers=args.layers).to(DEV)
    elif args.ifhead:
        gen = FreeVocoderIF(dim=args.dim, n_layers=args.layers).to(DEV)
    else:
        gen = FreeVocoder(dim=args.dim, n_layers=args.layers, causal=bool(args.causal),
                          nfft=args.nfft, win=args.win, hop=args.hop).to(DEV)
    if args.init_free:
        gen.load_from_free(torch.load(args.init_free, map_location=DEV)["gen"])
        print(f"warm-start (backbone+mag) from FreeVocoder {args.init_free}", flush=True)
    elif args.init:
        miss, unexp = gen.load_state_dict(torch.load(args.init, map_location=DEV)["gen"], strict=False)
        if miss or unexp:
            print(f"warm-start PARTIAL: missing={list(miss)} unexpected={list(unexp)}", flush=True)
        print(f"warm-start gen from {args.init}", flush=True)
    mpd = MultiPeriodDiscriminator(h).to(DEV)
    mrd = MultiResolutionDiscriminator(h).to(DEV)
    cqtd = None
    if args.cqt_disc:
        from cqt_disc import MSSubBandCQTDisc
        cqtd = MSSubBandCQTDisc(sr=SR).to(DEV)
        print("added MS-SB-CQT discriminator (harmonic sharpness)", flush=True)
    fn_mel = MultiScaleMelSpectrogramLoss(sampling_rate=SR).to(DEV)
    og = torch.optim.AdamW(gen.parameters(), args.lr, betas=(0.8, 0.99))
    _dparams = list(mpd.parameters()) + list(mrd.parameters()) + (list(cqtd.parameters()) if cqtd else [])
    od = torch.optim.AdamW(_dparams, args.disc_lr or args.lr, betas=(0.8, 0.99))
    use_amp = bool(args.amp) and DEV == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ({k: v.detach().clone().float() for k, v in gen.state_dict().items()}
           if args.ema > 0 else None)

    def ac():
        return torch.autocast("cuda", dtype=torch.float16, enabled=use_amp)
    print(f"FreeVocoder {sum(p.numel() for p in gen.parameters())/1e6:.1f}M "
          f"UNIVERSAL on {DEV}", flush=True)

    t_buf = time.time()
    buf = build_buffer(args.buf, args.frames, include_target=args.include_target,
                       target_repeat=args.target_repeat)
    print(f"buffer {len(buf)} utts (holdout={HOLDOUT}) in "
          f"{(time.time()-t_buf)/60:.1f}m", flush=True)

    ev_uids = [p.stem for p in sorted(CACHE.glob("*.npz"))[-3:]]
    ev = []
    for uid in ev_uids + UNSEEN:
        x, _ = librosa.load(str(DATA / (uid + ".wav")), sr=SR, mono=True)
        ev.append((uid, x.astype(np.float32)))

    rng = np.random.default_rng(0)
    ckdir = ROOT / f"training/checkpoints/{args.tag}"
    ckdir.mkdir(parents=True, exist_ok=True)

    def sample():
        mels, ys = [], []
        for i in rng.integers(0, len(buf), args.batch):
            x = buf[i]
            nf = len(x) // HOP
            s = rng.integers(0, nf - args.frames)
            xw = x[s * HOP:(s + args.frames) * HOP]
            if args.mel_framing == "causal":
                mel = mel_la(torch.tensor(xw), n_fft=args.mel_nfft, hop=args.hop, look=args.mel_look)[0]
            else:
                mel = get_mel_spectrogram(torch.tensor(xw).unsqueeze(0), h)[0]
            mels.append(mel[..., :args.frames])
            ys.append(torch.tensor(xw[:args.frames * HOP]))
        return torch.stack(mels).to(DEV), torch.stack(ys).to(DEV)

    @torch.no_grad()
    def evaluate(step):
        gen.eval()
        backup = None
        if ema is not None:
            backup = {k: v.detach().clone() for k, v in gen.state_dict().items()}
            gen.load_state_dict({k: ema[k].to(backup[k].dtype) for k in backup})
        for uid, x in ev:
            nf = min(len(x) // HOP, int(8.0 * SR) // HOP)
            gt = x[:nf * HOP]
            if args.mel_framing == "causal":
                mel = mel_la(torch.tensor(gt).to(DEV), n_fft=args.mel_nfft, hop=args.hop, look=args.mel_look)
            else:
                mel = get_mel_spectrogram(torch.tensor(gt).unsqueeze(0), h).to(DEV)
            y = gen(mel).squeeze().cpu().numpy()
            sf.write(TRIAGE / f"{uid}_{args.tag}.wav", gain_match(y, gt[:len(y)]), SR)
        if backup is not None:
            gen.load_state_dict(backup)
        gen.train()
        print(f"[eval {step}] wrote {len(ev)} -> {args.tag}", flush=True)

    t0 = time.time()
    evaluate(0)
    chunk = 0
    for it in range(1, args.steps + 1):
        if hasattr(gen, "warm"):
            gen.warm = max(0.0, 1.0 - it / args.comb_warm) if args.comb_warm else 0.0
        if args.refresh_every and it % args.refresh_every == 0:
            chunk += 1
            buf = []                       # free old buffer BEFORE building new
            gc.collect()                   # (else 2x RAM peak -> OOM-kill)
            buf = build_buffer(args.buf, args.frames, chunk=chunk,
                               include_target=args.include_target,
                               target_repeat=args.target_repeat)
            print(f"[refresh {it}] buffer -> chunk {chunk} "
                  f"({len(buf)} utts, ~{chunk*args.buf} corpus wavs seen)", flush=True)
        mel, y = sample()
        y = y.unsqueeze(1)
        with ac():
            y_hat = gen(mel).unsqueeze(1)          # gen returns fp32 (iSTFT forced)
        if not torch.isfinite(y_hat).all():        # fp16-overflow guard: skip step, keep weights
            og.zero_grad(set_to_none=True); od.zero_grad(set_to_none=True)
            if it % 200 == 0:
                print(f"it {it:6d} non-finite y_hat -> skipped", flush=True)
            continue
        n = min(y.shape[-1], y_hat.shape[-1])
        y, y_hat = y[..., :n], y_hat[..., :n]

        od.zero_grad()
        with ac():
            yr, yg, _, _ = mpd(y, y_hat.detach())
            ldf, _, _ = discriminator_loss(yr, yg)
            yr, yg, _, _ = mrd(y, y_hat.detach())
            ldr, _, _ = discriminator_loss(yr, yg)
            d_all = ldf + ldr
            if cqtd is not None:
                yr, yg, _, _ = cqtd(y, y_hat.detach())
                ldc, _, _ = discriminator_loss(yr, yg)
                d_all = d_all + ldc
        scaler.scale(d_all).backward()
        if args.r1 > 0 and it % args.r1_every == 0:      # lazy R1 (assumes --amp 0)
            y_r = y.detach().requires_grad_(True)
            yr_m, _, _, _ = mpd(y_r, y_r)
            yr_r, _, _, _ = mrd(y_r, y_r)
            rsum = sum(o.sum() for o in yr_m) + sum(o.sum() for o in yr_r)
            gr = torch.autograd.grad(rsum, y_r, create_graph=True)[0]
            r1 = (0.5 * args.r1 * args.r1_every) * gr.pow(2).flatten(1).sum(1).mean()
            r1.backward()
        scaler.step(od)

        og.zero_grad()
        with ac():
            loss_mel = fn_mel(y, y_hat) * args.lam_mel
            yr, yg, fr, fg = mpd(y, y_hat)
            lfm_f, (lg_f, _) = feature_loss(fr, fg), generator_loss(yg)
            yr, yg, fr, fg = mrd(y, y_hat)
            lfm_r, (lg_r, _) = feature_loss(fr, fg), generator_loss(yg)
            if cqtd is not None:
                yr, yg, fr, fg = cqtd(y, y_hat)
                lg_c, _ = generator_loss(yg)
                lg_r = lg_r + lg_c
                lfm_r = lfm_r + 2.0 * feature_loss(fr, fg)     # paper: 2*L_fm for CQT disc
            loss_mr = mrstft(y, y_hat) * args.lam_mrstft
            loss_hf = hf_loss(y, y_hat) * args.lam_hf if args.lam_hf > 0 else 0.0
            loss_crest = crest_loss(y_hat) * args.lam_crest if args.lam_crest > 0 else 0.0
            loss_sh = sharpness_loss(y, y_hat) * args.lam_sharp if args.lam_sharp > 0 else 0.0
            if args.lam_gd > 0:
                with torch.autocast("cuda", enabled=False):   # phase angle needs fp32
                    loss_gd = gd_loss(y.float(), y_hat.float()) * args.lam_gd
            else:
                loss_gd = 0.0
            loss_es = env_stab_loss(y, y_hat) * args.lam_envstab if args.lam_envstab > 0 else 0.0
            g_all = lg_f + lg_r + lfm_f + lfm_r + loss_mel + loss_mr + loss_sh + loss_gd + loss_es + loss_hf + loss_crest
        scaler.scale(g_all).backward()
        scaler.unscale_(og)
        gn = torch.nn.utils.clip_grad_norm_(gen.parameters(), args.gen_clip)
        if torch.isfinite(gn):
            scaler.step(og)
        else:
            og.zero_grad(set_to_none=True)      # skip step on non-finite grad (protect params)
        scaler.update()
        if ema is not None:
            with torch.no_grad():
                for k, v in gen.state_dict().items():
                    ema[k].mul_(args.ema).add_(v.detach().float(), alpha=1.0 - args.ema)

        if it % 200 == 0:
            es = loss_es.item() if args.lam_envstab > 0 else 0.0
            print(f"it {it:6d} mel {loss_mel.item()/15:.3f} "
                  f"gen {(lg_f+lg_r).item():.3f} fm {(lfm_f+lfm_r).item():.3f} "
                  f"es {es:.4f} d {(ldf+ldr).item():.3f} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if it % args.eval_every == 0:
            evaluate(it)
            sd = ema if ema is not None else gen.state_dict()
            torch.save({"gen": sd, "step": it, "args": vars(args)}, ckdir / "last.pt")
            if all(torch.isfinite(v).all() for v in sd.values()):   # keep last FINITE ckpt (NaN-proof)
                torch.save({"gen": sd, "step": it, "args": vars(args)}, ckdir / "last_good.pt")
    torch.save({"gen": gen.state_dict(), "step": args.steps,
                "args": vars(args)}, ckdir / "last.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
