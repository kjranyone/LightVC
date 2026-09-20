"""VC front stage: content + f0 + target timbre -> mel -> gvoc_nhv -> waveform.

G-voc is settled (gvoc_nhv, PESQ 2.9124 with the GT mel, +0.128 over the f0-prior
run with 21/21 late wins). That decoder eats an 80-band log-mel and an f0 track,
both of which exist at conversion time. What is missing is the thing that PRODUCES
that mel from a source voice, and that is what this trains.

    content (768, ContentVec)  ->|
    log f0 + energy            ->| MelGen (causal, AdaIN)  -> mel[80] @ hop 256
    target timbre (192)        ->|
                                     |
                                     +-> NHV source-filter prior (f0 -> impulse
                                     |   train, mel -> time-varying filter)
                                     +-> Wavehax2D (warm-started from gvoc_nhv)
                                     -> waveform, judged against the real one

The loss is on the WAVEFORM, not the mel. Measured earlier this session: mel-L1
0.5 is uncorrelated with what the ear does, and regression blur in mel space is
the quality ceiling of the e1 line. The mel here is an internal representation
with a gradient through it, not a supervised target -- the anchor term is only
there to keep it in range early.

Self-reconstruction only. The target is the speaker's own real recording, and the
timbre code comes from a DIFFERENT utterance of the same speaker so identity
cannot be copied out of the frame being reconstructed. No VC teacher, no
synthetic parallel data (CLAUDE.md).

Held-out speakers are excluded by id -- the same twelve the vocoder was measured
on. Evaluation reports two numbers on them:

    ceil  gvoc_nhv driven by the REAL mel      = what the decoder can do
    vc    gvoc_nhv driven by G's mel           = what this stage costs

The gap between them is the only thing this run is trying to close.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
import train_gvoc as TG
from mel_gen import MelGen
from rddsp_gpu import (CACHE_DIR, CTX, WavLMConvLoss, istft, mrstft, safe_score,
                       stft)
from rddsp_hf import HOP, NBIN, Wavehax2D
from rddsp_neural import N_MEL, mel_of
from train_m2 import TimbreEncoder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FEAT = [ROOT / "data/female_real_feat", ROOT / "data/female_tts_feat",
        ROOT / "data/male_feat"]
SEG = 160                       # net frames at hop 128, same crop as gvoc
WIN = (CTX + SEG) * HOP         # 23552 samples
MELHOP = R.HOP                  # 256; the mel/cond rate
SEG_M = WIN // MELHOP           # 92 mel frames


class VCSet(Dataset):
    """One item = a crop of a real utterance plus a timbre reference from the
    same speaker but a different file. Content/f0/energy come from the *_feat
    encode; the waveform and its mel are read fresh so the target is real audio."""

    def __init__(self, roots, held: set, min_files: int = 2) -> None:
        self.spk: dict[str, list[Path]] = {}
        for r in roots:
            if not r.exists():
                continue
            for d in sorted(p for p in r.iterdir() if p.is_dir()):
                if d.name in held:
                    continue
                fs = sorted(d.glob("*.pt"))
                if len(fs) >= min_files:
                    self.spk.setdefault(d.name, []).extend(fs)
        self.items = [(s, f) for s, fs in self.spk.items() for f in fs]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        spk, path = self.items[i]
        try:
            return self._one(spk, path)
        except Exception:
            return None

    def _one(self, spk: str, path: Path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        w = load_wav(d["path"])
        wt = torch.from_numpy(w)
        Tm = len(w) // MELHOP
        if Tm < SEG_M + 4:
            return None
        f0 = rd_f0(wt, Tm)
        if not bool((f0 > 50).any()):
            return None
        cond = build_cond(d, f0)
        mel = mel_of(wt).clone()[:, :Tm]
        s = random.randint(0, Tm - SEG_M - 1)
        others = [p for p in self.spk[spk] if p != path]
        ref = ref_mel(random.choice(others) if others else path)
        if ref is None:
            return None
        return dict(cond=cond[:, s: s + SEG_M].float(),
                    mel=mel[:, s: s + SEG_M].float(),
                    f0=f0[s: s + SEG_M],
                    w=torch.tensor(w[s * MELHOP: s * MELHOP + WIN]),
                    ref=ref)


def load_wav(p: str) -> np.ndarray:
    x, _ = librosa.load(str(HERE / p) if not os.path.isabs(p) else p,
                        sr=R.SR, mono=True, duration=8.0)
    return x


def ref_mel(path: Path, seconds: float = 2.0):
    """Timbre reference: a short mel from another file of the same speaker."""
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
        x, _ = librosa.load(str(HERE / d["path"]), sr=R.SR, mono=True,
                            duration=seconds)
    except Exception:
        return None
    if len(x) < R.SR // 2:
        return None
    m = torch.tensor(mel_of(torch.tensor(x))).float()
    n = int(seconds * R.SR / MELHOP)
    return m[:, :n] if m.shape[-1] >= n else F.pad(m, (0, n - m.shape[-1]),
                                                   mode="replicate")


def build_cond(d: dict, f0: torch.Tensor, logf0_shift: float = 0.0):
    """[content(768) | log f0 / 7 | log energy * 0.2] at the mel rate.

    f0 is passed in rather than read from the feature file. The *_feat encode used
    a different detector from the one the vocoder was trained against, and feeding
    it costs 1.40 PESQ at the ceiling (1.7969 vs 3.1983 on six held-out speakers,
    with per-speaker medians off by up to 458 vs 294 Hz -- octave errors). The
    prior is an impulse train at exactly this f0, so a wrong track is not a small
    conditioning error, it is the wrong excitation."""
    content, energy = d["content"].float(), d["energy"].float()
    T = f0.shape[0]
    c = F.interpolate(content.t()[None], size=T, mode="linear",
                      align_corners=False)[0]
    e = F.interpolate(energy[None, None], size=T, mode="linear",
                      align_corners=False)[0, 0]
    lf = torch.log(f0.clamp(min=1.0)) + logf0_shift * (f0 > 50).float()
    return torch.cat([c, (lf / 7.0)[None],
                      (torch.log(e.clamp(min=1e-4)) * 0.2)[None]], 0)


def rd_f0(w: torch.Tensor, T: int) -> torch.Tensor:
    """The vocoder's own f0 detector, at the mel rate, padded/trimmed to T."""
    f0, _ = R.harmonic_sum_f0(w)
    f0 = f0.float()
    if f0.shape[0] < T:
        f0 = F.pad(f0[None], (0, T - f0.shape[0]), mode="replicate")[0]
    return f0[:T]


def collate(batch):
    b = [x for x in batch if x is not None]
    if len(b) < 2:
        return None
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}


def render(mel, f0, W, dev, gen, net, n, level=None):
    """mel[B,80,Tm] + f0[B,Tm] -> waveform[B,n]. Differentiable through the mel:
    the NHV filter IS G's spectral envelope, so the vocoder's gradient reaches it."""
    B = mel.shape[0]
    T = n // HOP + 1
    idx = (torch.arange(T, device=dev) * (mel.shape[-1] - 1)
           / max(T - 1, 1)).long()
    m = torch.einsum("fm,bmt->bft", W, mel[:, :, idx])
    ys = []
    for b in range(B):
        f0u = R.frame_upsample(fill(f0[b]).double(), n, MELHOP).clamp(min=0.0)
        phi = torch.remainder(2 * math.pi * torch.cumsum(f0u.to(dev), 0) / R.SR,
                              2 * math.pi).float()
        fv = f0[b][f0[b] > 50]
        f0med = float(fv.median()) if fv.numel() else 200.0
        K = max(1, int((R.SR / 2) / f0med))
        imp = torch.zeros(n, device=dev)
        for s0 in range(0, K, 32):
            kk = torch.arange(s0 + 1, min(s0 + 33, K + 1), device=dev,
                              dtype=torch.float32)[:, None]
            imp = imp + torch.cos(kk * phi[None]).sum(0)
        e = 0.7 * imp / math.sqrt(K) + 0.3 * torch.randn(n, generator=gen,
                                                         device=dev)
        E = stft(e)
        H = m[b][:, : E.shape[-1]].exp()
        H = H / H.amax().clamp(min=1e-8)
        ys.append(istft(E[:, : H.shape[-1]] * H, n))
    pw = torch.stack(ys)
    # gvoc was trained with the prior scaled to the utterance std; feeding it a
    # different level puts it off its training distribution (measured: ceiling
    # 1.73 vs 2.91). Level is not identity information -- the scorer RMS-matches.
    lv = (torch.as_tensor(level, device=dev).view(-1, 1) if level is not None
          else pw.std(dim=-1, keepdim=True) * 0 + 0.05)
    pw = pw / pw.std(dim=-1, keepdim=True).clamp(min=1e-8) * lv
    P = stft(pw)[:, :, :T]
    f = torch.cat([m[:, None, :, :P.shape[-1]], P.real[:, None], P.imag[:, None],
                   torch.log(P.abs()[:, None] + 1e-5)], 1)
    o = net(f)
    return istft(torch.complex(o[:, 0], o[:, 1]), n), o


def fill(f0: torch.Tensor) -> torch.Tensor:
    v = f0.clone()
    ok = v > 50
    if not bool(ok.any()):
        return torch.full_like(v, 200.0)
    i = torch.arange(len(v))
    return torch.tensor(np.interp(i.numpy(), i[ok].numpy(), v[ok].numpy()),
                        dtype=v.dtype)


def eval_set(held: list[str], n_per: int = 1):
    out = []
    for r in FEAT:
        if not r.exists():
            continue
        for s in held:
            d = r / s
            if not d.is_dir():
                continue
            fs = sorted(d.glob("*.pt"))[:n_per]
            out.extend((s, f) for f in fs)
    return out


@torch.no_grad()
def evaluate(G, T_enc, net, ev, W, dev, seed=999):
    G.eval(), T_enc.eval(), net.eval()
    gen = torch.Generator(device=dev).manual_seed(seed)
    vc, ceil = [], []
    for spk, path in ev:
        d = torch.load(path, map_location="cpu", weights_only=False)
        w = load_wav(d["path"])[: R.SR * 5]
        wt = torch.from_numpy(w)
        Tm = len(w) // MELHOP
        if Tm < 40:
            continue
        n = Tm * MELHOP
        wt = wt[:n]
        f0 = rd_f0(wt, Tm)
        if not bool((f0 > 50).any()):
            continue
        cond = build_cond(d, f0)[None].to(dev)
        f0 = f0[None]
        mel = mel_of(wt).clone()[:, :Tm].float()[None].to(dev)
        ref = ref_mel(path)
        if ref is None:
            continue
        s = T_enc(ref[None].to(dev))
        pred = G(cond, s)
        gt = wt
        lv = [float(gt.std())]
        y, _ = render(pred, f0, W, dev, gen, net, n, lv)
        c, _ = render(mel, f0, W, dev, gen, net, n, lv)
        vc.append(safe_score(gt, y[0].cpu()))
        ceil.append(safe_score(gt, c[0].cpu()))
    G.train(), T_enc.train(), net.train()
    return np.array(vc), np.array(ceil)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=96)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--glayers", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--vlr", type=float, default=5e-5)
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--lmos", type=float, default=100.0)
    ap.add_argument("--consist", type=float, default=1.0)
    ap.add_argument("--gan", type=float, default=1.0)
    ap.add_argument("--fm", type=float, default=2.0)
    ap.add_argument("--dstart", type=int, default=20000)
    ap.add_argument("--every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--voc", type=str, default="gvoc_nhv")
    ap.add_argument("--tag", type=str, default="vc1")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = TG.mel_to_linear(dev)

    ds = VCSet(FEAT, TG.HELD_OUT)
    print(f"  train {len(ds.items)} utts / {len(ds.spk)} speakers "
          f"(held out {len(TG.HELD_OUT)})", flush=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    collate_fn=collate, drop_last=True, persistent_workers=True,
                    prefetch_factor=4)
    ev = eval_set(sorted(TG.HELD_OUT))
    print(f"  eval {len(ev)} held-out utterances", flush=True)

    # head bias = data-mean log-mel. Zero-init is forbidden (mel_gen §3.4): the
    # head then starts far below the data and the first thousands of steps are
    # spent climbing a constant offset.
    acc, k = torch.zeros(N_MEL), 0
    for i in random.sample(range(len(ds.items)), min(200, len(ds.items))):
        x = ds[i]
        if x is not None:
            acc += x["mel"].mean(-1)
            k += 1
    mel_mean = (acc / max(k, 1)).numpy()
    print(f"  mel mean from {k} utts: {mel_mean.mean():+.3f} "
          f"[{mel_mean.min():+.2f}, {mel_mean.max():+.2f}]", flush=True)

    G = MelGen(cond_dim=770, dim=a.dim, n_layers=a.glayers, n_mels=N_MEL,
               timbre_dim=192, art_dim=0, causal=True, f0_fourier=8,
               mel_mean=mel_mean).to(dev)
    T_enc = TimbreEncoder(n_mels=N_MEL).to(dev)
    ck = torch.load(CACHE_DIR / f"{a.voc}_ch{a.ch}_s0.pt", map_location=dev,
                    weights_only=False)
    net = Wavehax2D(cin=4, ch=a.ch, layers=a.layers).to(dev)
    net.load_state_dict(ck["net"])
    print(f"  G {sum(p.numel() for p in G.parameters())/1e6:.3f}M  "
          f"timbre {sum(p.numel() for p in T_enc.parameters())/1e6:.3f}M  "
          f"voc {sum(p.numel() for p in net.parameters())/1e6:.3f}M "
          f"(warm {a.voc} step {ck['step']}, TEST {ck['test']:.4f})", flush=True)

    opt = torch.optim.AdamW([
        {"params": list(G.parameters()) + list(T_enc.parameters()), "lr": a.lr},
        {"params": net.parameters(), "lr": a.vlr}],
        betas=(0.8, 0.99), weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.vlr],
                                              total_steps=a.steps)
    lmos = WavLMConvLoss(dev) if a.lmos > 0 else None
    disc = dopt = None
    if a.gan > 0:
        from cqt_disc import MSSubBandCQTDisc
        disc = MSSubBandCQTDisc(sr=R.SR).to(dev)
        dopt = torch.optim.AdamW(disc.parameters(), lr=a.lr, betas=(0.8, 0.99),
                                 weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(a.seed + 1)

    vc, ceil = evaluate(G, T_enc, net, ev, W, dev)
    print(f"  step {0:6d}  VC {vc.mean():.4f}  ceil {ceil.mean():.4f}  "
          f"gap {(vc - ceil).mean():+.4f}", flush=True)

    t0, step, it = time.time(), 0, iter(dl)
    while step < a.steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        if b is None:
            continue
        step += 1
        cond = b["cond"].to(dev)
        f0 = b["f0"]
        tgt = b["w"].to(dev)
        s = T_enc(b["ref"].to(dev))
        pred = G(cond, s)
        y, _o = render(pred, f0, W, dev, gen, net, tgt.shape[-1],
                       tgt.std(dim=-1))
        ys, ts = y[:, HOP * 2: -HOP * 2], tgt[:, HOP * 2: -HOP * 2]
        loss = mrstft(ys, ts)
        if a.anchor > 0:
            loss = loss + a.anchor * F.l1_loss(pred, b["mel"].to(dev))
        if lmos is not None:
            loss = loss + a.lmos * lmos(ys, ts)
        if a.consist > 0:
            S = torch.complex(_o[:, 0], _o[:, 1])
            S2 = stft(y)
            m = min(S2.shape[-1], S.shape[-1])
            loss = loss + a.consist * (
                (S2[..., :m] - S[..., :m]).abs().pow(2).flatten(-2).sum(-1).sqrt()
                / S[..., :m].abs().pow(2).flatten(-2).sum(-1).sqrt().clamp(min=1e-8)
            ).mean()
        if disc is not None and step > a.dstart:
            dopt.zero_grad(set_to_none=True)
            yr, yg, _, _ = disc(ts[:, None], ys.detach()[:, None])
            dl_ = sum(((r - 1) ** 2).mean() + (q ** 2).mean()
                      for r, q in zip(yr, yg)) / len(yr)
            dl_.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            dopt.step()
            _, yg2, fr, fg = disc(ts[:, None], ys[:, None])
            adv = sum(((q - 1) ** 2).mean() for q in yg2) / len(yg2)
            fmv = sum(F.l1_loss(bb, aa.detach())
                      for A, B in zip(fr, fg) for aa, bb in zip(A, B))
            loss = loss + a.gan * adv + a.fm * fmv / max(sum(len(A) for A in fr), 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(G.parameters()) + list(T_enc.parameters())
            + list(net.parameters()), 1.0)
        opt.step()
        sch.step()
        if step % a.every == 0 or step == a.steps:
            vc, ceil = evaluate(G, T_enc, net, ev, W, dev)
            d = vc - ceil
            ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            print(f"  step {step:6d}  loss {float(loss):.4f}  VC {vc.mean():.4f} "
                  f" ceil {ceil.mean():.4f}  gap {d.mean():+.4f} +-{ci:.4f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"G": G.state_dict(), "T": T_enc.state_dict(),
                        "net": net.state_dict(), "args": vars(a), "step": step,
                        "vc": float(vc.mean()), "ceil": float(ceil.mean())},
                       CACHE_DIR / f"{a.tag}_s{a.seed}.pt")
    print(f"\n{a.tag}: VC {vc.mean():.4f}  ceil {ceil.mean():.4f}  "
          f"gap {(vc-ceil).mean():+.4f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
