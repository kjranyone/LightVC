"""S1-3: フル実音声コーパスでの codec 学習（Phase R→G・rev2 仕様）。

データ: female-dataset 全話者（48k mono・発話単位なし・左文脈付き crop）。
held 話者 24（S1-2 と同じ分割規則: 話者ソート末尾）+ namikawa を外部確認に。
Phase R (--r-steps) → Phase G (--g-steps)。crop/gain/quiet 仕様は rev2 §4.1。

    CUDA_VISIBLE_DEVICES=0 uv run python train_s1_3.py --tag s1_3_c32
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import soundfile
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, SAMPLE_RATE, HOP_LENGTH
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft, LOSS_FRAMES, CTX_FRAMES
from train_s1_g import MRD

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "female-dataset"
PAD = (CTX_FRAMES + LOSS_FRAMES) * HOP_LENGTH
MIN_SEC = 3.0


def build_index() -> tuple[list[Path], list[Path]]:
    spks = sorted([d for d in FD.iterdir() if d.is_dir()])
    held_spks = set(spks[-24:])
    tr, held = [], []
    for spk in spks:
        wavs = sorted(spk.glob("*.wav"))
        if spk in held_spks:
            held += wavs[:1]
        else:
            tr += wavs
    return tr, held


def load_wav(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SAMPLE_RATE:
        import torchaudio
        x = torchaudio.functional.resample(
            torch.from_numpy(x), sr, SAMPLE_RATE).numpy()
    return x


class QuietPicker:
    """quiet/breath 候補を 30% にする抽出器（RMS 閾値は話者混合分布から固定）。"""

    def __init__(self, paths: list[Path], rng: random.Random):
        vals = []
        for p in rng.sample(paths, min(300, len(paths))):
            try:
                x = load_wav(p)
                if len(x) > SAMPLE_RATE:
                    vals.append(float(np.sqrt((x ** 2).mean())))
            except Exception:
                continue
        vals.sort()
        self.th = vals[int(len(vals) * 0.25)] if vals else 0.02
        self.quiet: list[Path] = []
        self.loud: list[Path] = []
        for p in paths:
            (self.quiet if self._is_quiet(p) else self.loud).append(p)
        if not self.quiet:
            self.quiet = self.loud

    def _is_quiet(self, p: Path) -> bool:
        try:
            x = load_wav(p)
            return float(np.sqrt((x ** 2).mean())) < self.th and len(x) > SAMPLE_RATE
        except Exception:
            return False

    def draw(self, rng: random.Random) -> Path:
        if rng.random() < 0.3 and self.quiet:
            return self.quiet[rng.randrange(len(self.quiet))]
        return self.loud[rng.randrange(len(self.loud))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--r-steps", type=int, default=60000)
    ap.add_argument("--g-steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--phase-g-only", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)

    tr, held = build_index()
    print(f"  train utts {len(tr)} / held {len(held)}", flush=True)
    picker = QuietPicker(tr, rng)
    print(f"  quiet pool {len(picker.quiet)} / loud {len(picker.loud)}"
          f" (th {picker.th:.4f})", flush=True)

    channels = tuple(a.width * (2 ** i) for i in range(5))
    codec = CausalCodec(latent_dim=32, channels=channels).to(dev)
    print(f"  {a.tag}: {channels} {sum(p.numel() for p in codec.parameters())/1e6:.2f}M",
          flush=True)

    from bigvgan.discriminators import MultiPeriodDiscriminator
    from types import SimpleNamespace
    h = SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False,
                        discriminator_channel_mult=1)
    mpd = MultiPeriodDiscriminator(h).to(dev)
    mrd = MRD().to(dev)
    dopt = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()),
                             lr=2e-4, betas=(0.8, 0.99))
    opt = torch.optim.AdamW(codec.parameters(), lr=a.lr, betas=(0.8, 0.99))
    ema = {k: v.detach().clone() for k, v in codec.state_dict().items()}
    step0 = 0
    r_done = 0
    if a.resume:
        rk = torch.load(a.resume, map_location=dev)
        codec.load_state_dict(rk["net"])
        ema.update(rk.get("ema", {}))
        step0 = int(rk.get("step", 0))
        r_done = step0
        print(f"  resume: step {step0} ({a.resume})", flush=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, total_steps=a.r_steps + a.g_steps) if step0 == 0 else \
        torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    held_cache = []
    for p in held[:3] + [ROOT / "namikawa.mp3"]:
        w = load_wav(p)
        n = len(w) // HOP_LENGTH * HOP_LENGTH
        held_cache.append(torch.from_numpy(w[:n].copy()))
        soundfile.write(out_dir / f"held{len(held_cache)-1}_gt.wav", w[:n], SAMPLE_RATE)

    def render_held(sd=None):
        old = None
        if sd is not None:
            old = {k: v.detach().clone() for k, v in codec.state_dict().items()}
            codec.load_state_dict(sd)
        codec.eval()
        for i, w in enumerate(held_cache):
            with torch.no_grad():
                y = codec.decode(codec.encode(w[None, None].to(dev)))[0, 0].cpu().numpy()
            soundfile.write(out_dir / f"held{i}_{'ema' if sd else 'raw'}.wav",
                            np.clip(y, -1, 1), SAMPLE_RATE)
        codec.train()
        if old is not None:
            codec.load_state_dict(old)

    wcache: dict[Path, np.ndarray] = {}

    def get_wav(p: Path) -> np.ndarray:
        if p not in wcache:
            if len(wcache) > 400:
                wcache.pop(next(iter(wcache)))
            wcache[p] = load_wav(p)
        return wcache[p]

    def crop_batch():
        xs = []
        while len(xs) < a.batch:
            wav = get_wav(picker.draw(rng))
            T = wav.shape[-1] // HOP_LENGTH
            if T <= LOSS_FRAMES + 1:
                continue
            s = rng.randrange(1, T - LOSS_FRAMES)
            seg = torch.zeros(PAD)
            lo = max(0, (s - CTX_FRAMES) * HOP_LENGTH)
            take = torch.from_numpy(wav[lo:(s + LOSS_FRAMES) * HOP_LENGTH].copy())
            seg[-len(take):] = take
            if rng.random() < 0.5:
                db = rng.uniform(-30.0, 0.0)
                seg = seg * (10 ** (db / 20))
            xs.append(seg)
        return torch.stack(xs)[:, None].to(dev)

    def self_crop(x):
        n = x.shape[-1]
        cut = (n // 2730) * 2730
        return x[..., :cut]

    def loop(steps0, gan, t0, start=0):
        step = start
        while step < steps0:
            xb = crop_batch()
            yb, _ = codec(xb)
            L = LOSS_FRAMES * HOP_LENGTH
            y_loss, t_loss = yb[..., -L:], xb[..., -L:]
            lm = logmel_l1(y_loss, t_loss, mels)
            ms = mrstft(y_loss, t_loss)
            wl = F.l1_loss(y_loss, t_loss)
            loss = 15 * lm + 2 * ms + 1 * wl
            advv = torch.zeros((), device=dev)
            if gan:
                dopt.zero_grad(set_to_none=True)
                y_det = yb.detach()
                d_r, d_f, _, _ = mpd(self_crop(t_loss), self_crop(y_det))
                dl = sum(((r - 1) ** 2).mean() + (f ** 2).mean()
                         for r, f in zip(d_r, d_f))
                dr, fr = mrd(self_crop(t_loss))[0], mrd(self_crop(y_det))[0]
                dl = dl + ((dr - 1) ** 2).mean() + (fr ** 2).mean()
                dl.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(mpd.parameters()) + list(mrd.parameters()), 1.0)
                dopt.step()
                _, d_f2, fm_r, fm_f = mpd(self_crop(t_loss), self_crop(y_loss))
                advv = sum(((f - 1) ** 2).mean() for f in d_f2)
                fr2 = mrd(self_crop(y_loss))[0]
                advv = advv + ((fr2 - 1) ** 2).mean()
                fmv = sum(F.l1_loss(x.detach(), y)
                          for A, B in zip(fm_r, fm_f) for x, y in zip(A, B))
                fmrd = F.l1_loss(dr.detach(), fr2)
                loss = loss + 1.0 * advv + 2.0 * (fmv + fmrd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            opt.step()
            sch.step()
            with torch.no_grad():
                for k, v in codec.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
            step += 1
            if step % a.every == 0 or step == steps0:
                torch.cuda.empty_cache()
                print(f"  {'G' if gan else 'R'} {step:6d}  mel {float(lm):.4f}"
                      f"  mr {float(ms):.4f}"
                      + (f"  adv {float(advv):.3f}" if gan else "")
                      + f"  ({time.time()-t0:.0f}s)", flush=True)
                torch.save({"net": codec.state_dict(), "ema": ema,
                            "args": {"width": a.width}, "step": step},
                           out_dir / f"{a.tag}_last.pt")
                render_held()
                render_held(ema)

    t0 = time.time()
    if a.phase_g_only:
        loop(a.g_steps, True, t0, start=r_done)
    else:
        loop(a.r_steps, False, t0)
        loop(a.g_steps, True, t0)
    print(f"\n{a.tag}: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
