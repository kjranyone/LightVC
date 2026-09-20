"""S1-2: 5-10 話者・少量コーパス Phase R→G（汎化の最初の検証）。

rev2 §4.1 仕様: 左文脈 1.0s + loss 1.28s。gain aug は適用確率 0.5・
一様 [-30,0]dB。quiet/breath 候補 30%。Phase R (R_STEP) → Phase G (G_STEP)。
固定 held 発話 3 つ（無加工）を every 毎に再構成して経過 wav 保存。

    CUDA_VISIBLE_DEVICES=0 uv run python train_s1_2.py --tag s1_2_c32
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, SAMPLE_RATE, HOP_LENGTH
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft, LOSS_FRAMES, CTX_FRAMES

ROOT = Path(__file__).resolve().parent.parent
GAIN_AUG = __import__("os").environ.get("GAIN_AUG", "1") == "1"
FD = ROOT / "female-dataset"
SPEAKERS = ["0005e65d3f11f99d", "00218f323fbaddbf", "0025e9516c36b547",
            "0040664b2efd368c", "004c6b19ad8cd958", "007c9e6d5c047b7d",
            "007fc1fa0d86bdbc", "000883f1d8ffe583"]
HELD_SPK = "000883f1d8ffe583"          # 8 話者中 1 を held に
PAD = (CTX_FRAMES + LOSS_FRAMES) * HOP_LENGTH


def load_utts() -> tuple[list[tuple[np.ndarray, str]], list[str]]:
    import soundfile as sf
    tr, names = [], []
    for spk in SPEAKERS:
        d = FD / spk
        wavs = sorted(d.glob("*.wav"))[:6]
        for w in wavs:
            x, _ = sf.read(str(w), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if x.shape[-1] < SR_MIN:
                continue
            if librosa.get_samplerate(str(w)) != SAMPLE_RATE:
                import torchaudio
                x = torchaudio.functional.resample(
                    torch.from_numpy(x), librosa.get_samplerate(str(w)),
                    SAMPLE_RATE).numpy()
            if spk == HELD_SPK:
                names.append(str(w))
                continue
            tr.append((x, spk))
    held = names[:3]
    return tr, held


SR_MIN = 3 * SAMPLE_RATE


class MRD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ks = (3, 5, 7, 9, 11, 13)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(1 if i == 0 else 16, 16, k, stride=2, padding=k // 2)
             for i, k in enumerate(ks)])
        self.post = torch.nn.Conv1d(16, 1, 3, padding=1)

    def forward(self, x):
        h = x
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.1)
        return self.post(h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--r-steps", type=int, default=15000)
    ap.add_argument("--g-steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)

    tr, held = load_utts()
    print(f"  train utts {len(tr)} / held {len(held)}", flush=True)
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
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr,
                                              total_steps=a.r_steps + a.g_steps)
    ema = {k: v.detach().clone() for k, v in codec.state_dict().items()}
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    held_wavs = []
    for p in held:
        w, _ = librosa.load(p, sr=SAMPLE_RATE, mono=True)
        n = len(w) // HOP_LENGTH * HOP_LENGTH
        held_wavs.append(torch.from_numpy(w[:n].astype(np.float32)))
        soundfile.write(out_dir / f"held{len(held_wavs)-1}_gt.wav",
                        w[:n], SAMPLE_RATE)

    def render_held(sd=None):
        old = None
        if sd is not None:
            old = {k: v.detach().clone() for k, v in codec.state_dict().items()}
            codec.load_state_dict(sd)
        codec.eval()
        for i, w in enumerate(held_wavs):
            with torch.no_grad():
                y = codec.decode(codec.encode(w[None, None].to(dev)))[0, 0].cpu().numpy()
            soundfile.write(out_dir / f"held{i}_{'ema' if sd else 'raw'}.wav",
                            np.clip(y, -1, 1), SAMPLE_RATE)
        codec.train()
        if old is not None:
            codec.load_state_dict(old)

    def crop_batch():
        xs = []
        while len(xs) < a.batch:
            wav, _spk = tr[rng.randrange(len(tr))]
            T = wav.shape[-1] // HOP_LENGTH
            if T <= LOSS_FRAMES + 1:
                continue
            s = rng.randrange(1, T - LOSS_FRAMES)
            seg = torch.zeros(PAD)
            lo = max(0, (s - CTX_FRAMES) * HOP_LENGTH)
            take = torch.from_numpy(wav[lo:(s + LOSS_FRAMES) * HOP_LENGTH].copy())
            seg[-len(take):] = take
            if GAIN_AUG and rng.random() < 0.5:
                db = rng.uniform(-30.0, 0.0)
                seg = seg * (10 ** (db / 20))
            xs.append(seg)
        return torch.stack(xs)[:, None].to(dev)

    def self_crop(x):
        n = x.shape[-1]
        cut = (n // 2730) * 2730
        return x[..., :cut]

    def phase_loop(steps0, gan: bool, t0):
        step = 0
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
                dr, fr = mrd(self_crop(t_loss)), mrd(self_crop(y_det))
                dl = dl + ((dr - 1) ** 2).mean() + (fr ** 2).mean()
                dl.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(mpd.parameters()) + list(mrd.parameters()), 1.0)
                dopt.step()
                _, d_f2, fm_r, fm_f = mpd(self_crop(t_loss), self_crop(y_loss))
                advv = sum(((f - 1) ** 2).mean() for f in d_f2)
                fr2 = mrd(self_crop(y_loss))
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
    phase_loop(a.r_steps, False, t0)
    phase_loop(a.g_steps, True, t0)
    print(f"\n{a.tag}: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
