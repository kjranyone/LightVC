"""S1-2a: Phase G（MPD/MRD GAN 込み）・5-10 話者少量学習の前段 smoke。

S1-1 40k で Phase R の限界が確定（mel 0.057 でも SNR -3dB・耳ゴミ）。
本 smoke は 1 発話で GAN 追加が texture/位相を破壊せず押し上げるかだけを見る。
判別器は bigvgan MPD（periods 2,3,5,7,11=rev2 固定）+ MRD 6 段。
generator 構造変更なし（c32・rev2 ABI のまま）。

    CUDA_VISIBLE_DEVICES=0 uv run python train_s1_g.py --steps 6000 --tag s1_g_smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import OrderedDict
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, SAMPLE_RATE, HOP_LENGTH
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft, UTT, LOSS_FRAMES, CTX_FRAMES

ROOT = Path(__file__).resolve().parent.parent


class MRD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ks = (3, 5, 7, 9, 11, 13)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(1 if i == 0 else 16, 16, k, stride=2, padding=k // 2)
             for i, k in enumerate(ks)])
        self.post = torch.nn.Conv1d(16, 1, 3, padding=1)

    @staticmethod
    def crop_periods(x: torch.Tensor, periods) -> torch.Tensor:
        n = x.shape[-1]
        cut = min((n // p) * p for p in periods)
        return x[..., :cut]

    def forward(self, x):
        h = x
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.1)
            feats.append(h)
        return self.post(h), feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=str(ROOT / "results/s1_1_c32_long/s1_1_c32_long_last.pt"))
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    codec = CausalCodec(latent_dim=32, channels=tuple(a.width * (2 ** i) for i in range(5))).to(dev)
    ck = torch.load(a.resume, map_location=dev)
    codec.load_state_dict(ck["net"])
    print(f"  resume from {a.resume} (step {ck.get('step')})", flush=True)

    from bigvgan.discriminators import MultiPeriodDiscriminator
    from types import SimpleNamespace
    h = SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False, discriminator_channel_mult=1)
    mpd = MultiPeriodDiscriminator(h).to(dev)
    mrd = MRD().to(dev)
    dopt = torch.optim.AdamW(
        list(mpd.parameters()) + list(mrd.parameters()), lr=2e-4, betas=(0.8, 0.99))

    opt = torch.optim.AdamW(codec.parameters(), lr=a.lr, betas=(0.8, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    ema = {k: v.detach().clone() for k, v in codec.state_dict().items()}
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]

    w, _ = librosa.load(str(UTT), sr=SAMPLE_RATE, mono=True)
    wav = torch.from_numpy(w.astype(np.float32))
    T = wav.shape[-1] // HOP_LENGTH
    pad = (CTX_FRAMES + LOSS_FRAMES) * HOP_LENGTH
    import soundfile
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def render_full(name, sd=None):
        old = None
        if sd is not None:
            old = {k: v.detach().clone() for k, v in codec.state_dict().items()}
            codec.load_state_dict(sd)
        codec.eval()
        with torch.no_grad():
            n = T * HOP_LENGTH
            y = codec.decode(codec.encode(wav[:n][None, None].to(dev)))[0, 0].cpu().numpy()
        codec.train()
        if old is not None:
            codec.load_state_dict(old)
        soundfile.write(out_dir / name, np.clip(y, -1, 1), SAMPLE_RATE)

    PERIODS = [2, 3, 5, 7, 11]
    # 61440 = 2^10*60 → 11以外は割り切れる。crop は全 period 共通の安全長へ
    def self_crop(x):
        n = x.shape[-1]
        cut = (n // 2730) * 2730  # 2*3*5*7*13 の倍数(11を含む全periodで割り切れる長さ)
        return x[..., :cut]

    g = torch.Generator().manual_seed(a.seed)
    step = 0
    t0 = time.time()
    while step < a.steps:
        xs = []
        starts = torch.randint(0, T - LOSS_FRAMES, (a.batch,), generator=g)
        for s in starts.tolist():
            seg = torch.zeros(pad)
            lo = max(0, (s - CTX_FRAMES) * HOP_LENGTH)
            take = wav[lo:(s + LOSS_FRAMES) * HOP_LENGTH]
            seg[-len(take):] = take
            xs.append(seg)
        xb = torch.stack(xs)[:, None].to(dev)
        yb, _ = codec(xb)
        L = LOSS_FRAMES * HOP_LENGTH
        y_loss, t_loss = yb[..., -L:], xb[..., -L:]
        lm = logmel_l1(y_loss, t_loss, mels)
        ms = mrstft(y_loss, t_loss)
        wl = F.l1_loss(y_loss, t_loss)

        # discriminator step
        dopt.zero_grad(set_to_none=True)
        with torch.no_grad():
            y_det = yb.detach()
        d_real_mpd, d_fake_mpd, _, _ = mpd(self_crop(t_loss), self_crop(y_det))
        dl = sum(((r - 1) ** 2).mean() + (f ** 2).mean()
                 for r, f in zip(d_real_mpd, d_fake_mpd))
        dr, fr = mrd(self_crop(t_loss))[0], mrd(self_crop(self_crop(y_det)))[0]
        dl = dl + ((dr - 1) ** 2).mean() + (fr ** 2).mean()
        dl.backward()
        torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(mrd.parameters()), 1.0)
        dopt.step()

        # generator step (adv + feature matching)
        d_real_mpd, d_fake_mpd, fm_r, fm_f = mpd(self_crop(t_loss), self_crop(y_loss))
        adv = sum(((f - 1) ** 2).mean() for f in d_fake_mpd)
        fmv = sum(F.l1_loss(a_.detach(), b)
                  for A, B in zip(fm_r, fm_f) for a_, b in zip(A, B))
        dr2, fr2 = mrd(self_crop(t_loss))[0], mrd(self_crop(y_loss))[0]
        adv = adv + ((fr2 - 1) ** 2).mean()
        fmrd = F.l1_loss(dr2.detach(), fr2)
        loss = 15 * lm + 2 * ms + 1 * wl + 1.0 * adv + 2.0 * (fmv + fmrd)

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
        if step % a.every == 0 or step == a.steps:
            print(f"  step {step:5d}  mel {float(lm):.4f}  mr {float(ms):.4f}"
                  f"  adv {float(adv):.3f}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"net": codec.state_dict(), "ema": ema,
                        "args": {"width": a.width}, "step": step},
                       out_dir / f"{a.tag}_last.pt")
    render_full("final_raw.wav")
    render_full("final_ema.wav", ema)
    soundfile.write(out_dir / "gt.wav", wav.numpy(), SAMPLE_RATE)
    print(f"\n{a.tag}: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
