"""S1-1: 1発話 overfit（Phase R・左文脈付き crop・c32/c40 同一条件）。

仕様は current/causal_codec.md rev2 §4:
- crop = 左文脈 100 frame(1.0s) + loss 区間 128 frame(1.28s)
- loss (15*logmel + 2*mrstft + 1*wave_l1) は loss 区間のみ
- 発話先頭は zero-state を許可（左ゼロ埋め・loss は掛ける）
- GAN なし（Phase R）。gain aug なし（S1-1 は素の1発話）
- eval は固定発話の full zero-state pass を raw/EMA 両方で出力

    CUDA_VISIBLE_DEVICES=0 uv run python train_s1_1.py --width 32 --tag s1_1_c32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, SAMPLE_RATE, HOP_LENGTH

ROOT = Path(__file__).resolve().parent.parent
UTT = ROOT / "female-dataset/fe659435bbd284e8/fe659435bbd284e8_00005555.wav"
LOSS_FRAMES = 128
CTX_FRAMES = 100

MEL_SPECS = [(512, 160, 64), (1024, 256, 96), (2048, 480, 128)]  # (n_fft, hop, n_mels)


def build_mel(n_fft: int, hop: int, n_mels: int):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        n_mels=n_mels, f_min=0.0, f_max=24000.0, power=1.0,
        norm="slaney", mel_scale="slaney", center=True, pad_mode="reflect")


def logmel_l1(y: torch.Tensor, t: torch.Tensor, mels) -> torch.Tensor:
    loss = 0.0
    for m in mels:
        a = torch.log(m(y).clamp(min=1e-5))
        b = torch.log(m(t).clamp(min=1e-5))
        loss = loss + (a - b).abs().mean()
    return loss / len(mels)


def mrstft(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for n_fft, hop, _nm in MEL_SPECS:
        w = torch.hann_window(n_fft, device=y.device)
        def spec(x):
            return torch.stft(x.squeeze(1), n_fft, hop, n_fft, w,
                              return_complex=True).abs()
        Sy, St = spec(y), spec(t)
        sc = torch.norm(Sy - St, p="fro") / St.norm(p="fro").clamp(min=1e-8)
        lm = (torch.log(Sy.clamp(min=1e-5)) - torch.log(St.clamp(min=1e-5))).abs().mean()
        loss = loss + sc + lm
    return loss / len(MEL_SPECS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=32, choices=[32, 40])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    channels = tuple(a.width * (2 ** i) for i in range(5))     # (c,2c,4c,8c,16c)
    codec = CausalCodec(latent_dim=32, channels=channels).to(dev)
    n_par = sum(p.numel() for p in codec.parameters())
    print(f"  {a.tag}: channels {channels} params {n_par/1e6:.2f}M", flush=True)

    w, _ = librosa.load(str(UTT), sr=SAMPLE_RATE, mono=True)
    wav = torch.from_numpy(w.astype(np.float32))
    T = wav.shape[-1] // HOP_LENGTH
    pad = (CTX_FRAMES + LOSS_FRAMES) * HOP_LENGTH
    mels = [build_mel(nf, h, nm).to(dev) for nf, h, nm in MEL_SPECS]
    import soundfile

    opt = torch.optim.AdamW(codec.parameters(), lr=a.lr, betas=(0.8, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    ema = {k: v.detach().clone() for k, v in codec.state_dict().items()}
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def render_full(name: str, sd=None) -> None:
        old = None
        if sd is not None:
            old = {k: v.detach().clone() for k, v in codec.state_dict().items()}
            codec.load_state_dict(sd)
        codec.eval()
        with torch.no_grad():
            n = T * HOP_LENGTH
            x = wav[:n][None, None].to(dev)
            y = codec.decode(codec.encode(x))[0, 0].cpu().numpy()
        codec.train()
        if old is not None:
            codec.load_state_dict(old)
        soundfile.write(out_dir / name, np.clip(y, -1, 1), SAMPLE_RATE)

    g = torch.Generator().manual_seed(a.seed)
    step = 0
    t0 = time.time()
    render_full("step0_raw.wav")
    while step < a.steps:
        xs = []
        starts = torch.randint(0, T - LOSS_FRAMES, (a.batch,), generator=g)
        for s in starts.tolist():
            seg = torch.zeros(pad)
            lo = max(0, (s - CTX_FRAMES) * HOP_LENGTH)
            hi = (s + LOSS_FRAMES) * HOP_LENGTH
            take = wav[lo:hi]
            seg[-len(take):] = take
            xs.append(seg)
        xb = torch.stack(xs)[:, None].to(dev)
        yb, _zb = codec(xb)
        y_loss = yb[..., -(LOSS_FRAMES * HOP_LENGTH):]
        t_loss = xb[..., -(LOSS_FRAMES * HOP_LENGTH):]
        lm = logmel_l1(y_loss, t_loss, mels)
        ms = mrstft(y_loss, t_loss)
        wl = F.l1_loss(y_loss, t_loss)
        loss = 15 * lm + 2 * ms + 1 * wl
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
                  f"  w {float(wl):.5f}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"net": codec.state_dict(), "ema": ema,
                        "args": {"width": a.width, "channels": channels},
                        "step": step}, out_dir / f"{a.tag}_last.pt")
    render_full("final_raw.wav")
    render_full("final_ema.wav", ema)
    soundfile.write(out_dir / "gt.wav", wav.numpy(), SAMPLE_RATE)
    print(f"\n{a.tag}: done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
