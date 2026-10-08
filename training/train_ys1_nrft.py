"""Y-S1 decoder の潜在摂動頑健化 fine-tune(prereg: results/ys1_nrft/prereg.yaml)。

耳の事実(2026-09-23 chorus_probe): c32 decoder は潜在のずれを帯域によらず(rms~0.1でも)
コーラス/フェーザー化する。本腕は encoder を凍結し、decoder 入力 z に学習専用の摂動 δ を
足して実音声を教師に Phase G 損失(15 logmel + 2 mrstft + 1 wave L1 + adv + 2 FM)で微調整する。
推論グラフ・ABI・RTF は不変(重みのみ)。開始点は s1_3_c32 の EMA。判別器は保存が無いため
新規初期化し、最初の --d-warm step は判別器のみ更新する。

摂動(1サンプルごと・p_clean で無摂動): mode∈{low,mid,high,all,mix,hiatt}
  帯域雑音: 次元別 std(abi sd)に比例・総 rms r∈logU[r_lo,r_hi](生スケール)
  hiatt: z から自身の 16–50Hz 成分を係数 a∈U[0.1,0.5] で減衰(生成器の微細構造の出し切れなさの模擬)

    CUDA_VISIBLE_DEVICES=0 uv run python train_ys1_nrft.py --tag ys1_nrft
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, SAMPLE_RATE, HOP_LENGTH
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft, LOSS_FRAMES, CTX_FRAMES
from train_s1_g import MRD
from train_s1_3 import build_index, load_wav, PAD

ROOT = Path(__file__).resolve().parent.parent
LAT = ROOT / "data/ys1_latent"
BANDS = {"low": (0.0, 4.0), "mid": (4.0, 16.0), "high": (16.0, 50.01)}
MODES = ("low", "mid", "high", "all", "mix", "hiatt")


def band_t(x: torch.Tensor, lo: float, hi: float, fps: float = 100.0) -> torch.Tensor:
    X = torch.fft.rfft(x, dim=-1)
    f = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fps).to(x.device)
    return torch.fft.irfft(X * ((f >= lo) & (f < hi)).to(X.dtype), n=x.shape[-1], dim=-1)


def unit_noise(D: int, Fr: int, lo: float, hi: float, dev, g: torch.Generator) -> torch.Tensor:
    n = band_t(torch.randn(D, Fr, device=dev, generator=g), lo, hi)
    return n / n.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8)


def perturb(z: torch.Tensor, w_dim: torch.Tensor, a, g: torch.Generator, rng: random.Random):
    B, D, Fr = z.shape
    out = z.clone()
    modes = []
    for b in range(B):
        if rng.random() < a.p_clean:
            modes.append("clean")
            continue
        m = rng.choice(MODES)
        modes.append(m)
        if m == "hiatt":
            zc = z[b] - z[b].mean(-1, keepdim=True)
            out[b] = z[b] - rng.uniform(0.1, 0.5) * band_t(zc, *BANDS["high"])
            continue
        if m in BANDS:
            n = unit_noise(D, Fr, *BANDS[m], z.device, g)
        elif m == "all":
            n = unit_noise(D, Fr, 0.0, 50.01, z.device, g)
        else:
            w = np.random.default_rng(rng.randrange(1 << 30)).dirichlet([1.0, 1.0, 1.0])
            n = sum(math.sqrt(wk) * unit_noise(D, Fr, *BANDS[k], z.device, g)
                    for wk, k in zip(w, BANDS))
            n = n / n.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-8)
        r = math.exp(rng.uniform(math.log(a.r_lo), math.log(a.r_hi)))
        out[b] = z[b] + r * n * w_dim[:, None]
    return out, modes


def selfcheck(dev) -> None:
    g = torch.Generator(device=dev).manual_seed(0)
    sd = torch.load(LAT / "abi.pt", map_location=dev, weights_only=False)["sd"]
    w = sd / sd.pow(2).mean().sqrt()
    z = torch.zeros(2, 32, 228, device=dev)
    ns = argparse.Namespace(p_clean=0.0, r_lo=0.2, r_hi=0.2)
    for m in ("low", "mid", "high", "all", "mix"):
        zz = torch.zeros(1, 32, 228, device=dev)
        rr = random.Random(0)
        rr.choice = lambda seq, m=m: m
        out, _ = perturb(zz, w, ns, g, rr)
        rms = float(out.pow(2).mean().sqrt())
        assert abs(rms - 0.2) < 0.02, (m, rms)
        if m in BANDS:
            E = torch.fft.rfft(out[0], dim=-1).abs().pow(2).sum(0)
            f = torch.fft.rfftfreq(228, d=0.01).to(dev)
            lo, hi = BANDS[m]
            inb = float(E[(f >= lo) & (f < hi)].sum() / E.sum())
            assert inb > 0.999, (m, inb)
    print("  perturb selfcheck ok (rms=0.2±0.02・帯域内エネルギー>99.9%)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--d-warm", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--p-clean", type=float, default=0.25)
    ap.add_argument("--r-lo", type=float, default=0.04)
    ap.add_argument("--r-hi", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--tripwire", type=float, default=1.2,
                    help="joint 2k以降に held clean mel-L1 が開始時×この値を超えたら中止")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    selfcheck(dev)

    tr, held = build_index()
    print(f"  train utts {len(tr)} / held {len(held)}", flush=True)
    base = ROOT / "results/s1_3_c32/s1_3_c32_last.pt"
    ck = torch.load(base, map_location=dev, weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ck["ema"])
    for p in codec.encoder.parameters():
        p.requires_grad_(False)
    codec.encoder.eval()
    sd = torch.load(LAT / "abi.pt", map_location=dev, weights_only=False)["sd"]
    w_dim = sd / sd.pow(2).mean().sqrt()

    from bigvgan.discriminators import MultiPeriodDiscriminator
    from types import SimpleNamespace
    h = SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False,
                        discriminator_channel_mult=1)
    mpd = MultiPeriodDiscriminator(h).to(dev)
    mrd = MRD().to(dev)
    dopt = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()),
                             lr=2e-4, betas=(0.8, 0.99))
    dparams = list(codec.decoder.parameters())
    opt = torch.optim.AdamW(dparams, lr=a.lr, betas=(0.8, 0.99))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    ema = {k: v.detach().clone() for k, v in codec.state_dict().items()}
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    held_w = []
    for p in held[:3]:
        w = load_wav(p)
        n = min(len(w), 8 * SAMPLE_RATE) // HOP_LENGTH * HOP_LENGTH
        held_w.append(torch.from_numpy(w[:n].copy()).to(dev))

    def held_eval(state=None) -> dict:
        old = None
        if state is not None:
            old = {k: v.detach().clone() for k, v in codec.state_dict().items()}
            codec.load_state_dict(state)
        codec.decoder.eval()
        gg = torch.Generator(device=dev).manual_seed(123)
        clean, noisy = [], []
        with torch.no_grad():
            for w in held_w:
                z = codec.encode(w[None, None])
                y = codec.decode(z)[..., :w.shape[-1]]
                clean.append(float(logmel_l1(y[0], w[None], mels)))
                n = unit_noise(32, z.shape[-1], 0.0, 50.01, dev, gg)
                yn = codec.decode(z + 0.2 * n[None] * w_dim[None, :, None])[..., :w.shape[-1]]
                noisy.append(float(logmel_l1(yn[0], w[None], mels)))
        codec.decoder.train()
        if old is not None:
            codec.load_state_dict(old)
        return {"clean": float(np.mean(clean)), "noisy_r0.2": float(np.mean(noisy))}

    base_eval = held_eval()
    print(f"  base(held3 EMA開始点): {json.dumps(base_eval)}", flush=True)
    wcache: dict = {}

    def get_wav(p):
        if p not in wcache:
            if len(wcache) > 400:
                wcache.pop(next(iter(wcache)))
            wcache[p] = load_wav(p)
        return wcache[p]

    def crop_batch():
        xs = []
        while len(xs) < a.batch:
            wav = get_wav(tr[rng.randrange(len(tr))])
            T = wav.shape[-1] // HOP_LENGTH
            if T <= LOSS_FRAMES + 1:
                continue
            s = rng.randrange(1, T - LOSS_FRAMES)
            seg = torch.zeros(PAD)
            lo = max(0, (s - CTX_FRAMES) * HOP_LENGTH)
            take = torch.from_numpy(wav[lo:(s + LOSS_FRAMES) * HOP_LENGTH].copy())
            seg[-len(take):] = take
            if rng.random() < 0.5:
                seg = seg * (10 ** (rng.uniform(-30.0, 0.0) / 20))
            xs.append(seg)
        return torch.stack(xs)[:, None].to(dev)

    def self_crop(x):
        return x[..., :(x.shape[-1] // 2730) * 2730]

    L = LOSS_FRAMES * HOP_LENGTH
    t0 = time.time()
    total = a.d_warm + a.steps
    acc = {"lm_clean": [], "lm_pert": [], "adv": []}
    for step in range(1, total + 1):
        joint = step > a.d_warm
        xb = crop_batch()
        with torch.no_grad():
            z = codec.encode(xb)
        zp, modes = perturb(z, w_dim, a, g, rng)
        if joint:
            yb = codec.decode(zp)
        else:
            with torch.no_grad():
                yb = codec.decode(zp)
        y_loss, t_loss = yb[..., -L:], xb[..., -L:]
        dopt.zero_grad(set_to_none=True)
        y_det = y_loss.detach()
        d_r, d_f, _, _ = mpd(self_crop(t_loss), self_crop(y_det))
        dl = sum(((r - 1) ** 2).mean() + (f ** 2).mean() for r, f in zip(d_r, d_f))
        dr, fr = mrd(self_crop(t_loss))[0], mrd(self_crop(y_det))[0]
        dl = dl + ((dr - 1) ** 2).mean() + (fr ** 2).mean()
        dl.backward()
        torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(mrd.parameters()), 1.0)
        dopt.step()
        if joint:
            lm = logmel_l1(y_loss, t_loss, mels)
            ms = mrstft(y_loss, t_loss)
            wl = F.l1_loss(y_loss, t_loss)
            _, d_f2, fm_r, fm_f = mpd(self_crop(t_loss), self_crop(y_loss))
            advv = sum(((f - 1) ** 2).mean() for f in d_f2)
            fr2 = mrd(self_crop(y_loss))[0]
            advv = advv + ((fr2 - 1) ** 2).mean()
            fmv = sum(F.l1_loss(x.detach(), y) for A, B in zip(fm_r, fm_f) for x, y in zip(A, B))
            fmrd = F.l1_loss(dr.detach(), fr2)
            loss = 15 * lm + 2 * ms + 1 * wl + 1.0 * advv + 2.0 * (fmv + fmrd)
            if not torch.isfinite(loss):
                print(f"  non-finite loss at {step} -> abort", flush=True)
                return 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dparams, 1.0)
            opt.step()
            sch.step()
            with torch.no_grad():
                for k, v in codec.state_dict().items():
                    if k.startswith("decoder.") and v.dtype.is_floating_point:
                        ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
                for i, m in enumerate(modes):
                    li = float(logmel_l1(y_loss[i:i + 1], t_loss[i:i + 1], mels))
                    acc["lm_clean" if m == "clean" else "lm_pert"].append(li)
                acc["adv"].append(float(advv))
        if step % a.every == 0 or step == total:
            msg = {k: round(float(np.mean(v)), 4) for k, v in acc.items() if v}
            acc = {k: [] for k in acc}
            line = f"  {'J' if joint else 'D'} {step:6d}  dloss {float(dl):.3f}  {json.dumps(msg)}"
            if joint:
                ev = held_eval(ema)
                line += f"  held_ema {json.dumps({k: round(v, 4) for k, v in ev.items()})}"
                torch.save({"net": codec.state_dict(), "ema": ema, "args": {"width": 32},
                            "step": step, "cli": vars(a), "base": str(base.relative_to(ROOT)),
                            "base_eval": base_eval}, out_dir / f"{a.tag}_last.pt")
                if step - a.d_warm >= 2000 and ev["clean"] > base_eval["clean"] * a.tripwire:
                    print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
                    print(f"  TRIPWIRE: held clean mel-L1 {ev['clean']:.4f} > "
                          f"{a.tripwire}×base {base_eval['clean']:.4f} -> abort", flush=True)
                    return 2
            print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n{a.tag}: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
