"""S3-CODEC-F0: pitch-blind符号器 + f0条件付きNSF decoder の同時学習。

7負例の統一法則(pitch-rich zにはF0権威が奪えない)への直接的処方:
  1. decoderは励起キャリア+帯域制限マスク(zは≲50Hz包絡のみ寄与)
  2. 符号器はスクラッチ(zにpitchを載せる必然性を除去)
  3. **不変性正則化** ‖E(x)−E(shift_k(x))‖₁ — WORLDシフトペアでzの一致を強制
再構成は(z, 正しいf0)から行うため、pitchはf0経路で足り、zは包絡・質感へ特化する。

    CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
      uv run python train_s3_codec_f0.py --tag s3_codec_f0 --steps 60000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalEncoder, HOP_LENGTH
from nsf_decoder import NSFDecoder
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft
from train_s1_3 import build_index, load_wav

ROOT = Path(__file__).resolve().parent.parent
F0FIX = ROOT / "data/female_real_f0fix"
SHIFT_WAV = ROOT / "data/f0shift_wav"
CTX_F = 48
LOSS_F = 48
CROP = (CTX_F + LOSS_F) * HOP_LENGTH
F0_SRC_FPS = 44100 / 512.0
SIG_CH = (64, 96, 128, 192)


def f0_at_100(wav_path: Path, n_samples: int, s0: int, dev) -> torch.Tensor:
    fp = F0FIX / wav_path.parent.name / (wav_path.stem + ".pt")
    f0 = torch.load(fp, map_location="cpu", weights_only=False)["f0"].float()
    T = n_samples // HOP_LENGTH
    idx = ((torch.arange(T, dtype=torch.float64) + 1.0) * 100.0 / F0_SRC_FPS
           - 1.0).floor() + round(s0 / 48000.0 * F0_SRC_FPS)
    idx = idx.clamp(0, f0.shape[0] - 1).long()
    return f0[idx].to(dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam-inv", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)

    tr, held = build_index()
    tr = [p for p in tr
          if (F0FIX / p.parent.name / (p.stem + ".pt")).exists()]
    rng.shuffle(tr)
    shift_meta = json.loads((SHIFT_WAV / "meta.json").read_text()) \
        if (SHIFT_WAV / "meta.json").exists() else {}
    tr_shift = [p for p in tr if p.stem in shift_meta]
    print(f"  train utts {len(tr)} (shift-pair {len(tr_shift)}) / held {len(held)}",
          flush=True)

    enc = CausalEncoder(32, (32, 64, 128, 256, 512), (8, 5, 4, 3)).to(dev)
    dec = NSFDecoder(32, cond_ch=192, sig_ch=SIG_CH).to(dev)
    params = list(enc.parameters()) + list(dec.parameters())
    print(f"  params {sum(p.numel() for p in params)/1e6:.2f}M", flush=True)
    mels = [build_mel(nf, h, nm).to(dev) for nf, h, nm in MEL_SPECS]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, np.ndarray] = {}

    def get_wav(p: Path):
        if p not in cache:
            if len(cache) > 500:
                cache.pop(next(iter(cache)))
            try:
                x = load_wav(p)
                cache[p] = x.astype(np.float32) if len(x) > CROP + 480 else None
            except Exception:  # noqa: BLE001
                cache[p] = None
        return cache[p]

    def eval_held_and_gate() -> tuple[float, dict]:
        enc.eval(); dec.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for p in held[:4]:
                x = get_wav(p)
                if x is None:
                    continue
                cap = 192000
                xt = torch.from_numpy(x[: min(len(x), cap)
                                        // HOP_LENGTH * HOP_LENGTH])[None].to(dev)
                z = enc(xt[None])[0]
                f0 = f0_at_100(p, xt.shape[-1], 0, dev)
                y = dec(z, f0[None])
                mn = min(y.shape[-1], xt.shape[-1])
                seg = slice(CTX_F * HOP_LENGTH, mn)
                tot += float(logmel_l1(y[..., seg], xt[..., seg], mels))
                n += 1
        gate = {}
        if n:
            p = held[0]
            x = get_wav(p)
            if x is not None:
                cap = 192000
                xt = torch.from_numpy(x[: min(len(x), cap)
                                        // HOP_LENGTH * HOP_LENGTH])[None].to(dev)
                z = enc(xt[None])[0]
                f0 = f0_at_100(p, xt.shape[-1], 0, dev)
                f0v = float(f0[f0 > 0].median()) if (f0 > 0).any() else 0.0
                meds = {}
                for st in (0.0, 7.0, 12.0):
                    fs = torch.where(f0 > 0, f0 * 2.0 ** (st / 12.0), f0)
                    torch.manual_seed(7)
                    y = dec(z, fs[None])[0, 0].detach().cpu().numpy()
                    wf = ROOT / "results/diag_cfm_audit"
                    import librosa as _lb
                    import pyworld as _pw
                    w44 = _lb.resample(y.astype(np.float64), orig_sr=48000,
                                       target_sr=44100)
                    f0o, t_ = _pw.harvest(w44, 44100, f0_floor=65, f0_ceil=1000,
                                          frame_period=512 / 44100 * 1000)
                    f0o = _pw.stonemask(w44, f0o, t_, 44100)
                    v = f0o[f0o > 60]
                    meds[f"st{int(st)}"] = float(np.median(v)) if len(v) else 0.0
                b = meds["st0"]
                if b and meds["st7"] and meds["st12"]:
                    gate = {"true": round(f0v, 1),
                            "st0": round(b, 1),
                            "sweep7": round(12 * np.log2(meds["st7"] / b), 2),
                            "sweep12": round(12 * np.log2(meds["st12"] / b), 2)}
        enc.train(); dec.train()
        return tot / max(n, 1), gate

    best = 1e9
    t0 = time.time()
    step = 0
    while step < a.steps:
        xs, f0s, xs2 = [], [], []
        while len(xs) < a.batch:
            pool = tr_shift if rng.random() < 0.5 and tr_shift else tr
            p = rng.choice(pool)
            x = get_wav(p)
            if x is None:
                continue
            s0 = rng.randrange(0, len(x) - CROP - 1)
            s0 = s0 // HOP_LENGTH * HOP_LENGTH
            seg0 = x[s0:s0 + CROP]
            loss_region = seg0[CTX_F * HOP_LENGTH:]
            if float(np.sqrt((loss_region ** 2).mean())) < 1e-3:
                continue
            xs.append(seg0)
            f0s.append(f0_at_100(p, CROP, s0, dev))
            if pool is tr_shift:
                x2 = get_wav(SHIFT_WAV / p.parent.name / (p.stem + ".wav"))
                if x2 is not None and len(x2) > s0 + CROP:
                    xs2.append(x2[s0:s0 + CROP])
                else:
                    xs2.append(xs[-1])
            else:
                xs2.append(xs[-1])
        step += 1
        xb = torch.from_numpy(np.stack(xs)).reshape(a.batch, 1, CROP).to(dev)
        x2b = torch.from_numpy(np.stack(xs2)).reshape(a.batch, 1, CROP).to(dev)
        f0b = torch.stack(f0s)
        z = enc(xb)
        with torch.no_grad():
            z2 = enc(x2b)
        y = dec(z, f0b)
        seg = slice(CTX_F * HOP_LENGTH, CROP)
        loss = 15 * logmel_l1(y[..., seg], xb[..., seg], mels) + 2 * mrstft(
            y[..., seg], xb[..., seg]) + 1 * torch.nn.functional.l1_loss(
            y[..., seg], xb[..., seg]) + a.lam_inv * (z - z2).abs().mean()
        if not torch.isfinite(loss) or float(loss.detach()) > 1e6:
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if step % a.every == 0 or step == a.steps:
            evl, gate = eval_held_and_gate()
            print(f"  step {step:6d}  loss {float(loss.detach()):.4f}"
                  f"  held-mel {evl:.4f}  gate {gate}  ({time.time()-t0:.0f}s)",
                  flush=True)
            ck = {"enc": enc.state_dict(), "dec": dec.state_dict(),
                  "step": step, "cli": vars(a), "held_mel": evl, "gate": gate}
            torch.save(ck, out_dir / f"{a.tag}_last.pt")
            if evl < best:
                best = evl
                torch.save(ck, out_dir / f"{a.tag}_best.pt")
    print(f"\n{a.tag}: best held-mel {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
