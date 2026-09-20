"""S2-NSF: 信号路NSF decoderのフル学習（encoder=S1-3凍結・z空間不変）。

データ/損失はtrain_s1_f0headと同一（f0fix由来f0@100fps・logmel_l1+mrstft）。
f0掃引ゲート(G2)は学習中のbest ckptでも評価可能。

    CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
      uv run python train_s2_nsf.py --tag s2_nsf --steps 60000
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import HOP_LENGTH
from nsf_decoder import NSFCodec
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft
from train_s1_3 import build_index, load_wav

ROOT = Path(__file__).resolve().parent.parent
F0FIX = ROOT / "data/female_real_f0fix"
CTX_F = 100
LOSS_F = 48
CROP = (CTX_F + LOSS_F) * HOP_LENGTH
F0_SRC_FPS = 44100 / 512.0


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
    print(f"  train utts {len(tr)} / held {len(held)}", flush=True)

    m = NSFCodec(dev)
    params = [p for p in m.decoder.parameters() if p.requires_grad]
    print(f"  decoder params {sum(p.numel() for p in params)/1e6:.2f}M",
          flush=True)
    mels = [build_mel(nf, h, nm).to(dev) for nf, h, nm in MEL_SPECS]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, np.ndarray] = {}

    def get_wav(p: Path):
        if p not in cache:
            if len(cache) > 400:
                cache.pop(next(iter(cache)))
            try:
                x = load_wav(p)
                cache[p] = x.astype(np.float32) if len(x) > CROP + 480 else None
            except Exception:  # noqa: BLE001
                cache[p] = None
        return cache[p]

    def eval_held() -> float:
        m.decoder.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for p in held[:4]:
                x = get_wav(p)
                if x is None:
                    continue
                xt = torch.from_numpy(x[: len(x) // HOP_LENGTH * HOP_LENGTH]
                                      )[None].to(dev)
                z = m.encode(xt[None])[0]
                f0 = f0_at_100(p, xt.shape[-1], 0, dev)
                y = m.decode(z, f0[None])
                mn = min(y.shape[-1], xt.shape[-1])
                seg = slice(CTX_F * HOP_LENGTH, mn)
                tot += float(logmel_l1(y[..., seg], xt[..., seg], mels))
                n += 1
        m.decoder.train()
        return tot / max(n, 1)

    best = 1e9
    t0 = time.time()
    step = 0
    while step < a.steps:
        xs, f0s = [], []
        while len(xs) < a.batch:
            p = rng.choice(tr)
            x = get_wav(p)
            if x is None:
                continue
            s0 = rng.randrange(0, len(x) - CROP - 1)
            s0 = s0 // HOP_LENGTH * HOP_LENGTH
            xs.append(x[s0:s0 + CROP])
            f0s.append(f0_at_100(p, CROP, s0, dev))
        step += 1
        xb = torch.from_numpy(np.stack(xs)).reshape(a.batch, 1, CROP).to(dev)
        f0b = torch.stack(f0s)
        z = m.encode(xb)
        y = m.decode(z, f0b)
        seg = slice(CTX_F * HOP_LENGTH, CROP)
        loss = logmel_l1(y[..., seg], xb[..., seg], mels) + mrstft(
            y[..., seg], xb[..., seg])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if step % a.every == 0 or step == a.steps:
            evl = eval_held()
            print(f"  step {step:6d}  loss {float(loss.detach()):.4f}"
                  f"  held-mel {evl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"dec": m.decoder.state_dict(), "step": step,
                        "cli": vars(a), "held_mel": evl},
                       out_dir / f"{a.tag}_last.pt")
            if evl < best:
                best = evl
                torch.save({"dec": m.decoder.state_dict(), "step": step,
                            "cli": vars(a), "held_mel": evl},
                           out_dir / f"{a.tag}_best.pt")
    print(f"\n{a.tag}: best held-mel {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
