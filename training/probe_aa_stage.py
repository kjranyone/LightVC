"""倍音間雑音の発生段の切り分け(学習なし): 生成器の段 i の LeakyReLU だけを折り返し防止版(2 倍に上げ → LeakyReLU → 下げる・
非因果のゼロ位相フィルタ=診断専用)に差し替え、倍音間の谷の深さ(500–3000Hz)と PESQ の変化を測る。

    CUDA_VISIBLE_DEVICES=0 uv run python probe_aa_stage.py ../results/nvoc4/last.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).parent))
import eval_nvoc as E
import nvoc as N
import probe_contrast_lib as C


def aa_lrelu(x: torch.Tensor, s: float) -> torch.Tensor:
    u = AF.resample(x, 1, 2, lowpass_filter_width=16)
    return AF.resample(F.leaky_relu(u, s), 2, 1, lowpass_filter_width=16)[..., :x.shape[-1]]


def generate(m: N.NVoc, mel: torch.Tensor, exc: torch.Tensor, aa: set) -> torch.Tensor:
    T = mel.shape[-1]
    x = m.pre(mel)
    n = T
    for i, r in enumerate(N.UPS):
        act = (lambda z, s: aa_lrelu(z, s)) if i in aa else (lambda z, s: F.leaky_relu(z, s))
        n *= r
        x = m.up[i](act(x, 0.1))[..., :n]
        x = x + m.src[i](exc)
        outs = []
        for rb in m.res[i]:
            h = x
            for a_, b_ in zip(rb.c1, rb.c2):
                h = h + b_(act(a_(act(h, 0.1)), 0.1))
            outs.append(h)
        x = sum(outs) / len(outs)
    act = (lambda z, s: aa_lrelu(z, s)) if len(N.UPS) in aa else (lambda z, s: F.leaky_relu(z, s))
    return m.post(act(x, 0.01)).squeeze(1)


def main() -> int:
    dev = "cuda"
    st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    nosrc = len(sys.argv) > 2 and sys.argv[2] == "nosrc"
    m = N.NVoc().to(dev)
    m.load_state_dict(st["ema"])
    m.eval()
    items = sorted(E.held_items(), key=lambda it: -float((it["f0"] > 0).mean()))[:8]
    for aa in ([], [0], [1], [2], [3], [4], [0, 1, 2, 3, 4]):
        cs, ps = [], []
        for it in items:
            f0 = it["f0"] * (0 if nosrc else 1)
            g = torch.Generator(device="cpu").manual_seed(0)
            noise = torch.randn(1, len(it["x"]), generator=g).to(dev)
            xin = torch.cat([torch.zeros(1, N.WIN - N.HOP), torch.from_numpy(it["x"])[None]], -1).to(dev)
            with torch.no_grad():
                mel = m.mel_ctx(xin)
                h = N.harmonic_source(torch.from_numpy(f0)[None].to(dev)[:, :mel.shape[-1]])
                y = generate(m, mel, torch.stack([h, noise[:, :h.shape[-1]]], 1), set(aa))[0].cpu().numpy()[N.DELAY:]
            cs.append(C.contrast(y.astype(np.float64), it["f0"]))
            ps.append(E.metrics(y, it["x"], 0, dev)["pesq"])
        print(f"AA at stages {aa or 'none'}: contrast {np.mean(cs):.3f}  pesq {np.mean(ps):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
