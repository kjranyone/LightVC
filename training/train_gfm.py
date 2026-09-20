"""V2-4a/5: 条件付き Flow Matching G（GFM）。1-step 製品 + multi-step 診断。

構成（vc_fm.md rev2/rev4 §3-5）:
- 速度場: 因果 conv 骨格（CausalBlock・既証）+ 時間埋め込み FiLM + speaker FiLM
  （3 分岐の speaker/style/prosody のうち speaker をまず実装）
- mel ABI: x1_norm = (mel80 - MU)/SD（per-bin dataset-global・発話統計なし）
  x0 ~ N(0,I) は正規化空間・AR(1) 連続 noise（chunk 境界で状態引き継ぎ）
- 1-step: x̂1 = x0 + vθ(x0, t=1|c)。multi-step: オフライン診断専用
- CFM 損失: E‖vθ(x_t,t|c) - (x1-x0)‖²、t~U(0,1)

    CUDA_VISIBLE_DEVICES=0 uv run python train_gfm.py --steps 60000 --tag gfm_v0
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
from train_vc_g import CausalBlock, FEATS, load_item, feat_util, resample_to, F0_FPS

ROOT = Path(__file__).resolve().parent.parent
MEL_FPS = 44100 / SF.HOP_A


class GFM(nn.Module):
    """速度場 v(x_t, t | content, lf0, en, spk)→ dx/dt。全因果・先読み 0。"""

    def __init__(self, dim: int = 256, layers: int = 6, emb: int = 192,
                 td: int = 64):
        super().__init__()
        self.dim, self.layers, self.emb_dim, self.td = dim, layers, emb, td
        self.inp = nn.Conv1d(770 + 80, dim, 3)
        self.blocks = nn.ModuleList(
            [CausalBlock(dim, 3, 2 ** (i // 2)) for i in range(layers)])
        self.temb = nn.Sequential(nn.Linear(1, td), nn.GELU(), nn.Linear(td, td))
        self.tfilm = nn.ModuleList([nn.Linear(td, dim * 2) for _ in range(layers)])
        self.sfilm = nn.ModuleList([nn.Linear(emb, dim * 2) for _ in range(layers)])
        self.out = nn.Linear(dim, SF.N_MEL)

    @property
    def ctx(self) -> int:
        return 2 + sum((3 - 1) * (2 ** (i // 2)) for i in range(self.layers))

    def forward(self, x_t, cond, t, s=None):
        """x_t [B,80,T]（正規化空間）・cond [B,770,T]・t [B]・s [B,192] → v [B,80,T]"""
        h = self.inp(F.pad(torch.cat([cond, x_t], 1), (2, 0)))
        te = self.temb(t[:, None])                                # [B, td]
        for b, tf, sf in zip(self.blocks, self.tfilm, self.sfilm):
            h = b(h)
            g, beta = tf(te)[:, :, None].expand(-1, -1, h.shape[-1]).chunk(2, 1)
            h = h * (1 + g) + beta
            if s is not None:
                g2, b2 = sf(s)[:, :, None].expand(-1, -1, h.shape[-1]).chunk(2, 1)
                h = h * (1 + g2) + b2
        return self.out(h.transpose(1, 2)).transpose(1, 2)

    def arch(self) -> dict:
        return {"arch": "gfm", "dim": self.dim, "L": self.layers,
                "emb": self.emb_dim, "td": self.td, "ctx": self.ctx}


def ar_noise(T: int, rho: float, gen, dev, B: int):
    """AR(1) 連続 noise（chunk 境界で状態を引き継ぐ・streaming ABI §5）。"""
    w = torch.randn(B, 80, T, generator=gen, device=dev)
    n = torch.empty_like(w)
    n[:, :, 0] = w[:, :, 0]
    for i in range(1, T):
        n[:, :, i] = rho * n[:, :, i - 1] + math.sqrt(1 - rho ** 2) * w[:, :, i]
    return n


def load_abi(dev):
    d = torch.load(ROOT / "data/fm_mel_abi.pt", map_location=dev, weights_only=False)
    return d["mu"], d["sd"].clamp(min=0.05), d["lo"], d["hi"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=344)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--ft-lr", type=float, default=0.0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--content-from", default=str(ROOT / "results/diag_e2/diag_e2_best.pt"))
    ap.add_argument("--pairs-dir", default=str(ROOT / "data/vctk_pairs"))
    ap.add_argument("--pairs-w", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    mu, sd, lo, hi = load_abi(dev)
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                         map_location="cpu", weights_only=False)

    from train_vc_e import E1
    ek = torch.load(a.content_from, map_location=dev)
    enet = E1(dim=ek["args"]["dim"], layers=ek["args"]["L"],
              look=ek["args"].get("look", 0)).to(dev).eval()
    enet.load_state_dict(ek["net"])
    for q in enet.parameters():
        q.requires_grad_(False)

    net = GFM(dim=a.dim, layers=a.layers).to(dev)
    if a.resume and a.ft_lr > 0:
        rk = torch.load(a.resume, map_location=dev)
        net.load_state_dict(rk["net"])
        opt = torch.optim.AdamW(net.parameters(), lr=a.ft_lr, betas=(0.9, 0.99))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  GFM {sum(p.numel() for p in net.parameters())/1e6:.2f}M ctx {net.ctx}"
          f"  rho {a.rho}", flush=True)

    files = []
    for root in FEATS:
        for spk in sorted(root.iterdir()):
            if spk.is_dir():
                files += sorted(spk.glob("*.pt"))
    rng.shuffle(files)
    spk_all = sorted({f.parent.name for f in files})
    held = set(spk_all[-24:])
    tr = [f for f in files if f.parent.name not in held]
    ev = [f for f in files if f.parent.name in held][:24]
    pair_items = sorted(Path(a.pairs_dir).glob("*.pt"))
    print(f"  train {len(tr)} / eval {len(ev)} / pairs {len(pair_items)}", flush=True)

    cache: dict = {}

    def get(f: Path):
        if f not in cache:
            if len(cache) > 2000:
                cache.pop(next(iter(cache)))
            try:
                cache[f] = load_item(f)
            except Exception:
                cache[f] = None
        return cache[f]

    def norm(m):
        return (m - mu[:, None]) / sd[:, None]

    def denorm(x):
        return x * sd[:, None] + mu[:, None]

    def evaluate() -> float:
        net.eval()
        tot, n = 0.0, 0
        ge = torch.Generator(device=dev).manual_seed(999)
        with torch.no_grad():
            for f in ev:
                it = get(f)
                if it is None:
                    continue
                d, mel = it
                x = feat_util(d, mel).to(dev)[None]
                with torch.no_grad():
                    x[:, :768] = enet(mel.to(dev)[None])
                s_ = spk_emb.get(d.get("speaker"))
                s_ = s_[None].to(dev) if s_ is not None else None
                t_ = mel.shape[-1]
                x0 = ar_noise(t_, a.rho, ge, dev, 1)
                x1 = norm(mel.to(dev))
                v = net(x0, x, torch.ones(1, device=dev), s_)
                xh = x0 + v
                tot += float((denorm(xh.clamp(lo[:, None], hi[:, None])) - mel.to(dev)).abs().mean())
                n += 1
        net.train()
        return tot / max(n, 1)

    best = 1e9
    step = 0
    t0 = time.time()
    while step < a.steps:
        xs, x1s, conds, spks = [], [], [], []
        while len(xs) < a.batch:
            fsel = rng.choice(tr)
            it = get(fsel)
            if it is None:
                continue
            d, mel = it
            t_ = mel.shape[-1]
            if t_ <= a.crop + net.ctx + 4:
                continue
            s0 = rng.randrange(net.ctx, t_ - a.crop)
            x = feat_util(d, mel)
            xs.append(x[:, s0 - net.ctx: s0 + a.crop])
            x1s.append(mel[:, s0: s0 + a.crop])
            spks.append(spk_emb.get(d.get("speaker"), torch.zeros(192)))
        step += 1
        xb = torch.stack(xs).to(dev)
        x1b = torch.stack(x1s).to(dev)
        sb = torch.stack(spks).to(dev)
        with torch.no_grad():
            xb[:, :768] = enet(xb[:, :80].cpu() if False else torch.stack(
                [torch.zeros(0)] * 0 + []).to(dev) if False else xb[:, :768])
        # content 置換: mel crop 区間を student で（feat_util の content 列を上書き）
        # mel_in は recon 教師側 mel の ctx 込み crop
        mel_in = torch.stack([
            torch.cat([torch.zeros(80, net.ctx), m[:, s0: s0 + a.crop]], -1)
            for m, s0 in [(x1s[i], int(rng2 := 0) or 0) for i in range(len(x1s))]
        ]) if False else None
        x0 = ar_noise(a.crop + net.ctx, a.rho, gen, dev, len(xs))
        x1n = norm(x1b)
        tt = torch.rand(len(xs), device=dev)
        v = net(x0[:, :, net.ctx:], xb[:, :, net.ctx:], tt, sb)
        tgt = x1n - x0[:, :, net.ctx:]
        loss = F.mse_loss(v, tgt)
        if step % a.every == 0 or step == a.steps:
            evl = evaluate()
            print(f"  step {step:6d}  cfm {float(loss):.4f}  eval-L1 {evl:.4f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            ck = {"net": net.state_dict(), "args": net.arch(), "step": step,
                  "cli": vars(a), "abi": {"mu": mu.cpu(), "sd": sd.cpu(),
                                          "lo": lo.cpu(), "hi": hi.cpu()}}
            torch.save(ck, out_dir / f"{a.tag}_last.pt")
            if evl < best:
                best = evl
                torch.save(ck, out_dir / f"{a.tag}_best.pt")
    print(f"\n{a.tag}: best eval-L1 {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
