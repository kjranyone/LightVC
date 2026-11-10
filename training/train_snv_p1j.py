"""P1 の本人らしさの判定器 J(synth_nam_vc §8.3・自前・学習の時だけの道具 = 推論の経路に入らない): 特徴の空間 Y(29)の 1.5 s の区間 → 学習の声の分類(AAM-softmax)。
学習 = P1 の学習の声(train_snv_p1.split)の普通の発話 t00〜t08 + 音素の無い発声・閉じた組の検証 = 同じ声の t09・開いた組の較正 = 検証の声(見ていない声)の EER。
P1 の学習中は重みを固定し、自前の reamp Ŷ_B が B と読まれる圧に使う。評価の物差し(ECAPA)とは別のモデル。
区間ごとの平均・分散の正規化(--norm cmvn = J2・P1c)では、区間の統計のずらし(写し + 平均のずらし)で判定器を動かせない。
    cd training && uv run --no-sync python train_snv_p1j.py --tag snv_p1_judge
    cd training && uv run --no-sync python train_snv_p1j.py --tag snv_p1c_judge --norm cmvn
入力側の PGD(--adv ε = 次元ごとの時間の標準偏差 σ_d の単位の L∞・最初の 3 割で 0 → ε・--adv_steps)で判定器の分類を頑健にする(J3)= 生成器との min-max ではない。
区間の正規化(cmvn)は区間の中でほぼ一定の次元(無声の周期性・高次)の標準偏差がほぼ 0 になり、どんな小さな摂動も拡大する(PGD で学習が崩れる・J2 が小さな δ で欺ける)。
cmvnf = 分母を max(区間の標準偏差, 0.5 σ_d)にする(平均への不変性は厳密なまま・σ_d = コーパスの時間の標準偏差)。
    cd training && uv run --no-sync python train_snv_p1j.py --tag snv_p1c_judge3 --norm cmvnf --adv 0.25
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
SEG = 300
EMB = 192
SD_FLOOR = 0.5


class Judge(nn.Module):
    def __init__(self, n_cls: int, ch: int = 256, norm: str = "none") -> None:
        super().__init__()
        self.norm = norm
        if norm == "cmvnf":
            self.register_buffer("sig", torch.ones(T.D_Y))
        layers, c_in = [], T.D_Y
        for d in (1, 2, 3, 4):
            layers += [nn.Conv1d(c_in, ch, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(ch), nn.ReLU()]
            c_in = ch
        self.body = nn.Sequential(*layers)
        self.att = nn.Conv1d(ch, 1, 1)
        self.emb = nn.Linear(2 * ch, EMB)
        self.w = nn.Parameter(torch.randn(n_cls, EMB) * 0.01)

    def embed(self, y: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm in ("cmvn", "cmvnf"):
            w = torch.ones_like(y[:, :1]) if m is None else m[:, None].to(y.dtype)
            n = w.sum(-1, keepdim=True).clamp(min=1.0)
            mu = (y * w).sum(-1, keepdim=True) / n
            sd = ((y - mu).pow(2) * w).sum(-1, keepdim=True).div(n).clamp(min=1e-4).sqrt()
            if self.norm == "cmvnf":
                sd = torch.maximum(sd, SD_FLOOR * self.sig[None, :, None])
            y = (y - mu) / sd * w
        h = self.body(y)
        logit = self.att(h)
        if m is not None:
            logit = logit.masked_fill(m[:, None] < 0.5, -1e9)
        w = logit.softmax(-1)
        mu = (h * w).sum(-1)
        sd = ((h - mu[..., None]).pow(2) * w).sum(-1).clamp(min=1e-6).sqrt()
        return F.normalize(self.emb(torch.cat([mu, sd], 1)), dim=-1)

    def logits(self, y: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
        return 30.0 * self.embed(y, m) @ F.normalize(self.w, dim=-1).T


def aam(cos30: torch.Tensor, lab: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    cos = (cos30 / 30.0).clamp(-1 + 1e-6, 1 - 1e-6)
    th = torch.acos(cos.gather(1, lab[:, None]))
    tgt = torch.cos(th + margin)
    out = cos.scatter(1, lab[:, None], tgt)
    return F.cross_entropy(30.0 * out, lab)


def crop(u: dict, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    y = u["y"]
    Tn = y.shape[1]
    if Tn <= SEG:
        out = np.zeros((T.D_Y, SEG), np.float32)
        out[:, :Tn] = y
        m = np.zeros(SEG, np.float32)
        m[:Tn] = 1.0
        return out, m
    s = rng.randint(0, Tn - SEG)
    return y[:, s:s + SEG], np.ones(SEG, np.float32)


def stack(cs: list[tuple[np.ndarray, np.ndarray]], dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(np.stack([c[0] for c in cs])).to(dev), torch.from_numpy(np.stack([c[1] for c in cs])).to(dev)


def eer(pos: np.ndarray, neg: np.ndarray) -> float:
    th = np.sort(np.concatenate([pos, neg]))
    fr = np.array([(pos < t).mean() for t in th])
    fa = np.array([(neg >= t).mean() for t in th])
    i = int(np.argmin(np.abs(fr - fa)))
    return float((fr[i] + fa[i]) / 2)


def dim_scale(items: list, n: int = 2000) -> np.ndarray:
    us = [u for _, u in items[:: max(1, len(items) // n)]]
    return np.mean([u["y"].std(1) for u in us], 0).astype(np.float32)


def pgd(J: Judge, y: torch.Tensor, m: torch.Tensor, lab: torch.Tensor, eps: torch.Tensor, steps: int) -> torch.Tensor:
    J.eval()
    e = eps[None, :, None]
    d = (torch.rand_like(y) * 2 - 1) * e
    for _ in range(steps):
        d.requires_grad_(True)
        g, = torch.autograd.grad(aam(J.logits(y + d, m), lab), d)
        d = torch.maximum(torch.minimum(d.detach() + 0.5 * e * g.sign(), e), -e)
    J.train()
    return y + d.detach() * m[:, None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--norm", choices=["none", "cmvn", "cmvnf"], default="none")
    ap.add_argument("--adv", type=float, default=0.0)
    ap.add_argument("--adv_steps", type=int, default=3)
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    torch.manual_seed(0)
    corpus = T.load_corpus()
    tr, va = T.split(sorted(corpus))
    cls = {n: i for i, n in enumerate(tr)}
    train_items = [(cls[n], u) for n in tr for u in corpus[n]["norm"][:-1] + corpus[n]["nv"]]
    held = [(cls[n], corpus[n]["norm"][-1]) for n in tr]
    print(f"classes {len(cls)} train items {len(train_items)} held {len(held)} val voices {len(va)}", flush=True)
    dev = "cuda"
    J = Judge(len(cls), norm=a.norm).to(dev)
    sig = torch.from_numpy(dim_scale(train_items)).to(dev)
    if a.norm == "cmvnf":
        J.sig.copy_(sig)
    print("dim scale (時間の標準偏差・次元ごと)", [round(float(x), 3) for x in sig], flush=True)
    opt = torch.optim.AdamW(J.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    t0 = time.time()
    for step in range(1, a.steps + 1):
        batch = [train_items[rng.randrange(len(train_items))] for _ in range(a.bs)]
        y, m = stack([crop(u, rng) for _, u in batch], dev)
        lab = torch.tensor([c for c, _ in batch], device=dev)
        if a.adv > 0:
            y = pgd(J, y, m, lab, a.adv * min(1.0, step / (0.3 * a.steps)) * sig, a.adv_steps)
        loss = aam(J.logits(y, m), lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"step {step} loss {float(loss):.3f} {(time.time() - t0) / step:.3f}s/step", flush=True)
    J.eval()
    with torch.no_grad():
        acc = []
        for i in range(0, len(held), 64):
            ch = held[i:i + 64]
            y, m = stack([crop(u, random.Random(1)) for _, u in ch], dev)
            acc += (J.logits(y, m).argmax(1).cpu().numpy() == np.array([c for c, _ in ch])).tolist()
        emb = {}
        for n in va:
            us = corpus[n]["norm"]
            emb[n] = [J.embed(*stack([crop(u, random.Random(2))], dev))[0].cpu().numpy() for u in us]
        pos, neg = [], []
        names = sorted(emb)
        for i, n in enumerate(names):
            for p in range(len(emb[n])):
                for q in range(p + 1, len(emb[n])):
                    pos.append(float(emb[n][p] @ emb[n][q]))
                for m in names[i + 1:]:
                    neg.append(float(emb[n][p] @ emb[m][0]))
    res = {"closed_set_top1_heldout_t09_1p5s": round(float(np.mean(acc)), 4), "open_set_eer_val_voices_1p5s": round(eer(np.array(pos), np.array(neg)), 4),
           "classes": len(cls), "train_items": len(train_items), "steps": a.steps, "seg_frames": SEG, "norm": a.norm, "adv_eps": a.adv, "adv_steps": a.adv_steps}
    print(json.dumps(res), flush=True)
    torch.save({"J": J.state_dict(), "classes": tr, "res": res, "norm": a.norm, "sig": sig.cpu()}, out / "judge.pt")
    (out / "judge.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
