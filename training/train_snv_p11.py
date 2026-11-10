"""P11(synth_nam_vc §8.3 rev5・原点): 話者 = 回路・操作者もまた回路・変換 = 回路の差し替えを、そのまま生成のモデルにして学ぶ。
y = C(a, lf, 有声, S)・演奏 a は話者に依らない事前分布(1 フレームごとの標準正規)・構造 S は話者の参照(登録)から。
演奏の取り出しは操作者の回路の逆 E(y, S_操作者)→ q(a | y, S)= 正規(μ, σ)。E は操作者自身の参照を読み(注意・FiLM)、操作者の回路で操作者の音を説明して外す(de-embed)。
C は P3 と同じ表の型 x(鍵 = 参照のフレームの E の演奏の平均・値 = 参照の Y)で、目標の回路で演奏を鳴らす(embed)。
学習 = 自己の reamp の ELBO: 再現の負の対数尤度(ラプラス・次元ごとの尺度を学ぶ)+ β KL(q‖N(0, I))(β は最初の 5k step で 0 → 1)。
E の入力は Y と有声だけ(高さ lf は C に明示で渡す・E には渡さない = P1 と同じ)。参照は同じ声の別の普通の発話 2 本(Batches)。
事前登録 = results/snv_p11/prereg.yaml。
    cd training && uv run --no-sync python train_snv_p11.py --tag snv_p11_r --prereg results/snv_p11/prereg.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
B_INIT = 0.05
BETA_RAMP = 5000


class EncS(nn.Module):
    def __init__(self, ch: int = 256, zdim: int = 256, heads: int = 4, dk: int = 64, dv: int = 16) -> None:
        super().__init__()
        self.heads, self.dk, self.dv = heads, dk, dv
        self.inp = nn.Conv1d(T.D_X, ch, 1)
        self.blocks = nn.ModuleList([T.GBlock(ch, d, zdim) for d in (1, 2, 4, 8, 16, 32) * 2])
        self.head = nn.Conv1d(ch, 2 * T.D_A, 1)
        self.genc = nn.Sequential(nn.Conv1d(T.D_X, ch, 5, padding=2), nn.ReLU(), nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(), nn.Conv1d(ch, ch, 5, padding=2))
        self.gz = nn.Linear(ch, zdim)
        self.kenc = nn.Sequential(nn.Conv1d(T.D_X, ch, 3, padding=1), nn.ReLU(), nn.Conv1d(ch, heads * dk, 1))
        self.venc = nn.Sequential(nn.Conv1d(T.D_X, ch, 3, padding=1), nn.ReLU(), nn.Conv1d(ch, heads * dv, 1))
        self.q = nn.Linear(ch, heads * dk)
        self.o = nn.Linear(heads * dv, ch)

    def structure(self, rx: torch.Tensor, rmask: torch.Tensor) -> dict:
        w = rmask[:, None].float()
        z = self.gz((self.genc(rx) * w).sum(-1) / w.sum(-1).clamp(min=1.0))
        B, Tr = rmask.shape
        k = self.kenc(rx).view(B, self.heads, self.dk, Tr).transpose(-1, -2)
        v = self.venc(rx).view(B, self.heads, self.dv, Tr).transpose(-1, -2)
        return {"z": z, "k": k, "v": v, "rmask": rmask}

    def forward(self, x: torch.Tensor, s: dict) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.inp(x)
        for i, blk in enumerate(self.blocks):
            h = blk(h, s["z"])
            if i == 3:
                B, C, L = h.shape
                q = self.q(h.transpose(1, 2)).view(B, L, self.heads, self.dk).transpose(1, 2)
                ctx = T.ref_attention(q, s["k"], s["v"], s["rmask"]).transpose(1, 2).reshape(B, L, self.heads * self.dv)
                h = h + self.o(ctx).transpose(1, 2)
        mu, lv = self.head(h).chunk(2, 1)
        return mu, lv.clamp(-10.0, 4.0)


def xin(y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.cat([y, v[:, None]], 1)


def lift(y: torch.Tensor, k: int, m: torch.Tensor | None) -> torch.Tensor:
    if not k:
        return y
    return torch.cat([y[:, :k], m.to(y)[None, k:25, None].expand(y.shape[0], 25 - k, y.shape[2]), y[:, 25:]], 1)


def structures(E: EncS, C: T.Circ, ry: torch.Tensor, rv: torch.Tensor, rmask: torch.Tensor) -> tuple[dict, dict]:
    rx = xin(ry, rv)
    sE = E.structure(rx, rmask)
    sC = C.structure(ry, E(rx, sE)[0], rmask)
    return sE, sC


def kl_of(mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mu * mu + lv.exp() - 1.0 - lv).sum(1)


def nll_of(yh: torch.Tensor, y: torch.Tensor, logb: torch.Tensor) -> torch.Tensor:
    return ((yh - y).abs() / logb.exp()[None, :, None] + (math.log(2.0) + logb)[None, :, None]).sum(1)


def unit_tests() -> None:
    torch.manual_seed(0)
    E, C = EncS(), T.Circ()
    ry, rv, rm = torch.randn(1, T.D_Y, 80), torch.ones(1, 80), torch.ones(1, 80, dtype=torch.bool)
    sE, sC = structures(E, C, ry, rv, rm)
    x = torch.randn(1, T.D_X, 300)
    x2 = x.clone()
    x2[..., 200:] = torch.randn(1, T.D_X, 100)
    m1, l1 = E(x, sE)
    m2, l2 = E(x2, sE)
    assert torch.allclose(m1[..., :200], m2[..., :200], atol=1e-5) and torch.allclose(l1[..., :200], l2[..., :200], atol=1e-5) and not torch.allclose(m1[..., 200:], m2[..., 200:]), "E が因果でない"
    mu, lv = torch.randn(2, T.D_A, 5), torch.randn(2, T.D_A, 5)
    kl = kl_of(mu, lv)
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(mu, (0.5 * lv).exp()), torch.distributions.Normal(0.0, 1.0)).sum(1)
    assert torch.allclose(kl, ref, atol=1e-4), "KL の式がずれる"
    assert float(kl_of(torch.zeros(1, T.D_A, 3), torch.zeros(1, T.D_A, 3)).abs().max()) < 1e-6, "事前分布と同じ事後で KL が 0 でない"
    y = torch.randn(1, T.D_Y, 7)
    lb = torch.full((T.D_Y,), math.log(0.05))
    ref2 = -torch.distributions.Laplace(y, 0.05).log_prob(y + 0.01).sum(1)
    assert torch.allclose(nll_of(y + 0.01, y, lb), ref2, atol=1e-3), "ラプラスの負の対数尤度がずれる"
    print("unit tests ok(E の因果性・KL・ラプラスの尤度)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prereg", default="results/snv_p11/prereg.yaml")
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (ROOT / a.prereg).exists():
        print(f"{a.prereg} が無い: 起動しない", flush=True)
        return 1
    print("prereg", ROOT / a.prereg, flush=True)
    torch.manual_seed(a.seed)
    unit_tests()
    t0 = time.time()
    corpus = T.load_corpus()
    tr, va = T.split(sorted(corpus))
    print(f"voices {len(corpus)} train {len(tr)} val {len(va)} load {time.time() - t0:.0f}s", flush=True)
    (out / "split.json").write_text(json.dumps({"train": tr, "val": va}))
    dev = "cuda"
    E, C = EncS().to(dev), T.Circ().to(dev)
    logb = nn.Parameter(torch.full((T.D_Y,), math.log(B_INIT), device=dev))
    params = list(E.parameters()) + list(C.parameters()) + [logb]
    print(f"params E {sum(p.numel() for p in E.parameters()) / 1e6:.2f}M C {sum(p.numel() for p in C.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = 200 if a.smoke else a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    trb, vab = T.Batches(corpus, tr, a.seed), T.Batches(corpus, va, 12345, p_nv=0.0)
    vbatches = [vab.sample(a.bs) for _ in range(8)]
    to_dev = lambda b: {k: x.to(dev, non_blocking=True) for k, x in b.items()}

    @torch.no_grad()
    def evaluate() -> dict:
        E.eval(), C.eval()
        r: dict = {k: [] for k in ("val_l1", "trivial", "kl", "nll", "fwd", "amax", "aent")}
        for vb in vbatches:
            b = to_dev(vb)
            m = b["ok"].clone()
            m[:, :T.WARM_SELF] = 0
            sE, sC = structures(E, C, b["ry"], b["rv"], b["rmask"])
            mu, lv = E(xin(b["y"], b["v"]), sE)
            yh = C(mu, b["lf"], b["v"], sC)
            r["val_l1"].append(float(T.masked_l1(yh, b["y"], m)))
            rmean = (b["ry"] * b["rmask"][:, None]).sum(-1, keepdim=True) / b["rmask"].sum(-1)[:, None, None].clamp(min=1)
            r["trivial"].append(float(T.masked_l1(rmean.expand_as(b["y"]), b["y"], m)))
            r["kl"].append(float((kl_of(mu, lv) * m).sum() / m.sum()))
            r["nll"].append(float((nll_of(yh, b["y"], logb) * m).sum() / m.sum()))
            perm = torch.roll(torch.arange(mu.shape[0], device=dev), 1)
            r["fwd"].append(float(T.masked_l1(C(mu, b["lf"], b["v"], {k: x[perm] for k, x in sC.items()}), b["y"], m)))
            h = C.inp(torch.cat([mu, b["lf"][:, None], b["v"][:, None]], 1))
            for i, blk in enumerate(C.blocks):
                h = blk(h, sC["z"])
                if i == 3:
                    B_, Ch, L = h.shape
                    q = C.q(h.transpose(1, 2)).view(B_, L, C.heads, C.dk).transpose(1, 2)
                    p = ((q @ sC["k"].transpose(-1, -2)) / math.sqrt(C.dk)).masked_fill(~sC["rmask"][:, None, None, :], -1e9).softmax(-1)
                    r["amax"].append(float(p.max(-1).values.mean()))
                    r["aent"].append(float(-(p * p.clamp(min=1e-12).log()).sum(-1).mean()))
                    break
        E.train(), C.train()
        return {k: float(np.mean(x)) for k, x in r.items()}

    ev0 = evaluate()
    print(f"floors: untrained val L1 {ev0['val_l1']:.4f} / trivial (参照の平均の Y) {ev0['trivial']:.4f}", flush=True)
    log = {"floors": {"untrained": ev0["val_l1"], "trivial": ev0["trivial"]}, "evals": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        b = to_dev(trb.sample(a.bs))
        m = b["ok"].clone()
        m[:, :T.WARM_SELF] = 0
        sE, sC = structures(E, C, b["ry"], b["rv"], b["rmask"])
        mu, lv = E(xin(b["y"], b["v"]), sE)
        aa = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        yh = C(aa, b["lf"], b["v"], sC)
        nll = (nll_of(yh, b["y"], logb) * m).sum() / m.sum()
        kl = (kl_of(mu, lv) * m).sum() / m.sum()
        beta = min(1.0, step / BETA_RAMP)
        loss = nll + beta * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} nll {float(nll):.2f} kl {float(kl):.2f} beta {beta:.2f} l1 {float(T.masked_l1(yh, b['y'], m)):.4f} gn {float(gn):.2f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 2000 == 0 or step == steps:
            ev = evaluate()
            rec = {"step": step, **ev, "train_nll": float(nll), "train_kl": float(kl), "beta": beta, "b": [round(float(x), 4) for x in logb.exp()], "sec": time.time() - t0}
            log["evals"].append(rec)
            print("eval", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "b"}), flush=True)
            ck = {"Es": E.state_dict(), "C": C.state_dict(), "logb": logb.detach().cpu(), "step": step}
            torch.save(ck, out / "last.pt")
            if step == steps // 2:
                torch.save(ck, out / f"step{step // 1000}k.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
            if ev["amax"] > 0.5:
                print(f"STOP: 注意が検索に寄る(最大 {ev['amax']:.3f})", flush=True)
                return 3
            if step == 10000 and ev["val_l1"] > 0.8 * ev["trivial"]:
                print(f"TRIPWIRE: val L1 {ev['val_l1']:.4f} > 0.8 × trivial {ev['trivial']:.4f} at 10k → stop", flush=True)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
