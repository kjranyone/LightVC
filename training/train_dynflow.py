"""分布で学習する動きの生成器 F(current/dyn_flow.md)。データ・文脈は train_rr.py(R)を再利用し、表からの残差 x = [(包絡 − Σp T)/10 (24)・相対 c0 (1)] を
フレームごとの条件付きフローマッチング(CFM)で学習する。条件 = R の文脈(問いの TCN の隠れ + 記憶の注意の読み)+ 表/10 + 直前 K フレームの x(自己回帰・正解に雑音と脱落)。
雑音 z は時間方向の AR(1)(ρ = 0.9)= 推論と同じ分布。損失 = 速度の MSE(有声 1・無声 0.3)。x は学習の初めに測った次元ごとの標準偏差で正規化(重みと一緒に保存)。
推論(sample)= 文脈を発話全体で一括(因果)→ フレームごとに S 段の Euler で標本 → 過去へ積む(先読み 0)。
    uv run python train_dynflow.py --tag diag_flow1 --steps 8000 --init results/diag_rr4/last.pt --p_real 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent
XD = 25
RHO = 0.9


def ar_noise(B: int, T: int, dev: str, gen: torch.Generator | None = None) -> torch.Tensor:
    e = torch.randn(B, XD, T, device=dev, generator=gen)
    z = torch.empty_like(e)
    z[..., 0] = e[..., 0]
    a = math.sqrt(1 - RHO ** 2)
    for t in range(1, T):
        z[..., t] = RHO * z[..., t - 1] + a * e[..., t]
    return z


class Flow(nn.Module):
    def __init__(self, nq: int, ch: int = 192, heads: int = 4, dk: int = 32, K: int = 4, hid: int = 384):
        super().__init__()
        self.cfg = {"nq": nq, "ch": ch, "heads": heads, "dk": dk, "K": K, "hid": hid}
        self.rr = RRm.RR(nq, ch, heads, dk)
        self.K = K
        cin = XD + 16 + ch + heads * dk + 24 + XD * K
        self.head = nn.Sequential(nn.Conv1d(cin, hid, 1), nn.LeakyReLU(0.1), nn.Conv1d(hid, hid, 1), nn.LeakyReLU(0.1), nn.Conv1d(hid, hid, 1), nn.LeakyReLU(0.1), nn.Conv1d(hid, XD, 1))
        self.register_buffer("xstd", torch.ones(XD))

    def context(self, q: torch.Tensor, mem: tuple, rm: torch.Tensor) -> tuple:
        rr = self.rr
        hq = rr.qb(rr.qin(q))
        qq = rr.qout(hq)
        B, _, T = qq.shape
        qq = qq.view(B, rr.h, rr.dk, T)
        k, v = mem
        s = torch.einsum("bhdt,bhdm->bhtm", qq, k) / math.sqrt(rr.dk)
        s = s.masked_fill(~rm[:, None, None, :], -1e4)
        w = s.softmax(-1)
        r = torch.einsum("bhtm,bhdm->bhdt", w, v).reshape(B, rr.h * rr.dk, T)
        return torch.cat([hq, r], 1), 1.0 / (w ** 2).sum(-1)

    def temb(self, tau: torch.Tensor) -> torch.Tensor:
        f = torch.exp(torch.linspace(0, math.log(100.0), 8, device=tau.device))
        a = tau[:, None] * f[None, :, None] * math.pi
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def velocity(self, x: torch.Tensor, tau: torch.Tensor, h: torch.Tensor, tab: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([x, self.temb(tau), h, tab / 10, past], 1))

    def past_of(self, xn: torch.Tensor) -> torch.Tensor:
        if self.K == 0:
            return xn.new_zeros(xn.shape[0], 0, xn.shape[-1])
        return torch.cat([F.pad(xn, (k, 0))[..., : xn.shape[-1]] for k in range(1, self.K + 1)], 1)

    @torch.no_grad()
    def sample_nopast(self, h: torch.Tensor, tab: torch.Tensor, steps: int = 6, gen: torch.Generator | None = None, temp: float = 1.0) -> torch.Tensor:
        """過去フレームを使わない標本(全フレーム並列・因果・自己回帰の露出の偏りが無い)。"""
        B, _, T = h.shape
        x = ar_noise(B, T, h.device, gen) * temp
        past = torch.zeros(B, XD * self.K, T, device=h.device)
        for s in range(steps):
            tau = torch.full((B, 1, T), s / steps, device=h.device)
            x = x + self.velocity(x, tau[:, 0], h, tab, past) / steps
        return x * self.xstd[None, :, None]

    @torch.no_grad()
    def sample(self, h: torch.Tensor, tab: torch.Tensor, steps: int = 6, gen: torch.Generator | None = None, temp: float = 1.0) -> torch.Tensor:
        if self.K == 0:
            return self.sample_nopast(h, tab, steps, gen, temp)
        B, _, T = h.shape
        z = ar_noise(B, T, h.device, gen) * temp
        out = torch.zeros(B, XD, T, device=h.device)
        hist = torch.zeros(B, XD * self.K, 1, device=h.device)
        for t in range(T):
            x = z[..., t:t + 1]
            for s in range(steps):
                tau = torch.full((B, 1), s / steps, device=h.device)
                x = x + self.velocity(x, tau, h[..., t:t + 1], tab[..., t:t + 1], hist) / steps
            out[..., t:t + 1] = x
            hist = torch.cat([x, hist[:, : XD * (self.K - 1)]], 1)
        return out * self.xstd[None, :, None]


def targets(b: dict) -> torch.Tensor:
    de = (b["y"] - b["tab"]) / 10
    d0 = b["yc0"] - b["c0b"]
    v = (b["yv"] & b["tm"]).float()
    d0 = d0 - (d0 * v).sum(1, keepdim=True) / v.sum(1, keepdim=True).clamp(min=1)
    return torch.cat([de, d0[:, None]], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--p_male", type=float, default=0.5)
    ap.add_argument("--p_real", type=float, default=0.5)
    ap.add_argument("--init", default="")
    ap.add_argument("--K", type=int, default=4, help="自己回帰の過去フレーム数(0 = 自己回帰なし・diag_flow2)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--align", default="rr", choices=["rr", "c1"], help="交差ペアの入力側の整列(train_rr.py と同じ・diag_flow3c は c1)")
    a = ap.parse_args()
    RRm.CFG["align"] = a.align
    od = ROOT / "results" / a.tag
    od.mkdir(parents=True, exist_ok=True)
    assert (od / "prereg.yaml").exists(), "事前登録が無い"
    log = open(od / "train.log", "a")
    print(f"prereg: {od / 'prereg.yaml'}", file=log, flush=True)
    torch.manual_seed(0)
    dev = "cuda"
    st = RRm.Store()
    held = RRm.held_females()
    fems = sorted(s for s in {s for (c, s, _) in st.f if c == "fem_neutral"} if all(("fem_neutral", s, f"t{t:02d}_neutral") in st.f for t in range(10)))
    tr_f, va_f = [s for s in fems if s not in held], [s for s in fems if s in held]
    males = sorted(s for s in {s for (c, s, _) in st.f if c == "male_par"} if all(("male_par", s, f"f{t:02d}") in st.f for t in range(10)))
    tr_m = [m for m in males if m not in RRm.EVAL_MALES and m not in RRm.VAL_MALES and m in st.stats]
    va_m = [m for m in males if m in RRm.VAL_MALES and m in st.stats]
    reals = sorted(s for s, v in st.real.items() if len(v) >= 8)
    rv_ = set(reals[::20])
    tr_r, va_r = [s for s in reals if s not in rv_], sorted(rv_)
    ds = RRm.Pairs(st, tr_f, tr_m, 10 ** 9, a.p_male, 0, crop=400, real=tr_r, p_real=a.p_real)
    vds = RRm.Pairs(st, va_f, va_m, 144, 0.5, 1, fixed=True, crop=400, real=va_r, p_real=1 / 3)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, collate_fn=RRm.collate, persistent_workers=True, prefetch_factor=4)
    vdl = torch.utils.data.DataLoader(vds, batch_size=8, num_workers=2, collate_fn=RRm.collate)
    net = Flow(nq=RRm.K + 24 + 3, K=a.K).to(dev)
    if a.init:
        sd = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)["net"]
        net.rr.load_state_dict(sd)
    it = iter(dl)
    acc = []
    for _ in range(50):
        b = next(it)
        x = targets(b)
        v = (b["yv"] & b["tm"])
        acc.append(x.permute(0, 2, 1)[v].numpy())
    xs = np.concatenate(acc)
    net.xstd.copy_(torch.from_numpy(xs.std(0).clip(1e-3)).float())
    print("xstd", np.round(xs.std(0), 3).tolist(), file=log, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)

    def step_loss(b: dict) -> tuple:
        mem = net.rr.memory(b["rp"], b["rv"])
        h, eff = net.context(b["q"], mem, b["rm"])
        x1 = targets(b) / net.xstd[None, :, None]
        B, _, T = x1.shape
        z = ar_noise(B, T, dev)
        tau = torch.rand(B, 1, T, device=dev)
        xt = (1 - tau) * z + tau * x1
        past = net.past_of(x1) + 0.3 * torch.randn(B, XD * net.K, T, device=dev)
        past = past * (torch.rand(B, 1, 1, device=dev) > 0.2).float()
        vp = net.velocity(xt, tau[:, 0], h, b["tab"], past)
        w = (b["yv"].float() + 0.3 * (~b["yv"]).float()) * b["tm"].float()
        l = (((vp - (x1 - z)) ** 2).mean(1) * w).sum() / w.sum().clamp(min=1)
        return l, eff

    @torch.no_grad()
    def validate() -> dict:
        net.eval()
        res: dict = {}
        torch.manual_seed(1)
        for b in vdl:
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            for i, kd in enumerate(b["kind"]):
                bi = {k: (v[i:i + 1] if torch.is_tensor(v) else v) for k, v in b.items()}
                l, eff = step_loss(bi)
                res.setdefault(kd, []).append([float(l), float(eff[0][:, bi["tm"][0]].median())])
        net.train()
        return {k: {"cfm": round(float(np.mean([r[0] for r in v])), 4), "eff": round(float(np.median([r[1] for r in v])), 1), "n": len(v)} for k, v in res.items()}

    v0 = validate()
    print("init", json.dumps(v0), file=log, flush=True); print("init", v0, flush=True)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = next(it)
        b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        loss, _ = step_loss(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(gn):
            continue
        opt.step(); sched.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": round(float(loss.detach()), 4), "s_per_step": round((time.time() - t0) / step, 3)}), file=log, flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            v = validate()
            print("eval", step, json.dumps(v), file=log, flush=True); print("eval", step, v, flush=True)
            torch.save({"net": net.state_dict(), "cfg": net.cfg, "step": step, "args": vars(a)}, od / "last.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
