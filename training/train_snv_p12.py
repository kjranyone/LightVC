"""P12(synth_nam_vc §8.3 rev5.1・原点): 話者 = 回路の値 θ を、その話者の音に当てはめる(NAM の型)。操作者もまた回路 = 操作者の θ も登録で当てはめ、演奏の取り出しに使う。
y = G(a, lf, 有声; θ_s)・演奏 a ~ N(0, I)(1 フレームごと)・E(y, 有声; θ_s)→ q(a | y, θ_s)(操作者の回路の逆)。G と E は因果の膨張畳み込み(GBlock 12 段)で、各段の FiLM が θ を読む(表の読みも注意も無い = 連続の回路)。
学習の声は θ を声ごとに持つ(全発話で共有 = フレームあたりの代価なし・事前 N(0, I) を声のフレーム数で割って足す = MAP)。損失 = 自己の reamp の ELBO(ラプラス・尺度を学ぶ)+ β KL(β は 5k で 0 → 1)。
新しい声(目標・操作者)は E・G・尺度を止めて、登録の発話の ELBO + 事前で θ を当てはめる(fit_theta・Adam 300 歩・初期値 0)。
事前登録 = results/snv_p12/prereg.yaml。
    cd training && uv run --no-sync python train_snv_p12.py --tag snv_p12_f --prereg results/snv_p12/prereg.yaml
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

import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = Path(__file__).resolve().parent.parent
D_TH = 128
BETA_RAMP = 5000
FIT_STEPS = 300
FIT_LR = 0.05


class Net(nn.Module):
    def __init__(self, d_in: int, d_out: int, ch: int = 256) -> None:
        super().__init__()
        self.inp = nn.Conv1d(d_in, ch, 1)
        self.blocks = nn.ModuleList([T.GBlock(ch, d, D_TH) for d in (1, 2, 4, 8, 16, 32) * 2])
        self.head = nn.Conv1d(ch, d_out, 1)

    def forward(self, x: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, th)
        return self.head(h)


class Circuit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.E = Net(T.D_X, 2 * T.D_A)
        self.G = Net(T.D_A + 2, T.D_Y)
        self.logb = nn.Parameter(torch.full((T.D_Y,), math.log(P11.B_INIT)))

    def enc(self, y: torch.Tensor, v: torch.Tensor, th: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, lv = self.E(P11.xin(y, v), th).chunk(2, 1)
        return mu, lv.clamp(-10.0, 4.0)

    def dec(self, a: torch.Tensor, lf: torch.Tensor, v: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        return self.G(torch.cat([a, lf[:, None], v[:, None]], 1), th)

    def elbo_terms(self, y: torch.Tensor, lf: torch.Tensor, v: torch.Tensor, m: torch.Tensor, th: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lv = self.enc(y, v, th)
        a = mu + (0.5 * lv).exp() * torch.randn_like(mu) if sample else mu
        yh = self.dec(a, lf, v, th)
        nll = (P11.nll_of(yh, y, self.logb) * m).sum() / m.sum()
        kl = (P11.kl_of(mu, lv) * m).sum() / m.sum()
        return nll, kl, yh


def fit_theta(M: Circuit, ys: list[torch.Tensor], lfs: list[torch.Tensor], vs: list[torch.Tensor], steps: int = FIT_STEPS) -> torch.Tensor:
    dev = ys[0].device
    L = max(y.shape[-1] for y in ys)
    Y = torch.zeros(len(ys), T.D_Y, L, device=dev)
    LF, V, Mk = (torch.zeros(len(ys), L, device=dev) for _ in range(3))
    for i, (y, lf, v) in enumerate(zip(ys, lfs, vs)):
        n = y.shape[-1]
        Y[i, :, :n], LF[i, :n], V[i, :n], Mk[i, :n] = y, lf, v, 1.0
    Mk[:, :T.WARM_SELF] = 0
    th = torch.zeros(1, D_TH, device=dev, requires_grad=True)
    op = torch.optim.Adam([th], FIT_LR)
    nf = float(Mk.sum())
    req = [p.requires_grad for p in M.parameters()]
    M.requires_grad_(False)
    with torch.enable_grad():
        for _ in range(steps):
            nll, kl, _ = M.elbo_terms(Y, LF, V, Mk, th.expand(len(ys), -1), sample=False)
            loss = nll + kl + 0.5 * (th * th).sum() / nf
            op.zero_grad()
            loss.backward()
            op.step()
    for p, r in zip(M.parameters(), req):
        p.requires_grad_(r)
    return th.detach()


def unit_tests() -> None:
    torch.manual_seed(0)
    M = Circuit()
    th = torch.randn(1, D_TH)
    y, v = torch.randn(1, T.D_Y, 300), torch.ones(1, 300)
    y2 = y.clone()
    y2[..., 200:] = torch.randn(1, T.D_Y, 100)
    m1, _ = M.enc(y, v, th)
    m2, _ = M.enc(y2, v, th)
    assert torch.allclose(m1[..., :200], m2[..., :200], atol=1e-5) and not torch.allclose(m1[..., 200:], m2[..., 200:]), "E が因果でない"
    a, lf = torch.randn(1, T.D_A, 300), torch.randn(1, 300)
    a2, lf2 = a.clone(), lf.clone()
    a2[..., 200:], lf2[:, 200:] = torch.randn(1, T.D_A, 100), 0.3
    g1, g2 = M.dec(a, lf, v, th), M.dec(a2, lf2, v, th)
    assert torch.allclose(g1[..., :200], g2[..., :200], atol=1e-5) and not torch.allclose(g1[..., 200:], g2[..., 200:]), "G が因果でない"
    w0 = [p.detach().clone() for p in M.parameters()]
    ys = [torch.randn(T.D_Y, 400) * 0.1]
    lfs, vs = [torch.zeros(400)], [torch.ones(400)]
    mk = torch.ones(1, 400)
    mk[:, :T.WARM_SELF] = 0
    l0 = sum(float(x) for x in M.elbo_terms(ys[0][None], lfs[0][None], vs[0][None], mk, torch.zeros(1, D_TH), sample=False)[:2])
    tf = fit_theta(M, ys, lfs, vs, steps=50)
    l1 = sum(float(x) for x in M.elbo_terms(ys[0][None], lfs[0][None], vs[0][None], mk, tf, sample=False)[:2])
    assert l1 < l0, "θ の当てはめで ELBO が下がらない"
    assert all(torch.equal(p, q) for p, q in zip(M.parameters(), w0)), "θ の当てはめが回路の重みを変えた"
    print(f"unit tests ok(E・G の因果性・θ の当てはめで −ELBO {l0:.2f} → {l1:.2f}・重みは不変)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prereg", default="results/snv_p12/prereg.yaml")
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
    nfr = torch.tensor([sum(u["y"].shape[1] for u in corpus[n]["norm"] + corpus[n]["nv"]) for n in tr], dtype=torch.float32)
    dev = "cuda"
    M = Circuit().to(dev)
    TH = nn.Embedding(len(tr), D_TH).to(dev)
    nn.init.normal_(TH.weight, 0.0, 0.01)
    nfr = nfr.to(dev)
    params = list(M.parameters()) + list(TH.parameters())
    print(f"params E {sum(p.numel() for p in M.E.parameters()) / 1e6:.2f}M G {sum(p.numel() for p in M.G.parameters()) / 1e6:.2f}M θ {TH.weight.numel() / 1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW([{"params": list(M.parameters()), "weight_decay": 0.0}, {"params": list(TH.parameters()), "weight_decay": 0.0}], lr=a.lr)
    steps = 200 if a.smoke else a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    trb = T.Batches(corpus, tr, a.seed)
    to_dev = lambda b: {k: x.to(dev, non_blocking=True) for k, x in b.items()}
    vnames = [v for v in va if v.startswith("ef")][:6] + [v for v in va if v.startswith("em")][:2]
    vu = {v: [corpus[v]["norm"][j] for j in range(len(corpus[v]["norm"]))] for v in vnames}
    tt = lambda u, k: torch.from_numpy(u[k]).to(dev)

    @torch.no_grad()
    def evaluate() -> dict:
        M.eval()
        r: dict = {"val_l1": [], "fwd": [], "kl": []}
        ths = {v: fit_theta(M, [tt(u, "y") for u in vu[v][:3]], [tt(u, "lf") for u in vu[v][:3]], [tt(u, "v") for u in vu[v][:3]]) for v in vnames}
        for i, v in enumerate(vnames):
            for u in vu[v][8:10]:
                y, lf, vv = tt(u, "y")[None], tt(u, "lf")[None], tt(u, "v")[None]
                mk = torch.ones_like(vv)
                mk[:, :T.WARM_SELF] = 0
                mu, lv = M.enc(y, vv, ths[v])
                r["val_l1"].append(float(T.masked_l1(M.dec(mu, lf, vv, ths[v]), y, mk)))
                r["kl"].append(float((P11.kl_of(mu, lv) * mk).sum() / mk.sum()))
                r["fwd"].append(float(T.masked_l1(M.dec(mu, lf, vv, ths[vnames[(i + 1) % len(vnames)]]), y, mk)))
        M.train()
        return {k: float(np.mean(x)) for k, x in r.items()}

    log = {"evals": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        b = to_dev(trb.sample(a.bs))
        m = b["ok"].clone()
        m[:, :T.WARM_SELF] = 0
        th = TH(b["cls"])
        nll, kl, yh = M.elbo_terms(b["y"], b["lf"], b["v"], m, th)
        prior = (0.5 * (th * th).sum(1) / nfr[b["cls"]]).mean()
        beta = min(1.0, step / BETA_RAMP)
        loss = nll + beta * kl + prior
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} nll {float(nll):.2f} kl {float(kl):.2f} beta {beta:.2f} l1 {float(T.masked_l1(yh, b['y'], m)):.4f} |θ| {float(TH.weight.norm(dim=1).mean()):.2f} gn {float(gn):.2f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 2000 == 0 or step == steps:
            ev = evaluate()
            rec = {"step": step, **ev, "train_nll": float(nll), "train_kl": float(kl), "beta": beta, "theta_norm": float(TH.weight.norm(dim=1).mean()), "sec": time.time() - t0}
            log["evals"].append(rec)
            print("eval", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            ck = {"M": M.state_dict(), "TH": TH.state_dict(), "step": step}
            torch.save(ck, out / "last.pt")
            if step == steps // 2:
                torch.save(ck, out / f"step{step // 1000}k.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
