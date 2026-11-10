"""段 T の腕と評価(paper_snv §6・results/snv_t/prereg.yaml・diag): 合成の世界(snv_t_world.py)で、同じ網・同じ予算の腕を学び、正解の reamp との変換の誤差の比 e・演奏の話者の率 spk・話者を跨いだ対応 corr を測る。
腕: a_det(自己の reamp の L1)・b_vae(+ KL N(0, I)・尺度を学ぶ)・b_vae_fix(尺度 0.05)・c_white(演奏を話者の参照の統計で白くする)・
d_tpl(+ 内容ごとの共有の型の事前分布・単調な対応の探索・尺度を学ぶ)・d_tpl_fix(尺度 0.05)・oracle(真の演奏で描画を学ぶ)。
    cd training && uv run --no-sync python train_snv_t.py --arm d_tpl --K 4 [--nocross]
T5(results/snv_t5/prereg.yaml): --nref 1(参照と登録の発話 3 → 1)・--dphi 4(構造の次元 32 → 4)・--device cpu・--out results/snv_t5
T6(results/snv_t6/prereg.yaml): --tpl_lr r(型の lr を r 倍)・--tpl_ema(型 = 対応づけた演奏の平均の移動平均・γ = --tpl_gamma)・--tpl_ev p(型を更新する出現の割合)・--seed
T9(results/snv_t9/prereg.yaml・paper_snv 系 2e・§5 M7): --rot r(世界の演奏に話者ごとの回転)・--arm f_share(固い共有: 復号器の入力 = 内容の共有の軌跡 μ_w[τ](全ての声の再現で学ぶ)+ 残差 r(--share_res kl)・E は勾配を止めた軌跡を当てる)。share_r2 = 学習の内容 20 を 8 声が演じた E の演奏の 1 声を抜いた型の R²
T8(results/snv_t8/prereg.yaml・paper_snv §5 M5): --critic λ(型からのずれ r = μ_q − μ_w[τ] から学習の話者を当てる因果の判別器 Q を交互に学び、演奏の側は −λ × Q の交差エントロピー)。tpl_r2 = 型が事後の平均の分散を説明する割合(最後の 1000 step の平均)
T11(results/snv_t11/prereg.yaml・paper_snv 系 2g・§5 M9): --n_utt(声ごとの発話の数)・--crop f(学習を発話の f 倍の切り出しで・位置は付帯の量)・--align fixed|free|band(両端固定・端の自由・時刻の比の ±--band × L の帯)・
--tpl_est grad|ste(軌跡 = 勾配の引数 / E の出力の対応づけた移動平均 + straight-through)・--align_warm(時刻の比の暖機の step)。
T12 / T13(paper_snv 系 2h・§5 M10・M11): --side k(C に y の平均と傾きを声の参照で標準化した明示の側の入力)・--group K(束 = 同じ内容の K 声・切り出しは同じ時刻の比の窓)・--tpl_est gste(声どうしで平均した勾配)・gswap(+ 同じ組の別の声のずれを C に見せる)。f_share の健全さ(最後の 1000 step): vr_pm = 軌跡の切り出しの中の分散 / 束の中の分散・vr_u = 真の演奏の同じ比・cover = 切り出しが通る位置の数 / 時刻の比の期待・pm_sq = 軌跡の大きさ・T14(rev17・§5 M12): gloo(勾配から自分を除く = 声の出力は同じ組の同じ位置の他の声の再現の勾配だけ)・rho = 勾配の平均のうち自分のフレームの割合・T15(rev18・§5 M13): --student(推論の E2 は sg(軌跡) への回帰だけで学ぶ・構造は勾配なしで読む・評価は E2・fit_s = E2 の当ての誤差 / 軌跡の分散)
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
import torch.nn.functional as F_

import snv_t_world as W

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/snv_t"
D_PHI, CH, STEPS, BS, LR = 32, 128, 20000, 32, 1e-3
N_REF, N_ENR, N_TU = 3, 3, 4
B_FIX, SHRINK, BETA_RAMP = 0.05, 0.1, 2000
ALIGN_WARM = 2000
DEV = "cuda"


class Block(nn.Module):
    def __init__(self, dil: int) -> None:
        super().__init__()
        self.dil = dil
        self.conv = nn.Conv1d(CH, 2 * CH, 5, dilation=dil)
        self.film = nn.Linear(D_PHI, 2 * CH)
        self.out = nn.Conv1d(CH, CH, 1)

    def forward(self, h: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        a, b = self.conv(F_.pad(h, (4 * self.dil, 0))).chunk(2, 1)
        g, s = self.film(phi).chunk(2, -1)
        a = a * (1 + g[..., None]) + s[..., None]
        return h + self.out(torch.tanh(a) * torch.sigmoid(b))


class Net(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.inp = nn.Conv1d(d_in, CH, 1)
        self.blocks = nn.ModuleList([Block(d) for d in (1, 2, 4, 8)])
        self.head = nn.Conv1d(CH, d_out, 1)

    def forward(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, phi)
        return self.head(h)


class Critic(nn.Module):
    def __init__(self, n_spk: int) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(W.D_U, 64, 5)
        self.c2 = nn.Conv1d(64, 64, 5, dilation=2)
        self.out = nn.Conv1d(64, n_spk, 1)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        h = F_.relu(self.c1(F_.pad(r, (4, 0))))
        h = F_.relu(self.c2(F_.pad(h, (8, 0))))
        return self.out(h)


class Struct(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Linear(W.D_Y, CH), nn.ReLU(), nn.Linear(CH, CH), nn.ReLU())
        self.g = nn.Linear(CH, D_PHI)

    def forward(self, ry: torch.Tensor, rm: torch.Tensor) -> torch.Tensor:
        h = self.f(ry.transpose(1, 2)) * rm[..., None]
        return self.g(h.sum(1) / rm.sum(1, keepdim=True).clamp(min=1.0))


def zca(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = m[:, None].double()
    mu = (x.double() * w).sum(-1) / w.sum(-1).clamp(min=1.0)
    d = x.double() - mu[..., None]
    cov = torch.einsum("bit,bjt->bij", d * w, d) / w.sum(-1).clamp(min=1.0)[..., None]
    D = cov.shape[-1]
    cov = (1 - SHRINK) * cov + SHRINK * (cov.diagonal(dim1=-2, dim2=-1).sum(-1) / D)[:, None, None] * torch.eye(D, dtype=cov.dtype, device=cov.device)
    e, q = torch.linalg.eigh(cov)
    return mu.float(), ((q / e.clamp(min=1e-8).sqrt()[:, None, :]) @ q.transpose(-1, -2)).float()


def mas(logp: torch.Tensor, tl: torch.Tensor, ll: torch.Tensor) -> torch.Tensor:
    B, T, L = logp.shape
    neg = -1e9
    jj = torch.arange(L, device=logp.device)[None]
    Q = torch.where(jj == 0, logp[:, 0], torch.full_like(logp[:, 0], neg))
    back = torch.zeros(B, T, L, dtype=torch.bool, device=logp.device)
    for t in range(1, T):
        move = torch.cat([torch.full_like(Q[:, :1], neg), Q[:, :-1]], 1)
        cm = move > Q
        Qn = torch.where(cm, move, Q) + logp[:, t]
        act = (t < tl)[:, None]
        Q = torch.where(act, Qn, Q)
        back[:, t] = cm & act
    j = ll - 1
    path = torch.zeros(B, T, dtype=torch.long, device=logp.device)
    ar = torch.arange(B, device=logp.device)
    for t in range(T - 1, -1, -1):
        act = t < tl
        path[:, t] = torch.where(act, j, torch.zeros_like(j))
        mv = back[ar, t, j] & act & (t > 0)
        j = (j - mv.long()).clamp(min=0)
    return path


@torch.no_grad()
def ema_update(tpl: torch.Tensor, seen: torch.Tensor, ci: torch.Tensor, path: torch.Tensor, mu: torch.Tensor, m: torch.Tensor, ll: torch.Tensor, gamma: float, use: np.ndarray) -> None:
    for b in range(len(ci)):
        if not use[b]:
            continue
        c, n, L = int(ci[b]), int(m[b].sum()), int(ll[b])
        j = path[b, :n]
        cnt = torch.bincount(j, minlength=L)[:L].float()
        s1 = torch.zeros(L, mu.shape[1], device=mu.device).index_add_(0, j, mu[b, :, :n].T)
        obs = cnt > 0
        mean = s1 / cnt.clamp(min=1.0)[:, None]
        if not bool(seen[c]):
            idx = torch.arange(L, device=mu.device)
            oi = idx[obs]
            near = oi[(idx[:, None] - oi[None, :]).abs().argmin(1)]
            tpl.data[c, :L] = mean[near]
            seen[c] = True
        else:
            tpl.data[c, :L][obs] = gamma * tpl.data[c, :L][obs] + (1 - gamma) * mean[obs]


def band_mask(lp: torch.Tensor, off: torch.Tensor, nall: torch.Tensor, ll: torch.Tensor, delta: float) -> torch.Tensor:
    tt = torch.arange(lp.shape[1], device=lp.device)[None].float()
    c = (off[:, None].float() + tt + 0.5) / nall[:, None].float() * ll[:, None].float()
    w = torch.clamp(delta * ll.float(), min=1.0)[:, None, None]
    jj = torch.arange(lp.shape[2], device=lp.device)[None, None].float()
    return lp.masked_fill((jj - c[..., None]).abs() > w, -1e9)


def var_ratio(x: torch.Tensor, m: torch.Tensor) -> float:
    w = m[:, None]
    n = w.sum(-1).clamp(min=1.0)
    mu = (x * w).sum(-1) / n
    within = ((((x - mu[..., None]) ** 2) * w).sum(-1) / n).sum(1).mean()
    g = (x * w).sum((0, 2)) / w.sum()
    total = (((x - g[None, :, None]) ** 2) * w).sum() / w.sum()
    return float(within / total.clamp(min=1e-12))


@torch.no_grad()
def ema_vec(tpl: torch.Tensor, seen: torch.Tensor, ci: torch.Tensor, path: torch.Tensor, mu: torch.Tensor, m: torch.Tensor, ll: torch.Tensor, gamma: float) -> None:
    Lm, D = tpl.shape[1], tpl.shape[2]
    valid = m > 0
    uk, inv = torch.unique((ci[:, None] * Lm + path)[valid], return_inverse=True)
    mean = torch.zeros(len(uk), D, device=mu.device).index_add_(0, inv, mu.transpose(1, 2)[valid]) / torch.bincount(inv, minlength=len(uk)).float()[:, None]
    flat = tpl.view(-1, D)
    cid = uk // Lm
    new = ~seen[cid]
    flat[uk[~new]] = gamma * flat[uk[~new]] + (1 - gamma) * mean[~new]
    flat[uk[new]] = mean[new]
    for c in torch.unique(cid[new]).tolist():
        L = int(ll[(ci == c).nonzero()[0, 0]])
        pos = uk[new & (cid == c)] - c * Lm
        idx = torch.arange(L, device=mu.device)
        oi = pos[pos < L]
        tpl[c, :L] = tpl[c, :L][oi[(idx[:, None] - oi[None, :]).abs().argmin(1)]]
    seen[cid] = True


def group_ste(mu: torch.Tensor, pv: torch.Tensor, path: torch.Tensor, m: torch.Tensor, group: int, Lm: int) -> torch.Tensor:
    B, D, _ = mu.shape
    key = (torch.arange(B, device=mu.device) // group)[:, None] * Lm + path
    valid = m > 0
    kv = key[valid]
    G = (B + group - 1) // group
    s = torch.zeros(G * Lm, D, device=mu.device, dtype=mu.dtype).index_add(0, kv, mu.transpose(1, 2)[valid])
    cnt = torch.zeros(G * Lm, device=mu.device, dtype=mu.dtype).index_add(0, kv, torch.ones(len(kv), device=mu.device, dtype=mu.dtype))
    g = (s / cnt.clamp(min=1.0)[:, None])[key].transpose(1, 2)
    return pv + g - g.detach()


def side_feats(y: torch.Tensor, ry: torch.Tensor, rm: torch.Tensor, k: int) -> torch.Tensor:
    lin = torch.linspace(-1.0, 1.0, y.shape[1], device=y.device)
    f = lambda x: torch.stack([x.mean(1), (x * lin[None, :, None]).mean(1)], 1)[:, :k]
    fr, w = f(ry), rm[:, None]
    n = w.sum(-1).clamp(min=1.0)
    mu = (fr * w).sum(-1) / n
    sd = ((((fr - mu[..., None]) ** 2) * w).sum(-1) / n).sqrt() + 1e-3
    return (f(y) - mu[..., None]) / sd[..., None]


def pad(seqs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(len(s) for s in seqs)
    X = np.zeros((len(seqs), seqs[0].shape[1], L), np.float32)
    M = np.zeros((len(seqs), L), np.float32)
    for i, s in enumerate(seqs):
        X[i, :, :len(s)], M[i, :len(s)] = s.T, 1.0
    return torch.from_numpy(X).to(DEV), torch.from_numpy(M).to(DEV)


def unit_tests() -> None:
    torch.manual_seed(0)
    net, phi = Net(W.D_Y, 2 * W.D_U), torch.randn(1, D_PHI)
    x = torch.randn(1, W.D_Y, 60)
    x2 = x.clone()
    x2[..., 40:] = torch.randn(1, W.D_Y, 20)
    assert torch.allclose(net(x, phi)[..., :40], net(x2, phi)[..., :40], atol=1e-5), "網が因果でない"
    lp = torch.full((1, 6, 3), -10.0)
    for t, j in enumerate([0, 0, 1, 1, 2, 2]):
        lp[0, t, j] = 0.0
    assert mas(lp.to(DEV), torch.tensor([6], device=DEV), torch.tensor([3], device=DEV)).cpu().tolist() == [[0, 0, 1, 1, 2, 2]], "単調な対応の探索がずれる"
    xs = torch.randn(1, 4, 500) * torch.tensor([0.5, 1.0, 2.0, 3.0])[None, :, None]
    mu, Wm = zca(xs, torch.ones(1, 500))
    z = torch.einsum("bij,bjt->bit", Wm, xs - mu[..., None])
    v = (xs - mu[..., None]).pow(2).mean(-1)[0]
    want = (v / ((1 - SHRINK) * v + SHRINK * v.mean())).sqrt()
    assert float((z.pow(2).mean(-1).sqrt()[0] - want).abs().max()) < 0.05, "白くする段が収縮込みの解析値とずれる"
    tp = torch.zeros(2, 4, 2)
    sn = torch.zeros(2, dtype=torch.bool)
    mu = torch.arange(10.0).repeat(2, 1)[None]
    ema_update(tp, sn, torch.tensor([1]), torch.tensor([[0, 0, 2, 2, 2, 0, 0, 0, 0, 0]]), mu, torch.tensor([[1.0] * 5 + [0.0] * 5]), torch.tensor([4]), 0.5, np.array([True]))
    assert bool(sn[1]) and tp[1, :, 0].tolist() == [0.5, 0.5, 3.0, 3.0] and float(tp[0].abs().sum()) == 0.0, f"型の初期化がずれる {tp[1, :, 0].tolist()}"
    ema_update(tp, sn, torch.tensor([1]), torch.tensor([[0, 1, 1, 3, 3, 0, 0, 0, 0, 0]]), mu, torch.tensor([[1.0] * 5 + [0.0] * 5]), torch.tensor([4]), 0.5, np.array([True]))
    assert tp[1, :, 0].tolist() == [0.25, 1.0, 3.0, 3.25], f"型の移動平均がずれる {tp[1, :, 0].tolist()}"
    from train_snv_i import mas_free
    lp = torch.full((1, 10, 10), -5.0)
    lp[:, :, 0] = 0.0
    one = lambda v: torch.tensor([v])
    assert mas_free(lp, one(10), one(10)).tolist() == [[0] * 10], "端の自由な対応が 1 点に留まらない(張り付きの再現)"
    pb = mas_free(band_mask(lp, one(0), one(10), one(10), 0.1), one(10), one(10))[0]
    ctr = (torch.arange(10) + 0.5)
    assert bool(((pb.float() - ctr).abs() <= 1.0).all()) and bool((pb[1:] >= pb[:-1]).all()) and int(pb[-1]) >= 8, f"帯の対応が帯を外れる {pb.tolist()}"
    pc = mas_free(band_mask(lp[:, :5], one(5), one(10), one(10), 0.1), one(5), one(10))[0]
    assert int(pc[0]) >= 4 and int(pc[-1]) >= 8, f"切り出しの位置が帯に入らない {pc.tolist()}"
    mu = torch.randn(1, 3, 4, requires_grad=True)
    pv = torch.randn(1, 3, 4)
    x = mu + (pv - mu).detach()
    x.sum().backward()
    assert torch.allclose(x, pv) and torch.allclose(mu.grad, torch.ones_like(mu)), "straight-through の値か勾配がずれる"
    tq = torch.zeros(2, 4, 2)
    sq = torch.zeros(2, dtype=torch.bool)
    mq = torch.tensor([[1.0] * 5 + [0.0] * 5])
    mu10 = torch.arange(10.0).repeat(2, 1)[None]
    ema_vec(tq, sq, torch.tensor([1]), torch.tensor([[0, 0, 2, 2, 2, 0, 0, 0, 0, 0]]), mu10, mq, torch.tensor([4]), 0.5)
    assert bool(sq[1]) and tq[1, :, 0].tolist() == [0.5, 0.5, 3.0, 3.0] and float(tq[0].abs().sum()) == 0.0, f"束の型の初期化がずれる {tq[1, :, 0].tolist()}"
    ema_vec(tq, sq, torch.tensor([1]), torch.tensor([[0, 1, 1, 3, 3, 0, 0, 0, 0, 0]]), mu10, mq, torch.tensor([4]), 0.5)
    assert tq[1, :, 0].tolist() == [0.25, 1.0, 3.0, 3.25], f"束の型の移動平均がずれる {tq[1, :, 0].tolist()}"
    mg = torch.randn(4, 3, 5, requires_grad=True)
    pg = torch.randn(4, 3, 5)
    pathg = torch.tensor([[0, 0, 1, 1, 2]] * 2 + [[0, 1, 1, 2, 2]] * 2)
    xg = group_ste(mg, pg, pathg, torch.ones(4, 5), 2, 3)
    wg = torch.randn(4, 3, 5)
    (xg * wg).sum().backward()
    want = torch.zeros_like(mg)
    for b in range(4):
        for t in range(5):
            sel = [(b2, t2) for b2 in range(4) if b2 // 2 == b // 2 for t2 in range(5) if pathg[b2, t2] == pathg[b, t]]
            want[b, :, t] = sum(wg[b2, :, t2] for b2, t2 in sel) / len(sel)
    assert torch.allclose(xg, pg) and torch.allclose(mg.grad, want, atol=1e-6), "同じ内容の声どうしで平均した straight-through がずれる"
    sf = side_feats(torch.randn(2, 29, 7) * 3 + 1, torch.randn(2, 29, 50) * 3 + 1, torch.ones(2, 50), 2)
    assert sf.shape == (2, 2, 7), "側の入力の形がずれる"
    m2 = torch.ones(2, 6)
    assert var_ratio(torch.ones(2, 3, 6) * torch.tensor([1.0, 2.0])[:, None, None], m2) == 0.0, "定数の軌跡の分散の比が 0 でない"
    print("unit tests ok(因果・単調な対応・白くする段・型の移動平均・帯の対応・straight-through・束の型・声どうしで平均した straight-through)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["a_det", "b_vae", "b_vae_fix", "c_white", "d_tpl", "d_tpl_fix", "oracle", "f_share"])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--nocross", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nref", type=int, default=3)
    ap.add_argument("--dphi", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/snv_t")
    ap.add_argument("--tpl_lr", type=float, default=1.0)
    ap.add_argument("--tpl_ema", action="store_true")
    ap.add_argument("--tpl_gamma", type=float, default=0.9)
    ap.add_argument("--tpl_ev", type=float, default=1.0)
    ap.add_argument("--critic", type=float, default=0.0)
    ap.add_argument("--rot", type=float, default=0.0)
    ap.add_argument("--share_res", default="none", choices=["none", "kl"])
    ap.add_argument("--n_utt", type=int, default=W.N_UTT)
    ap.add_argument("--crop", type=float, default=0.0)
    ap.add_argument("--align", default="fixed", choices=["fixed", "free", "band"])
    ap.add_argument("--band", type=float, default=0.1)
    ap.add_argument("--tpl_est", default="grad", choices=["grad", "ste", "gste", "gswap", "gloo"])
    ap.add_argument("--side", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--align_warm", type=int, default=ALIGN_WARM)
    ap.add_argument("--student", action="store_true")
    a = ap.parse_args()
    assert a.crop == 0 or a.align != "fixed", "切り出しで両端固定の対応は使えない"
    assert (not a.side and not a.group and not a.student) or a.arm == "f_share", "側の入力・束の組・推論の E の回帰は f_share だけ"
    assert a.tpl_est not in ("gste", "gswap", "gloo") or (a.group and a.crop), "声どうしで平均した straight-through は束の組と切り出しで"
    global D_PHI, N_REF, N_ENR, DEV, OUT
    D_PHI, N_REF, N_ENR, DEV, OUT = a.dphi, a.nref, a.nref, a.device, ROOT / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    if a.device == "cpu":
        torch.set_num_threads(3)
    tag = f"{a.arm}_K{a.K}" + ("_nx" if a.nocross else "") + (f"_r{a.nref}" if a.nref != 3 else "") + (f"_p{a.dphi}" if a.dphi != 32 else "")
    tag += (f"_tl{a.tpl_lr:g}" if a.tpl_lr != 1.0 else "") + (f"_ema{a.tpl_gamma:g}_ev{a.tpl_ev:g}" if a.tpl_ema else "") + (f"_crit{a.critic:g}" if a.critic else "") + (f"_rot{a.rot:g}" if a.rot else "") + ("_res" if a.share_res == "kl" else "") + (f"_s{a.seed}" if a.seed else "")
    tag += (f"_u{a.n_utt}" if a.n_utt != W.N_UTT else "") + (f"_c{a.crop:g}" if a.crop else "") + (f"_{a.align}" if a.align != "fixed" else "") + (f"{a.band:g}" if a.align == "band" else "")
    tag += (f"_{a.tpl_est}" if a.tpl_est != "grad" else "") + (f"_w{a.align_warm}" if a.align_warm != ALIGN_WARM else "")
    tag += (f"_side{a.side}" if a.side else "") + (f"_g{a.group}" if a.group else "") + ("_stu" if a.student else "")
    print("prereg", OUT / "prereg.yaml", "paper current/paper_snv.md", "run", tag, flush=True)
    unit_tests()
    torch.manual_seed(a.seed)
    w = W.World(rot=a.rot)
    contents, pairs = W.design(w, a.K, cross_sex=not a.nocross, n_utt=a.n_utt)
    rng = np.random.default_rng(2000 + a.seed)
    U, Y, S, C = [], [], [], []
    for s, ci in pairs:
        u = w.playing(contents[ci], rng)
        U.append(u.astype(np.float32)), Y.append(w.render(u, s, rng).astype(np.float32)), S.append(s), C.append(ci)
    by_spk: dict = {}
    for i, s in enumerate(S):
        by_spk.setdefault(s, []).append(i)
    nseg = np.array([len(c) for c in contents])
    trng = np.random.default_rng(3000)
    test = {s: [w.playing(w.content(trng), trng) for _ in range(N_ENR + N_TU)] for s in w.test}
    testy = {s: [w.render(u, s, trng).astype(np.float32) for u in us] for s, us in test.items()}
    print(f"utts {len(U)} contents {len(contents)} speakers {len(by_spk)} K {a.K} cross {not a.nocross}", flush=True)

    St, E, Fn = Struct().to(DEV), Net(W.D_Y, 2 * W.D_U).to(DEV), Net(W.D_U + a.side, W.D_Y).to(DEV)
    by_c: dict = {}
    for i, ci_ in enumerate(C):
        by_c.setdefault(ci_, []).append(i)
    assert not a.group or min(len(v) for v in by_c.values()) >= a.group, "束の組の声の数が内容の声の数を超える"
    params = list(St.parameters()) + list(E.parameters()) + list(Fn.parameters())
    learn_b = a.arm in ("b_vae", "d_tpl", "f_share")
    logb = nn.Parameter(torch.full((W.D_Y,), math.log(B_FIX), device=DEV), requires_grad=learn_b)
    if learn_b:
        params.append(logb)
    tpl_on = a.arm.startswith("d_tpl")
    groups = [{"params": params}]
    if tpl_on:
        tpl = nn.Parameter(0.1 * torch.randn(len(contents), 2 * int(nseg.max()), W.D_U, device=DEV), requires_grad=not a.tpl_ema)
        seen = torch.zeros(len(contents), dtype=torch.bool, device=DEV)
        logsig = nn.Parameter(torch.zeros(W.D_U, device=DEV))
        params += [logsig]
        if not a.tpl_ema:
            groups.append({"params": [tpl], "lr": LR * a.tpl_lr})
    Er = None
    if a.arm == "f_share":
        if a.tpl_est in ("ste", "gste", "gswap", "gloo"):
            tpl = torch.zeros(len(contents), 2 * int(nseg.max()), W.D_U, device=DEV)
            seen = torch.zeros(len(contents), dtype=torch.bool, device=DEV)
        else:
            tpl = nn.Parameter(0.1 * torch.randn(len(contents), 2 * int(nseg.max()), W.D_U, device=DEV))
            groups.append({"params": [tpl]})
        from train_snv_i import mas_free
        hl: dict = {"vr_pm": [], "vr_u": [], "cover": [], "pm_sq": [], "fit": []} | ({"rho": []} if a.group else {})
        Er = Net(W.D_Y + W.D_U, 2 * W.D_U).to(DEV) if a.share_res == "kl" else None
        E2 = Net(W.D_Y, 2 * W.D_U).to(DEV) if a.student else None
        if E2 is not None:
            params += list(E2.parameters())
            hl["fit_s"] = []
        if Er is not None:
            params += list(Er.parameters())
    erng = np.random.default_rng(5000 + a.seed)
    spk_ix = {sp: k for k, sp in enumerate(sorted(by_spk))}
    Q = Critic(len(spk_ix)).to(DEV) if a.critic else None
    opt_q = torch.optim.Adam(Q.parameters(), lr=LR) if Q is not None else None
    r2s: list = []
    opt = torch.optim.Adam(groups, lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    brng = np.random.default_rng(4000 + a.seed)

    def refs(s: int, skip: int) -> np.ndarray:
        idx = [i for i in by_spk[s] if i != skip]
        return np.concatenate([Y[i] for i in brng.choice(idx, N_REF, replace=False)])

    t0 = time.time()
    for step in range(1, a.steps + 1):
        if a.group:
            G_ = BS // a.group
            cs = brng.choice(len(contents), G_, replace=False)
            bi = np.concatenate([brng.choice(by_c[int(c_)], a.group, replace=False) for c_ in cs])
            rr_ = brng.random(G_) * (1.0 - a.crop)
            lc = [max(8, int(round(a.crop * len(Y[i])))) for i in bi]
            off = [min(len(Y[i]) - l_, int(round(rr_[k // a.group] * len(Y[i])))) for k, (i, l_) in enumerate(zip(bi, lc))]
            y, m = pad([Y[i][o:o + l_] for i, o, l_ in zip(bi, off, lc)])
            ut, _ = pad([U[i][o:o + l_] for i, o, l_ in zip(bi, off, lc)])
        elif a.crop:
            bi = brng.choice(len(U), BS, replace=False)
            lc = [max(8, int(round(a.crop * len(Y[i])))) for i in bi]
            off = [int(brng.integers(0, len(Y[i]) - l_ + 1)) for i, l_ in zip(bi, lc)]
            y, m = pad([Y[i][o:o + l_] for i, o, l_ in zip(bi, off, lc)])
            ut, _ = pad([U[i][o:o + l_] for i, o, l_ in zip(bi, off, lc)])
        else:
            bi = brng.choice(len(U), BS, replace=False)
            off = [0] * len(bi)
            y, m = pad([Y[i] for i in bi])
            ut, _ = pad([U[i] for i in bi])
        offt = torch.tensor(off, device=DEV)
        nall = torch.tensor([len(Y[i]) for i in bi], device=DEV)
        ry, rm = pad([refs(S[i], i) for i in bi])
        phi = St(ry, rm)
        if a.arm == "oracle":
            yh = Fn(ut, phi)
            loss = ((yh - y).abs().mean(1) * m).sum() / m.sum()
        elif a.arm == "f_share":
            mu = E(y, phi).chunk(2, 1)[0]
            ci = torch.tensor([C[i] for i in bi], device=DEV)
            ll = torch.tensor([2 * nseg[C[i]] for i in bi], device=DEV)
            n = m.sum(1).long()
            with torch.no_grad():
                if step <= a.align_warm:
                    tt = torch.arange(m.shape[1], device=DEV)[None].float()
                    path = torch.minimum(((offt[:, None] + tt) / nall[:, None].float() * ll[:, None].float()).long(), ll[:, None] - 1) * (tt < n[:, None]).long()
                else:
                    T_ = tpl[ci].detach()
                    lp = -0.5 * ((mu.detach().transpose(1, 2)[:, :, None, :] - T_[:, None, :, :]) ** 2).sum(-1)
                    lp = lp.masked_fill(~(torch.arange(T_.shape[1], device=DEV)[None, None] < ll[:, None, None]), -1e9)
                    if a.align == "fixed":
                        path = mas(lp, n, ll)
                    else:
                        path = mas_free(band_mask(lp, offt, nall, ll, a.band) if a.align == "band" else lp, n, ll)
                if a.tpl_est in ("ste", "gste", "gswap", "gloo"):
                    if a.group:
                        ema_vec(tpl, seen, ci, path, mu.detach(), m, ll, a.tpl_gamma)
                    else:
                        ema_update(tpl, seen, ci, path, mu.detach(), m, ll, a.tpl_gamma, np.ones(len(bi), bool))
            pm = torch.gather(tpl[ci], 1, path[..., None].expand(-1, -1, W.D_U)).transpose(1, 2)
            if step > a.steps - 1000:
                with torch.no_grad():
                    hl["vr_pm"].append(var_ratio(pm, m)), hl["vr_u"].append(var_ratio(ut, m)), hl["pm_sq"].append(float(((pm ** 2).sum(1) * m).sum() / m.sum()))
                    hl["cover"].append(float(np.mean([len(set(path[k, :int(n[k])].tolist())) / max(1.0, float(n[k]) / float(nall[k]) * float(ll[k])) for k in range(len(bi))])))
                    gm = (pm * m[:, None]).sum((0, 2)) / m.sum()
                    tv = (((pm - gm[None, :, None]) ** 2).sum(1) * m).sum() / m.sum()
                    hl["fit"].append(float((((mu - pm) ** 2).sum(1) * m).sum() / m.sum() / tv.clamp(min=1e-12)))
                    if E2 is not None:
                        hl["fit_s"].append(float((((E2(y, phi).chunk(2, 1)[0] - pm) ** 2).sum(1) * m).sum() / m.sum() / tv.clamp(min=1e-12)))
                    if a.group:
                        from train_snv_i import own_share
                        hl["rho"].append(own_share(path, m, torch.arange(len(bi), device=DEV) // a.group, tpl.shape[1]))
            if Er is not None:
                rmu, rlv = Er(torch.cat([y, pm.detach()], 1), phi).chunk(2, 1)
                rlv = rlv.clamp(-10.0, 4.0)
                rr = rmu + (0.5 * rlv).exp() * torch.randn_like(rmu)
                kl = ((0.5 * (rmu * rmu + rlv.exp() - 1 - rlv).sum(1)) * m).sum() / m.sum()
            else:
                rr = torch.zeros_like(pm)
                kl = torch.zeros((), device=DEV)
            if a.tpl_est in ("gswap", "gloo"):
                from train_snv_i import group_ste as group_ste_i
                xc = group_ste_i(mu, pm, path, m, torch.arange(len(bi), device=DEV) // a.group, tpl.shape[1], swap=a.tpl_est == "gswap", loo=a.tpl_est == "gloo")
            else:
                xc = group_ste(mu, pm, path, m, a.group, tpl.shape[1]) if a.tpl_est == "gste" else (mu + (pm - mu).detach() if a.tpl_est == "ste" else pm)
            yh = Fn(torch.cat([xc + rr, side_feats(y, ry, rm, a.side)], 1) if a.side else xc + rr, phi)
            b = logb.exp()[None, :, None]
            nll = (((yh - y).abs() / b + torch.log(2 * b)).sum(1) * m).sum() / m.sum()
            le = (((mu - (pm + rr).detach()) ** 2).sum(1) * m).sum() / m.sum()
            loss = nll + min(1.0, step / BETA_RAMP) * kl + le + 1e-3 * ((pm ** 2).sum(1) * m).sum() / m.sum()
            if E2 is not None:
                loss = loss + (((E2(y, phi.detach()).chunk(2, 1)[0] - (pm + rr).detach()) ** 2).sum(1) * m).sum() / m.sum()
            if Q is not None and Er is not None:
                res_r = rmu
                lab = torch.tensor([spk_ix[S[i]] for i in bi], device=DEV)[:, None].expand(-1, rmu.shape[-1])
                ce = lambda lg: (F_.cross_entropy(lg, lab, reduction="none") * m).sum() / m.sum()
                loss = loss - a.critic * ce(Q(res_r))
        else:
            mu, lv = E(y, phi).chunk(2, 1)
            lv = lv.clamp(-10.0, 4.0)
            if a.arm in ("a_det", "c_white"):
                uu = mu
                if a.arm == "c_white":
                    with torch.no_grad():
                        rmu = E(ry, phi).chunk(2, 1)[0]
                        st = zca(rmu, rm)
                    uu = torch.einsum("bij,bjt->bit", st[1], uu - st[0][..., None])
                yh = Fn(uu, phi)
                loss = ((yh - y).abs().mean(1) * m).sum() / m.sum()
            else:
                uu = mu + (0.5 * lv).exp() * torch.randn_like(mu)
                yh = Fn(uu, phi)
                b = logb.exp()[None, :, None]
                nll = (((yh - y).abs() / b + torch.log(2 * b)).sum(1) * m).sum() / m.sum()
                if tpl_on:
                    ci = torch.tensor([C[i] for i in bi], device=DEV)
                    T_ = tpl[ci]
                    ll = torch.tensor([2 * nseg[C[i]] for i in bi], device=DEV)
                    sig2 = (2 * logsig).exp()
                    with torch.no_grad():
                        dif = mu.detach().transpose(1, 2)[:, :, None, :] - T_.detach()[:, None, :, :]
                        lp = -0.5 * ((dif ** 2) / sig2).sum(-1)
                        lmask = torch.arange(T_.shape[1], device=DEV)[None, None] < ll[:, None, None]
                        lp = lp.masked_fill(~lmask, -1e9)
                        path = mas(lp, m.sum(1).long(), ll)
                        if a.tpl_ema:
                            ema_update(tpl, seen, ci, path, mu.detach(), m, ll, a.tpl_gamma, erng.random(len(bi)) < a.tpl_ev)
                            T_ = tpl[ci]
                    pm = torch.gather(T_, 1, path[..., None].expand(-1, -1, W.D_U)).transpose(1, 2)
                    kl = (0.5 * (2 * logsig[None, :, None] - lv + (lv.exp() + (mu - pm) ** 2) / sig2[None, :, None] - 1)).sum(1)
                    if step > a.steps - 1000:
                        with torch.no_grad():
                            mb = (mu * m[:, None]).sum((0, 2)) / m.sum()
                            r2s.append(1.0 - float((((mu - pm) ** 2) * m[:, None]).sum() / (((mu - mb[None, :, None]) ** 2) * m[:, None]).sum()))
                    if Q is not None:
                        res_r = mu - pm.detach()
                        lab = torch.tensor([spk_ix[S[i]] for i in bi], device=DEV)[:, None].expand(-1, mu.shape[-1])
                        ce = lambda lg: (F_.cross_entropy(lg, lab, reduction="none") * m).sum() / m.sum()
                else:
                    kl = 0.5 * (mu * mu + lv.exp() - 1 - lv).sum(1)
                kl = (kl * m).sum() / m.sum()
                loss = nll + min(1.0, step / BETA_RAMP) * kl
                if tpl_on and Q is not None:
                    loss = loss - a.critic * ce(Q(res_r))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for g in groups for q in g["params"]], 5.0)
        opt.step()
        sched.step()
        if Q is not None and ((tpl_on and a.arm.startswith("d_tpl")) or (a.arm == "f_share" and Er is not None)):
            opt_q.zero_grad(set_to_none=True)
            qloss = ce(Q(res_r.detach()))
            qloss.backward()
            opt_q.step()
        if step % 1000 == 0:
            extra = f" kl {float(kl):.2f} nll {float(nll):.2f} b {float(logb.exp().mean()):.4f}" if a.arm not in ("a_det", "c_white", "oracle") else ""
            extra += f" le {float(le):.4f}" if a.arm == "f_share" else ""
            extra += f" q_ce {float(qloss):.3f}" if Q is not None else ""
            print(f"step {step} loss {float(loss):.4f}{extra} {(time.time() - t0) / step * 1000:.1f}ms/step", flush=True)

    St.eval(), E.eval(), Fn.eval()
    Einf = E2 if a.arm == "f_share" and E2 is not None else E
    res = evaluate(a, w, test, testy, St, Einf, Fn)
    if a.arm != "oracle":
        from snv_diag.i3_share_probe import r2_loo
        by_c: dict = {}
        for i, ci_ in enumerate(C):
            by_c.setdefault(ci_, []).append(i)
        vals = []
        with torch.no_grad():
            for ci_ in [c_ for c_, ii in sorted(by_c.items()) if len(ii) >= 8][:20]:
                xs = []
                for i in by_c[ci_][:8]:
                    rr_, rm_ = pad([refs(S[i], i)])
                    yy, _ = pad([Y[i]])
                    xs.append(Einf(yy, St(rr_, rm_)).chunk(2, 1)[0][0].T.cpu().numpy())
                vals.append(r2_loo(xs))
        res["share_r2"] = round(float(np.mean(vals)), 4)
    res |= {"run": tag, "arm": a.arm, "K": a.K, "cross": not a.nocross, "steps": a.steps, "nref": a.nref, "dphi": a.dphi, "device": a.device, "utts": len(U), "contents": len(contents), "learned_b": float(logb.exp().mean()), "sec": round(time.time() - t0, 1),
            "tpl_r2_last1k": round(float(np.mean(r2s)), 4) if r2s else None, "critic": a.critic, "q_ce_last": float(qloss) if Q is not None else None, "q_chance": math.log(len(spk_ix)),
            "n_utt": a.n_utt, "crop": a.crop, "align": a.align, "band": a.band, "tpl_est": a.tpl_est, "align_warm": a.align_warm, "visits_per_content": round(a.steps * BS / len(contents), 1)}
    if a.arm == "f_share":
        res |= {k: round(float(np.mean(v)), 4) for k, v in hl.items()} | {"le_last": round(float(le), 4)}
        with torch.no_grad():
            tq = tpl.detach()
            ac = {}
            for lag in (1, 4):
                cs_ = [float(F_.cosine_similarity(tq[c_, lag:2 * nseg[c_]], tq[c_, :2 * nseg[c_] - lag], dim=-1).mean()) for c_ in range(len(contents)) if 2 * nseg[c_] > lag]
                ac[f"tpl_ac{lag}"] = round(float(np.mean(cs_)), 4)
        res |= ac | {"side": a.side, "group": a.group, "student": a.student}
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print("result", json.dumps({k: v for k, v in res.items() if k not in ("per_pair",)}), flush=True)
    return 0


@torch.no_grad()
def evaluate(a, w: W.World, test: dict, testy: dict, St: Struct, E: Net, Fn: Net) -> dict:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    from sklearn.linear_model import Ridge
    spk = list(test)
    phi = {}
    for s in spk:
        ry, rm = pad([np.concatenate(testy[s][:N_ENR])])
        phi[s] = St(ry, rm)
    uhat, white = {}, {}
    for s in spk:
        ys, ms = pad(testy[s])
        mu = E(ys, phi[s].expand(len(testy[s]), -1)).chunk(2, 1)[0]
        uhat[s] = [mu[i, :, :len(testy[s][i])].T.cpu().numpy() for i in range(len(testy[s]))]
        if a.arm == "c_white":
            ry, rm = pad([np.concatenate(testy[s][:N_ENR])])
            white[s] = zca(E(ry, phi[s]).chunk(2, 1)[0], rm)
    errs, cross, same = [], [], []
    for o in spk:
        u_o, y_o = test[o][N_ENR], testy[o][N_ENR]
        src = torch.from_numpy((u_o if a.arm == "oracle" else uhat[o][N_ENR]).T.astype(np.float32))[None].to(DEV)
        if a.arm == "c_white":
            src = torch.einsum("bij,bjt->bit", white[o][1], src - white[o][0][..., None])
        if getattr(a, "side", 0):
            yo_t, _ = pad([y_o])
            ro_t, rmo = pad([np.concatenate(testy[o][:N_ENR])])
            src = torch.cat([src, side_feats(yo_t, ro_t, rmo, a.side)], 1)
        for t in spk:
            if t == o:
                continue
            truth = w.render(u_o, t)
            yh = Fn(src, phi[t])[0].T.cpu().numpy()
            e = float(np.abs(yh - truth).mean() / np.abs(y_o - truth).mean())
            errs.append(e)
            (cross if w.sex[o] != w.sex[t] else same).append(e)
    lab = lambda s: np.full(0, 0)
    tr_x, tr_y, te_x, te_y, tr_u, te_u = [], [], [], [], [], []
    for i, s in enumerate(spk):
        uh = uhat[s][N_ENR:]
        ut = test[s][N_ENR:]
        for k in range(N_TU):
            (tr_x if k < 2 else te_x).append(uh[k]), (tr_u if k < 2 else te_u).append(ut[k])
            (tr_y if k < 2 else te_y).append(np.full(len(uh[k]), i))
    trX, teX, trY, teY, trU, teU = map(np.concatenate, (tr_x, te_x, tr_y, te_y, tr_u, te_u))
    spk_acc = float(QDA(reg_param=0.1).fit(trX, trY).score(teX, teY))
    spk_true = float(QDA(reg_param=0.1).fit(trU, trY).score(teU, teY))
    r2 = lambda p, t: 1.0 - ((p - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum()
    shared = Ridge(alpha=1e-2).fit(trX, trU)
    per, sh = [], []
    for i in range(len(spk)):
        mt, me = trY == i, teY == i
        per.append(r2(Ridge(alpha=1e-2).fit(trX[mt], trU[mt]).predict(teX[me]), teU[me]))
        sh.append(r2(shared.predict(teX[me]), teU[me]))
    return {"e": round(float(np.mean(errs)), 4), "e_cross": round(float(np.mean(cross)), 4), "e_same": round(float(np.mean(same)), 4),
            "spk": round(spk_acc, 4), "spk_true_u": round(spk_true, 4), "corr_shared_r2": round(float(np.mean(sh)), 4), "corr_per_r2": round(float(np.mean(per)), 4),
            "corr": round(float(np.mean(sh) / max(np.mean(per), 1e-6)), 4), "pairs": len(errs)}


if __name__ == "__main__":
    raise SystemExit(main())
