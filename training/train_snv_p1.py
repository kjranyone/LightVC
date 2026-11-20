"""P1(synth_nam_vc §8): 製品の経路の上の 演奏 E + 回路 C(表の型 x)を、自己の reamp と自前の reamp の循環で学ぶ。前段 = P0a(snv_p0_front v3・data/snv_p_feat)・描画 = rvoc3hi(固定)。
特徴の空間 Y(29)= 描画器の界面から高さを除いたもの [c0n・c1..c24 / 10・周期性 × 有声]。高さ(lf = log(f0 / 200)・有声)は明示の入力で、E には渡さない。
E(因果)の入力 = [包絡 − その話者の登録の平均(有声のフレーム)/ 10・周期性・有声](30)→ 演奏 a(64)。C(因果)= [a・lf・有声] を、目標の参照(同じ声の普通の発話 2 本)を
a で読む注意(鍵 = E(参照)・値 = 参照の Y)と参照の大域の要約(FiLM)で Y へ描く。
損失: self = C(E(y_A), 参照_A) ≈ y_A。rend(両腕)= C(E(Ŷ_A), 参照_A) ≈ y_A(Ŷ_A = 自己の描き直し・勾配を止める)。
自前の reamp Ŷ_B = C(a_A, 参照_B, lf_A + k 半音)(k ∈ {−12, −7, −5, 0, 5, 7, 12}・B = 同じ束の別の声)= 同じ演奏を別の回路に通した音。
judge(腕 j・wj)= 固定した自前の判定器 J(train_snv_p1j)が Ŷ_B を B と読む圧(交差エントロピー × LAM_J・勾配は C と E へ・RAMP の間に 0 → 1 で掛ける)。
cyc(腕 wj)= C(E(Ŷ_B), 参照_A) ≈ y_A(循環 = 交換を通っても演奏が残る・勾配は Ŷ_B の生成にも流す)。
事前登録 = results/snv_p1/prereg.yaml。
    cd training && uv run --no-sync python train_snv_p1.py --arm n --tag snv_p1_n
    cd training && uv run --no-sync python train_snv_p1.py --arm wj --tag snv_p1_wj --judge results/snv_p1_judge/judge.pt
    cd training && uv run --no-sync python train_snv_p1.py --arm wj --tag snv_p1c_wc --judge results/snv_p1c_judge/judge.pt --prereg results/snv_p1c/prereg.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pae as PA

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / os.environ.get("SNV_FEAT", "data/snv_p_feat")
ENVC = os.environ.get("SNV_ENVC", "")
D_Y = 29
D_X = 30
D_A = 64
CROP = 800
WARM_SELF = 200
WARM_CYC = 400
REF_UTT = 2
REF_MAX = 1000
P_NV = 0.5
SEMIS = (-12, -7, -5, 0, 5, 7, 12)
F0_LO, F0_HI = 70.0, 900.0
VAL_F = 30
VAL_M = 6
LN2_12 = math.log(2.0) / 12
LAM_J = 0.05
RAMP = (4000, 8000)
J_SEG = 300


def to_y(env: np.ndarray, per: np.ndarray, f0: np.ndarray) -> np.ndarray:
    v = (f0 > 0).astype(np.float32)
    return np.concatenate([((env[0] - PA.C0_SIL) / 10)[None], env[1:25] / 10, np.clip(per, 0, 1.5) * v[None]], 0).astype(np.float32)


def y_env(y: torch.Tensor) -> torch.Tensor:
    return torch.cat([y[:, :1] * 10 + PA.C0_SIL, y[:, 1:25] * 10], 1)


def e_in(y: torch.Tensor, v: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return torch.cat([(y_env(y) - mu[..., None]) / 10, y[:, 25:29], v[:, None]], 1)


def voiced_mean(env: np.ndarray, f0: np.ndarray) -> tuple[np.ndarray, int]:
    m = f0 > 0
    return env[:, m].sum(1), int(m.sum())


def split(voices: list[str]) -> tuple[list[str], list[str]]:
    fem = sorted(v for v in voices if v.startswith("ef"))
    mal = sorted(v for v in voices if v.startswith("em"))
    val = fem[:: max(1, len(fem) // VAL_F)][:VAL_F] + mal[-VAL_M:]
    return [v for v in voices if v not in val], val


def load_utt(p: Path) -> dict:
    d = np.load(p)
    f0, env, per = d["f0"], d["env"], d["per"]
    s, n = voiced_mean(env, f0)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    out = {"y": to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0.astype(np.float32), "vs": s.astype(np.float32), "vn": n}
    if ENVC:
        q = Path(str(p).replace("data/snv_p_feat", "data/snv_p_feat_st", 1))
        if q.exists():
            out["yc"] = to_y(np.load(q)["env"], per, f0)
    return out


def code_y(u: dict) -> dict:
    """段 P91: 符号が読む特徴の辞書(SNV_ENVC のとき、包絡を固定の窓の包絡 yc に替えたもの)。"""
    return (u | {"y": u["yc"]}) if "yc" in u else u


def load_corpus(nv: bool = True) -> dict:
    out = {}
    for vd in sorted((FEAT / "e1").iterdir()):
        norm = [load_utt(p) for p in sorted(vd.glob("t*.npz"))]
        if len(norm) < REF_UTT + 1:
            continue
        nvl = [load_utt(p) for p in sorted((FEAT / "e1nv" / vd.name).glob("*.npz"))] if nv else []
        out[vd.name] = {"norm": norm, "nv": nvl}
    return out


class Batches:
    def __init__(self, corpus: dict, names: list[str], seed: int, p_nv: float = P_NV) -> None:
        self.c, self.names, self.p_nv = corpus, names, p_nv
        self.cls = {n: i for i, n in enumerate(names)}
        self.rng = random.Random(seed)

    def _crop(self, u: dict, L: int) -> tuple:
        T = u["y"].shape[1]
        s = self.rng.randint(0, max(0, T - L))
        n = min(L, T - s)
        y = np.zeros((D_Y, L), np.float32)
        lf = np.zeros(L, np.float32)
        v = np.zeros(L, np.float32)
        ok = np.zeros(L, np.float32)
        y[:, :n], lf[:n], v[:n], ok[:n] = u["y"][:, s:s + n], u["lf"][s:s + n], u["v"][s:s + n], 1.0
        return y, lf, v, ok, n

    def sample(self, bs: int) -> dict:
        Y, LF, V, OK, RY, RV, RL, MU, F0M, CL = [], [], [], [], [], [], [], [], [], []
        for _ in range(bs):
            name = self.rng.choice(self.names)
            CL.append(self.cls[name])
            d = self.c[name]
            idx = list(range(len(d["norm"])))
            self.rng.shuffle(idx)
            use_nv = bool(d["nv"]) and self.rng.random() < self.p_nv
            tgt = self.rng.choice(d["nv"]) if use_nv else d["norm"][idx[0]]
            refs = [d["norm"][i] for i in (idx[1:1 + REF_UTT] if not use_nv else idx[:REF_UTT])]
            y, lf, v, ok, _ = self._crop(tgt, CROP)
            Y.append(y), LF.append(lf), V.append(v), OK.append(ok)
            vf = tgt["f0"][tgt["f0"] > 0]
            F0M.append(float(np.median(vf)) if len(vf) else 200.0)
            ry, rv = [], []
            for r in refs:
                yy, _, vv, _, n = self._crop(r, REF_MAX)
                ry.append(yy[:, :n]), rv.append(vv[:n])
            RY.append(np.concatenate(ry, 1)), RV.append(np.concatenate(rv))
            RL.append(RY[-1].shape[1])
            s = sum(r["vs"] for r in refs)
            n = max(1, sum(r["vn"] for r in refs))
            MU.append(s / n)
        L = max(RL)
        ry = np.zeros((bs, D_Y, L), np.float32)
        rv = np.zeros((bs, L), np.float32)
        rm = np.zeros((bs, L), bool)
        for i, (a, b, l) in enumerate(zip(RY, RV, RL)):
            ry[i, :, :l], rv[i, :l], rm[i, :l] = a, b, True
        t = lambda z: torch.from_numpy(np.stack(z) if isinstance(z, list) else z)
        return {"y": t(Y), "lf": t(LF), "v": t(V), "ok": t(OK), "ry": t(ry), "rv": t(rv), "rmask": t(rm), "mu": t(MU), "f0m": torch.tensor(F0M), "cls": torch.tensor(CL)}


class GBlock(nn.Module):
    def __init__(self, ch: int, dil: int, zdim: int = 0) -> None:
        super().__init__()
        self.dil = dil
        self.conv = nn.Conv1d(ch, 2 * ch, 3, dilation=dil)
        self.film = nn.Linear(zdim, 2 * ch) if zdim else None
        self.out = nn.Conv1d(ch, ch, 1)

    def forward(self, h: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        a, b = self.conv(F.pad(h, (2 * self.dil, 0))).chunk(2, 1)
        if self.film is not None:
            g, s = self.film(z).chunk(2, -1)
            a = a * (1 + g[..., None]) + s[..., None]
        return h + self.out(torch.tanh(a) * torch.sigmoid(b))


class Enc(nn.Module):
    def __init__(self, ch: int = 256) -> None:
        super().__init__()
        self.inp = nn.Conv1d(D_X, ch, 1)
        self.blocks = nn.ModuleList([GBlock(ch, d) for d in (1, 2, 4, 8, 16, 32)])
        self.outp = nn.Conv1d(ch, D_A, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.outp(h)


class Circ(nn.Module):
    def __init__(self, ch: int = 256, zdim: int = 256, heads: int = 4, dk: int = 64, dv: int = 16) -> None:
        super().__init__()
        self.heads, self.dk, self.dv = heads, dk, dv
        self.inp = nn.Conv1d(D_A + 2, ch, 1)
        self.blocks = nn.ModuleList([GBlock(ch, d, zdim) for d in (1, 2, 4, 8, 16, 32) * 2])
        self.head = nn.Conv1d(ch, D_Y, 1)
        self.genc = nn.Sequential(nn.Conv1d(D_Y, ch, 5, padding=2), nn.ReLU(), nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(), nn.Conv1d(ch, ch, 5, padding=2))
        self.gz = nn.Linear(ch, zdim)
        self.q = nn.Linear(ch, heads * dk)
        self.k = nn.Sequential(nn.Linear(D_A, ch), nn.ReLU(), nn.Linear(ch, heads * dk))
        self.venc = nn.Sequential(nn.Conv1d(D_Y, ch, 3, padding=1), nn.ReLU(), nn.Conv1d(ch, heads * dv, 3, padding=1))
        self.o = nn.Linear(heads * dv, ch)

    def structure(self, ry: torch.Tensor, ra: torch.Tensor, rmask: torch.Tensor) -> dict:
        w = rmask[:, None].float()
        z = self.gz((self.genc(ry) * w).sum(-1) / w.sum(-1).clamp(min=1.0))
        B, Tr = rmask.shape
        k = self.k(ra.transpose(1, 2)).view(B, Tr, self.heads, self.dk).transpose(1, 2)
        v = self.venc(ry).transpose(1, 2).reshape(B, Tr, self.heads, self.dv).transpose(1, 2)
        return {"z": z, "k": k, "v": v, "rmask": rmask}

    def forward(self, a: torch.Tensor, lf: torch.Tensor, v: torch.Tensor, s: dict) -> torch.Tensor:
        h = self.inp(torch.cat([a, lf[:, None], v[:, None]], 1))
        for i, blk in enumerate(self.blocks):
            h = blk(h, s["z"])
            if i == 3:
                B, C, T = h.shape
                q = self.q(h.transpose(1, 2)).view(B, T, self.heads, self.dk).transpose(1, 2)
                ctx = ref_attention(q, s["k"], s["v"], s["rmask"]).transpose(1, 2).reshape(B, T, self.heads * self.dv)
                h = h + self.o(ctx).transpose(1, 2)
        return self.head(h)


class CircQ(nn.Module):
    def __init__(self, ch: int = 256, zdim: int = 256, heads: int = 4, dk: int = 64, dv: int = 16, smax: float = 6.0) -> None:
        super().__init__()
        self.heads, self.dk, self.dv, self.smax = heads, dk, dv, smax
        self.inp = nn.Conv1d(D_A + 2, ch, 1)
        self.qblocks = nn.ModuleList([GBlock(ch, d, zdim) for d in (1, 2, 4, 8, 16, 32)])
        self.dinp = nn.Conv1d(heads * dv + 2, ch, 1)
        self.dblocks = nn.ModuleList([GBlock(ch, d, zdim) for d in (1, 2, 4, 8, 16, 32)])
        self.head = nn.Conv1d(ch, D_Y, 1)
        self.genc = nn.Sequential(nn.Conv1d(D_Y, ch, 5, padding=2), nn.ReLU(), nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(), nn.Conv1d(ch, ch, 5, padding=2))
        self.gz = nn.Linear(ch, zdim)
        self.q = nn.Linear(ch, heads * dk)
        self.k = nn.Sequential(nn.Linear(D_A, ch), nn.ReLU(), nn.Linear(ch, heads * dk))
        self.venc = nn.Sequential(nn.Conv1d(D_Y, ch, 3, padding=1), nn.ReLU(), nn.Conv1d(ch, heads * dv, 3, padding=1))
        self.ls = nn.Parameter(torch.zeros(()))

    def structure(self, ry: torch.Tensor, ra: torch.Tensor, rmask: torch.Tensor) -> dict:
        w = rmask[:, None].float()
        z = self.gz((self.genc(ry) * w).sum(-1) / w.sum(-1).clamp(min=1.0))
        B, Tr = rmask.shape
        k = F.normalize(self.k(ra.transpose(1, 2)).view(B, Tr, self.heads, self.dk), dim=-1).transpose(1, 2)
        v = self.venc(ry).transpose(1, 2).reshape(B, Tr, self.heads, self.dv).transpose(1, 2)
        return {"z": z, "k": k, "v": v, "rmask": rmask}

    def weights(self, a: torch.Tensor, lf: torch.Tensor, v: torch.Tensor, s: dict) -> torch.Tensor:
        h = self.inp(torch.cat([a, lf[:, None], v[:, None]], 1))
        for blk in self.qblocks:
            h = blk(h, s["z"])
        B, _, T = h.shape
        q = F.normalize(self.q(h.transpose(1, 2)).view(B, T, self.heads, self.dk), dim=-1).transpose(1, 2)
        logit = (q @ s["k"].transpose(-1, -2)) * (self.smax * torch.sigmoid(self.ls))
        return logit.masked_fill(~s["rmask"][:, None, None, :], -1e9).softmax(-1)

    def forward(self, a: torch.Tensor, lf: torch.Tensor, v: torch.Tensor, s: dict) -> torch.Tensor:
        p = self.weights(a, lf, v, s)
        B, _, T, _ = p.shape
        ctx = (p @ s["v"]).transpose(1, 2).reshape(B, T, self.heads * self.dv).transpose(1, 2)
        h = self.dinp(torch.cat([ctx, lf[:, None], v[:, None]], 1))
        for blk in self.dblocks:
            h = blk(h, s["z"])
        return self.head(h)


ATTN_CHUNK = 1000


def ref_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rmask: torch.Tensor) -> torch.Tensor:
    from torch.utils.checkpoint import checkpoint
    m = rmask[:, None, None, :]
    f = lambda qq: F.scaled_dot_product_attention(qq, k, v, attn_mask=m)
    L = q.shape[2]
    if L <= ATTN_CHUNK:
        return f(q)
    return torch.cat([checkpoint(f, q[:, :, i:i + ATTN_CHUNK], use_reentrant=False) if torch.is_grad_enabled() else f(q[:, :, i:i + ATTN_CHUNK]) for i in range(0, L, ATTN_CHUNK)], 2)


def masked_l1(y: torch.Tensor, t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    w = m[:, None]
    return ((y - t).abs() * w).sum() / (w.sum() * D_Y).clamp(min=1.0)


def structure_of(E: Enc, C: Circ, ry: torch.Tensor, rv: torch.Tensor, rmask: torch.Tensor, mu: torch.Tensor) -> dict:
    return C.structure(ry, E(e_in(ry, rv, mu)), rmask)


def pick_shift(f0m: torch.Tensor, rng: random.Random) -> torch.Tensor:
    ks = []
    for f in f0m.tolist():
        ok = [k for k in SEMIS if F0_LO <= f * 2 ** (k / 12) <= F0_HI]
        ks.append(rng.choice(ok) if ok else 0)
    return torch.tensor(ks, dtype=torch.float32)


def unit_tests() -> None:
    torch.manual_seed(0)
    E, C = Enc(), Circ()
    x = torch.randn(1, D_X, 300)
    a1 = E(x)
    x2 = x.clone()
    x2[..., 200:] = torch.randn(1, D_X, 100)
    a2 = E(x2)
    assert torch.allclose(a1[..., :200], a2[..., :200], atol=1e-5) and not torch.allclose(a1[..., 200:], a2[..., 200:]), "E が因果でない"
    s = C.structure(torch.randn(1, D_Y, 80), torch.randn(1, D_A, 80), torch.ones(1, 80, dtype=torch.bool))
    lf, v = torch.randn(1, 300), torch.ones(1, 300)
    y1 = C(a1, lf, v, s)
    lf2 = lf.clone()
    lf2[:, 200:] = 0.3
    y2 = C(a2, lf2, v, s)
    assert torch.allclose(y1[..., :200], y2[..., :200], atol=1e-5) and not torch.allclose(y1[..., 200:], y2[..., 200:]), "C が因果でない"
    env = np.random.default_rng(0).normal(0, 3, (25, 50)).astype(np.float32)
    per = np.random.default_rng(1).uniform(0, 1, (4, 50)).astype(np.float32)
    f0 = np.full(50, 150.0, np.float32)
    y = torch.from_numpy(to_y(env, per, f0))[None]
    assert torch.allclose(y_env(y)[0], torch.from_numpy(env), atol=1e-4), "Y → 包絡の往復がずれる"
    mu = torch.from_numpy(env.mean(1))[None]
    xe = e_in(y, torch.ones(1, 50), mu)
    assert xe.shape == (1, D_X, 50) and torch.allclose(xe[0, :25].mean(1), torch.zeros(25), atol=1e-4), "E の入力の登録の平均の差し引きがずれる"
    ks = pick_shift(torch.tensor([100.0, 300.0, 850.0]), random.Random(0))
    assert all(F0_LO <= f * 2 ** (k / 12) <= F0_HI for f, k in zip((100.0, 300.0, 850.0), ks.tolist())), "半音のずらしが範囲を外れる"
    print("unit tests ok(E・C の因果性・Y の往復・登録の平均・ずらしの範囲)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["n", "j", "wj"], required=True)
    ap.add_argument("--judge", default="results/snv_p1_judge/judge.pt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prereg", default="results/snv_p1/prereg.yaml")
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (ROOT / a.prereg).exists():
        print(f"{a.prereg} が無い: 起動しない", flush=True)
        return 1
    print("prereg", ROOT / a.prereg, flush=True)
    print("ship ledger: results/snv_p1/ship.json", flush=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed + 7)
    unit_tests()
    t0 = time.time()
    corpus = load_corpus()
    tr, va = split(sorted(corpus))
    n_nv = sum(len(corpus[n]["nv"]) for n in tr)
    print(f"voices {len(corpus)} train {len(tr)} val {len(va)} (F {sum(v.startswith('ef') for v in va)} M {sum(v.startswith('em') for v in va)}) "
          f"utts {sum(len(corpus[n]['norm']) for n in tr)} nv {n_nv} load {time.time() - t0:.0f}s", flush=True)
    (out / "split.json").write_text(json.dumps({"train": tr, "val": va}))
    dev = "cuda"
    E, C = Enc().to(dev), Circ().to(dev)
    J = None
    if a.arm != "n":
        from train_snv_p1j import Judge
        sj = torch.load(ROOT / a.judge, map_location="cpu", weights_only=False)
        assert sj["classes"] == tr, "判定器の声の組が P1 の学習の声と違う"
        J = Judge(len(tr), norm=sj.get("norm", "none")).to(dev).eval().requires_grad_(False)
        J.load_state_dict(sj["J"])
        print("judge", a.judge, json.dumps(sj["res"]), flush=True)
    print(f"params E {sum(p.numel() for p in E.parameters()) / 1e6:.2f}M C {sum(p.numel() for p in C.parameters()) / 1e6:.2f}M arm {a.arm}", flush=True)
    params = list(E.parameters()) + list(C.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    trb, vab = Batches(corpus, tr, a.seed), Batches(corpus, va, 12345, p_nv=0.0)
    vbatches = [vab.sample(a.bs) for _ in range(8)]

    def to_dev(b: dict) -> dict:
        return {k: x.to(dev, non_blocking=True) for k, x in b.items()}

    @torch.no_grad()
    def evaluate() -> tuple[float, float]:
        E.eval(), C.eval()
        ls, lt = [], []
        for vb in vbatches:
            b = to_dev(vb)
            m = b["ok"].clone()
            m[:, :WARM_SELF] = 0
            s = structure_of(E, C, b["ry"], b["rv"], b["rmask"], b["mu"])
            ls.append(float(masked_l1(C(E(e_in(b["y"], b["v"], b["mu"])), b["lf"], b["v"], s), b["y"], m)))
            rmean = (b["ry"] * b["rmask"][:, None]).sum(-1, keepdim=True) / b["rmask"].sum(-1)[:, None, None].clamp(min=1)
            lt.append(float(masked_l1(rmean.expand_as(b["y"]), b["y"], m)))
        E.train(), C.train()
        return float(np.mean(ls)), float(np.mean(lt))

    v0, triv = evaluate()
    print(f"floors: untrained val L1 {v0:.4f} / trivial (参照の平均の Y) {triv:.4f}", flush=True)
    log = {"floors": {"untrained": v0, "trivial": triv}, "evals": []}
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = to_dev(trb.sample(a.bs))
        ms = b["ok"].clone()
        ms[:, :WARM_SELF] = 0
        mc = b["ok"].clone()
        mc[:, :WARM_CYC] = 0
        sA = structure_of(E, C, b["ry"], b["rv"], b["rmask"], b["mu"])
        aA = E(e_in(b["y"], b["v"], b["mu"]))
        yA = C(aA, b["lf"], b["v"], sA)
        l_self = masked_l1(yA, b["y"], ms)
        l_rend = masked_l1(C(E(e_in(yA.detach(), b["v"], b["mu"])), b["lf"], b["v"], sA), b["y"], mc)
        loss = l_self + l_rend
        l_j = torch.zeros((), device=dev)
        l_cyc = torch.zeros((), device=dev)
        j_b = j_a = 0.0
        ramp = min(1.0, max(0.0, (step - RAMP[0]) / (RAMP[1] - RAMP[0]))) if not a.smoke else 1.0
        if J is not None and ramp > 0:
            perm = torch.roll(torch.arange(a.bs, device=dev), 1)
            k = pick_shift(b["f0m"].cpu(), rng).to(dev)
            sB = structure_of(E, C, b["ry"][perm], b["rv"][perm], b["rmask"][perm], b["mu"][perm])
            yB = C(aA, b["lf"] + k[:, None] * LN2_12 * b["v"], b["v"], sB)
            seg, sm = yB[..., WARM_CYC:WARM_CYC + J_SEG], b["ok"][:, WARM_CYC:WARM_CYC + J_SEG]
            lg = J.logits(seg, sm)
            l_j = F.cross_entropy(lg, b["cls"].to(dev)[perm])
            pr = lg.argmax(1)
            j_b = float((pr == b["cls"].to(dev)[perm]).float().mean())
            j_a = float((pr == b["cls"].to(dev)).float().mean())
            loss = loss + ramp * LAM_J * l_j
            if a.arm == "wj":
                l_cyc = masked_l1(C(E(e_in(yB, b["v"], b["mu"][perm])), b["lf"], b["v"], sA), b["y"], mc)
                loss = loss + ramp * l_cyc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} self {float(l_self):.4f} rend {float(l_rend):.4f} judge {float(l_j):.3f} (B {j_b:.2f} A {j_a:.2f}) cyc {float(l_cyc):.4f} gn {float(gn):.2f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 2000 == 0 or step == a.steps:
            vl, _ = evaluate()
            rec = {"step": step, "val_l1": vl, "train_self": float(l_self), "train_rend": float(l_rend), "train_judge": float(l_j), "judge_reads_B": j_b, "judge_reads_A": j_a,
                   "train_cyc": float(l_cyc), "sec": time.time() - t0}
            log["evals"].append(rec)
            print("eval", json.dumps(rec), flush=True)
            ck = {"E": E.state_dict(), "C": C.state_dict(), "arm": a.arm, "step": step, "cfg": {"D_A": D_A, "CROP": CROP}}
            torch.save(ck, out / "last.pt")
            if step == a.steps // 2:
                torch.save(ck, out / f"step{step // 1000}k.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
            if step == 10000 and vl > 0.8 * triv:
                print(f"TRIPWIRE: val L1 {vl:.4f} > 0.8 × trivial {triv:.4f} at 10k → stop", flush=True)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
