"""P2(synth_nam_vc §8): 自前の因果の自己教師学習で、話者を跨いだ調音の状態の対応を持つ演奏の表現を作る(CPC・負例は同じ発話の中 = 話者は一定で対比に使えない)。
入力 x = 前段 v3 の Y から高さと水準を除いたもの [c1..c24 · 包絡 / σ・周期性 / σ・有声・Δc0 / σ](30・200 fps・先読み 0)。σ = コーパスの時間の標準偏差(定数)。
符号器 = 因果の畳み込み 2 段 → z_t(256)・文脈 = 因果の膨張畳み込み 6 段(受容野 0.63 s)→ c_t(256)。予測 = W_k c_t ≈ z_{t+k}(k = 1, 2, 4, 8, 12, 16, 24 フレーム = 5〜120 ms)・
InfoNCE(負例 16 = 同じ切り出しの別の時刻)。データ = P1 の学習の声 564 の E1 の普通の発話 + 音素の無い発声(検証の 36 声は関門 NNC のために除く)。
事前登録 = results/snv_p2/prereg.yaml・関門 = snv_p2_nnc.py P=<tag>。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p2_cpc
--aug(P2b)= 入力側の構造の乱し: 発話ごとに包絡の周波数の一律の伸縮 α ∈ [0.8, 1.25](対数一様・DCT 25 の上の線形写像)+ 傾き ±0.5(対数振幅・帯域の端から端)。
乱した入力の文脈 c から乱していない入力の未来 z を当てる InfoNCE を元の InfoNCE と平均(ContentVec と同じ原理 = c を伸縮と傾きに不変に)。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p2b_cpc --aug --prereg results/snv_p2b/prereg.yaml
--subset N = 学習の声を N 本おきに(P2c = 量の切り分け)。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p2c_cpc --aug --subset 4 --prereg results/snv_p2b/prereg.yaml
--inv λ(P2d)= 明示の不変性: 元・乱し 1・乱し 2 の 3 つの視点の c を、束の有効なフレームで次元ごとに標準化し(3 視点を合わせた平均と標準偏差)、組ごとの二乗の差の平均を λ 倍で足す。
切り出しの先頭 WARM_INV フレームは除く(因果の文脈が伸縮を推し量る前)。InfoNCE = 3 視点の文脈から元の未来を当てる平均。
--amin/--amax = 伸縮の域・--eq = 包絡の滑らかな音色の乱し(mel の軸の上の余弦 2〜4 次・各次の振幅 U(−eq, eq) 対数振幅)。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p2d_cpc --aug --inv 1.0 --amin 0.77 --amax 1.32 --eq 0.25 --prereg results/snv_p2d/prereg.yaml
--spk λ(P2e)= 話者の不変性: 元の c に勾配反転(−λ・最初の 5k step で 0 から上げる)を挟んだ因果の畳み込みの話者の識別器(受容野 0.3 s・学習の声 564 のどれか・フレームごと)。
識別器は c から話者を当てるように学び、c は当てられないように学ぶ(切り出しの先頭 WARM_INV フレームは除く)。勾配の切り詰めは CPC と識別器で別々。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p2e_cpc --aug --spk 0.5 --prereg results/snv_p2e/prereg.yaml
--white D(P10)= 符号器の中で話者の登録の分布で白くする: 切り出しごとに同じ話者の別の普通の発話 2 本(各 ≤ ENR フレーム)を登録にし、
c と z を共有の線形で D 次元へ射影して、登録の有声のフレームの平均と 2 次のモーメント(収縮 0.1・勾配を止める)で白くした c_w・z_w の上の InfoNCE(共有の予測 W_w)。
乱し(--aug)は切り出しと登録に同じ伸縮・傾きを掛ける(話者の変化の模擬 = 白くする段が外す)。損失 = 元の文脈と乱した文脈の c_w から元の z_w の未来を当てる平均。
演奏 = c_w(推論の統計 = 操作者の登録)。--init = 既存の CPC の重みから始める。
--pair λ(P14)= 同じ文を別の声が読んだ組(E1 の学習の声・train_snv_p13.text_pairs)の、全長の発話の文脈 c の有声のフレームの袋の MMD²(train_snv_p13.mmd2)を λ 倍で足す。
CPC(同じ発話の中の負例)には声を持つ得が無く、再現の損失も無い = 袋の差の罰に逆らう力が無い。演奏 = 凍結した c そのもの(平均も引かない)。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p14_pcpc --aug --pair 1.0 --init results/snv_p2b_cpc/last.pt --steps 20000 --prereg results/snv_p14/prereg.yaml
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p10_wcpc --aug --white 64 --init results/snv_p2b_cpc/last.pt --steps 20000 --prereg results/snv_p10/prereg.yaml
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
D_IN = 30
CH = 256
KS = (1, 2, 4, 8, 12, 16, 24)
N_NEG = 16
CROP = 512
MIN_GAP = 5
ENR = 600
N_PAIR = 4
MMD_FR = 256
N_ENR = 2
SHRINK = 0.1
ALPHAS = np.exp(np.linspace(np.log(0.8), np.log(1.25), 33))
TILT = 1.0
WARM_INV = 64


def x_of(u: dict, sig: np.ndarray) -> np.ndarray:
    y = u["y"]
    dc0 = np.concatenate([[0.0], np.diff(y[0])]).astype(np.float32)
    return np.concatenate([y[1:29] / sig[1:29, None], u["v"][None], (dc0 / sig[29])[None]], 0).astype(np.float32)


def corpus_scale(us: list[dict]) -> np.ndarray:
    s = np.mean([u["y"].std(1) for u in us], 0)
    d = np.mean([np.diff(u["y"][0]).std() for u in us])
    return np.concatenate([s, [d]]).astype(np.float32)


class CPC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1 = nn.Conv1d(D_IN, CH, 3)
        self.e2 = nn.Conv1d(CH, CH, 3, dilation=2)
        self.e3 = nn.Conv1d(CH, CH, 1)
        self.ar_in = nn.Conv1d(CH, CH, 1)
        self.ar = nn.ModuleList([T.GBlock(CH, d) for d in (1, 2, 4, 8, 16, 32)])
        self.W = nn.ModuleList([nn.Linear(CH, CH) for _ in KS])

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.e1(F.pad(x, (2, 0))))
        h = F.relu(self.e2(F.pad(h, (4, 0))))
        z = self.e3(h)
        z = F.layer_norm(z.transpose(1, 2), (CH,)).transpose(1, 2)
        c = self.ar_in(z)
        for b in self.ar:
            c = b(c)
        return z, c


def info_nce(z: torch.Tensor, c: torch.Tensor, m: torch.Tensor, W: nn.ModuleList, g: torch.Generator) -> tuple[torch.Tensor, list[float]]:
    B, C, L = z.shape
    zt, ct = z.transpose(1, 2), c.transpose(1, 2)
    losses, accs = [], []
    for k, Wk in zip(KS, W):
        n = L - k
        pred = Wk(ct[:, :n])
        sc = pred @ zt.transpose(1, 2) / math.sqrt(C)
        valid = (m[:, :n] * m[:, k:]) > 0
        off = torch.randint(MIN_GAP, L - MIN_GAP, (B, n, N_NEG), generator=g, device=z.device)
        ar = torch.arange(n, device=z.device)[None, :, None] + k
        idx = torch.cat([ar.expand(B, n, 1), (ar + off) % L], 2)
        logit = torch.gather(sc, 2, idx)
        lab = torch.zeros(int(valid.sum()), dtype=torch.long, device=z.device)
        lv = logit[valid]
        losses.append(F.cross_entropy(lv, lab))
        accs.append(float((lv.argmax(1) == 0).float().mean()))
    return torch.stack(losses).mean(), accs


def load(names: list[str], sig: np.ndarray | None = None) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
    us, lab = [], []
    for i, v in enumerate(names):
        for p in sorted((T.FEAT / "e1" / v).glob("t*.npz")) + sorted((T.FEAT / "e1nv" / v).glob("*.npz")):
            us.append(T.load_utt(p)), lab.append(i)
    if sig is None:
        sig = corpus_scale(us[::7])
    return [np.concatenate([u["y"], u["v"][None]], 0) for u in us], sig, lab


def batch(ys: list[np.ndarray], bs: int, rng: random.Random, lab: list[int] | None = None, js: list | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = np.zeros((bs, T.D_Y + 1, CROP), np.float32)
    M = np.zeros((bs, CROP), np.float32)
    S = np.zeros(bs, np.int64)
    for i in range(bs):
        j = rng.randrange(len(ys))
        if js is not None:
            js.append(j)
        x = ys[j]
        S[i] = lab[j] if lab is not None else 0
        L = x.shape[1]
        s = rng.randint(0, max(0, L - CROP))
        n = min(CROP, L - s)
        X[i, :, :n], M[i, :n] = x[:, s:s + n], 1.0
    return torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(S)


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g: torch.Tensor) -> tuple:
        return -ctx.lam * g, None


class SpkAdv(nn.Module):
    def __init__(self, n_spk: int) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(CH, CH, 5)
        self.c2 = nn.Conv1d(CH, CH, 5, dilation=2)
        self.c3 = nn.Conv1d(CH, CH, 5, dilation=4)
        self.out = nn.Conv1d(CH, n_spk, 1)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.c1(F.pad(c, (4, 0))))
        h = F.relu(self.c2(F.pad(h, (8, 0))))
        h = F.relu(self.c3(F.pad(h, (16, 0))))
        return self.out(h)


def spk_loss(adv: SpkAdv, c: torch.Tensor, m: torch.Tensor, S: torch.Tensor, lam: float) -> tuple[torch.Tensor, float]:
    logit = adv(GradRev.apply(c, lam))
    w = m.clone()
    w[:, :WARM_INV] = 0
    sel = w > 0
    lg = logit.transpose(1, 2)[sel]
    lab = S[:, None].expand_as(w)[sel]
    return F.cross_entropy(lg, lab), float((lg.argmax(1) == lab).float().mean())


def x_torch(yv: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
    y, v = yv[:, :T.D_Y], yv[:, T.D_Y:]
    dc0 = torch.cat([torch.zeros_like(y[:, :1, :1]), y[:, :1, 1:] - y[:, :1, :-1]], -1)
    return torch.cat([y[:, 1:29] / sig[1:29, None], v, dc0 / sig[29]], 1)


def warp_mats(lo: float = 0.8, hi: float = 1.25) -> np.ndarray:
    import snv_p0c_anchor as A
    eye = np.eye(25, dtype=np.float32)
    alphas = ALPHAS if (lo, hi) == (0.8, 1.25) else np.exp(np.linspace(np.log(lo), np.log(hi), 33))
    return np.stack([A.warp_env(eye, float(a)) for a in alphas]).astype(np.float32)


def tilt_vec() -> np.ndarray:
    import pae as PA
    _, dct = PA._np_consts()
    return (dct[:25] @ np.linspace(-0.5, 0.5, dct.shape[0])).astype(np.float32)


def eq_basis() -> np.ndarray:
    import pae as PA
    _, dct = PA._np_consts()
    m = np.linspace(0.0, 1.0, dct.shape[0])
    return np.stack([dct[:25] @ np.cos(np.pi * j * m) for j in (2, 3, 4)]).astype(np.float32)


def augment(yv: torch.Tensor, Ms: torch.Tensor, tv: torch.Tensor, g: torch.Generator, eqb: torch.Tensor | None = None, eq: float = 0.0, fixed: tuple | None = None) -> torch.Tensor:
    B = yv.shape[0]
    y = yv[:, :T.D_Y]
    env = torch.cat([y[:, :1] * 10 + T.PA.C0_SIL, y[:, 1:25] * 10], 1)
    ia, tau = fixed if fixed is not None else (torch.randint(0, len(Ms), (B,), generator=g, device=yv.device), (torch.rand(B, generator=g, device=yv.device) * 2 - 1) * TILT)
    ew = torch.einsum("bij,bjt->bit", Ms[ia], env) + tau[:, None, None] * tv[None, :, None]
    if eqb is not None and eq > 0:
        amp = (torch.rand(B, len(eqb), generator=g, device=yv.device) * 2 - 1) * eq
        ew = ew + (amp @ eqb)[:, :, None]
    yw = torch.cat([(ew[:, :1] - T.PA.C0_SIL) / 10, ew[:, 1:25] / 10, y[:, 25:]], 1)
    return torch.cat([yw, yv[:, T.D_Y:]], 1)


def white_stats(h: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = m[:, None]
    mu = (h * w).sum(-1) / w.sum(-1).clamp(min=1.0)
    d = (h - mu[..., None]).double()
    cov = torch.einsum("bit,bjt->bij", d * w.double(), d) / w.double().sum(-1).clamp(min=1.0)[..., None]
    D = cov.shape[-1]
    cov = (1 - SHRINK) * cov + SHRINK * (cov.diagonal(dim1=-2, dim2=-1).sum(-1) / D)[:, None, None] * torch.eye(D, dtype=cov.dtype, device=cov.device)
    e, q = torch.linalg.eigh(cov)
    return mu, ((q / e.clamp(min=1e-8).sqrt()[:, None, :]) @ q.transpose(-1, -2)).float()


def white_apply(h: torch.Tensor, st: tuple) -> torch.Tensor:
    return torch.einsum("bij,bjt->bit", st[1], h - st[0][..., None])


class WHead(nn.Module):
    def __init__(self, dw: int) -> None:
        super().__init__()
        self.pc, self.pz = nn.Linear(CH, dw), nn.Linear(CH, dw)
        self.W = nn.ModuleList([nn.Linear(dw, dw) for _ in KS])

    def proj(self, z: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pz(z.transpose(1, 2)).transpose(1, 2), self.pc(c.transpose(1, 2)).transpose(1, 2)


def enr_batch(ys: list[np.ndarray], picks: list[list[int]], rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(picks)
    X = np.zeros((B * N_ENR, T.D_Y + 1, ENR), np.float32)
    M = np.zeros((B * N_ENR, ENR), np.float32)
    for i, js in enumerate(picks):
        for e, j in enumerate(js):
            x = ys[j]
            s = rng.randint(0, max(0, x.shape[1] - ENR))
            n = min(ENR, x.shape[1] - s)
            X[i * N_ENR + e, :, :n], M[i * N_ENR + e, :n] = x[:, s:s + n], 1.0
    return torch.from_numpy(X), torch.from_numpy(M)


def inv_loss(cs: list[torch.Tensor], m: torch.Tensor) -> torch.Tensor:
    w = m.clone()
    w[:, :WARM_INV] = 0
    sel = w.reshape(-1) > 0
    flat = [c.transpose(1, 2).reshape(-1, c.shape[1])[sel] for c in cs]
    allc = torch.cat(flat)
    mu, sd = allc.mean(0), allc.std(0).clamp(min=1e-4)
    zs = [(f - mu) / sd for f in flat]
    pairs = [(i, j) for i in range(len(zs)) for j in range(i + 1, len(zs))]
    return torch.stack([((zs[i] - zs[j]) ** 2).mean() for i, j in pairs]).mean()


def unit_test() -> None:
    torch.manual_seed(0)
    m = CPC()
    x = torch.randn(1, D_IN, 300)
    z1, c1 = m.encode(x)
    x2 = x.clone()
    x2[..., 200:] = torch.randn(1, D_IN, 100)
    z2, c2 = m.encode(x2)
    assert torch.allclose(c1[..., :200], c2[..., :200], atol=1e-5) and torch.allclose(z1[..., :200], z2[..., :200], atol=1e-5), "CPC が因果でない"
    c3 = torch.randn(2, CH, 200)
    m3 = torch.ones(2, 200)
    assert float(inv_loss([c3, c3.clone(), c3.clone()], m3)) < 1e-8, "同じ視点で不変性の損失が 0 でない"
    assert float(inv_loss([c3, torch.randn_like(c3), c3], m3)) > 0.5, "違う視点で不変性の損失が小さい"
    eqb = torch.from_numpy(eq_basis())
    yv = torch.randn(2, T.D_Y + 1, 50)
    Ms1 = torch.from_numpy(warp_mats())[16:17]
    assert abs(float(np.exp(np.linspace(np.log(0.8), np.log(1.25), 33))[16]) - 1.0) < 1e-6
    g0 = torch.Generator().manual_seed(0)
    out = augment(yv, Ms1, torch.zeros(25), g0, eqb, 0.0)
    assert torch.allclose(out, yv, atol=1e-4), "α = 1・傾き 0・音色 0 の乱しが恒等でない"
    cg = torch.randn(1, CH, 50, requires_grad=True)
    GradRev.apply(cg, 0.5).sum().backward()
    assert torch.allclose(cg.grad, torch.full_like(cg, -0.5)), "勾配反転が −λ でない"
    ad = SpkAdv(3)
    xa = torch.randn(1, CH, 300)
    xb = xa.clone()
    xb[..., 200:] = torch.randn(1, CH, 100)
    assert torch.allclose(ad(xa)[..., :200], ad(xb)[..., :200], atol=1e-5), "話者の識別器が因果でない"
    h = torch.randn(2, 16, 400) * torch.linspace(0.8, 1.2, 16)[None, :, None] + 1.5
    mh = torch.ones(2, 400)
    st = white_stats(h, mh)
    hw = white_apply(h, st)
    cv = torch.einsum("bit,bjt->bij", hw, hw) / 400
    assert float(hw.mean(-1).abs().max()) < 1e-4 and float((cv.diagonal(dim1=-2, dim2=-1) - 1).abs().max()) < 0.2, "白くする段が平均 0・分散 1 の近くにならない"
    fx = (torch.tensor([3, 30]), torch.tensor([0.2, -0.3]))
    yv2 = torch.randn(2, T.D_Y + 1, 50)
    Msx = torch.from_numpy(warp_mats())
    a1, a2 = augment(yv2, Msx, torch.from_numpy(tilt_vec()), g0, fixed=fx), augment(yv2, Msx, torch.from_numpy(tilt_vec()), g0, fixed=fx)
    assert torch.allclose(a1, a2), "固定した乱しが同じ入力で同じにならない"
    print("unit test ok(z・c の因果性・不変性の損失の恒等・乱しの恒等・勾配反転・識別器の因果・白くする段・固定した乱し)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--subset", type=int, default=1)
    ap.add_argument("--inv", type=float, default=0.0)
    ap.add_argument("--amin", type=float, default=0.8)
    ap.add_argument("--amax", type=float, default=1.25)
    ap.add_argument("--eq", type=float, default=0.0)
    ap.add_argument("--spk", type=float, default=0.0)
    ap.add_argument("--prereg", default="results/snv_p2/prereg.yaml")
    ap.add_argument("--white", type=int, default=0)
    ap.add_argument("--pair", type=float, default=0.0)
    ap.add_argument("--init", default="")
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (ROOT / a.prereg).exists():
        print(f"{a.prereg} が無い: 起動しない", flush=True)
        return 1
    print("prereg", ROOT / a.prereg, "aug", a.aug, "inv", a.inv, "alpha", (a.amin, a.amax), "eq", a.eq, "spk", a.spk, flush=True)
    unit_test()
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    tr = split["train"][:40] if a.smoke else split["train"][:: a.subset]
    xs, sig, lab = load(tr)
    norm_of: dict = {}
    if a.white or a.pair > 0:
        k = 0
        for i, v in enumerate(tr):
            nn_ = len(sorted((T.FEAT / "e1" / v).glob("t*.npz")))
            nv_ = len(sorted((T.FEAT / "e1nv" / v).glob("*.npz")))
            norm_of[i] = list(range(k, k + nn_))
            k += nn_ + nv_
        assert k == len(xs), "発話の数え方が load と合わない"
    pidx: list = []
    if a.pair > 0:
        import train_snv_p13 as P13
        vi = {v: i for i, v in enumerate(tr)}
        pidx = [(norm_of[vi[x[0]]][x[1]], norm_of[vi[y[0]]][y[1]]) for x, y in P13.text_pairs(tr) if x[1] < len(norm_of[vi[x[0]]]) and y[1] < len(norm_of[vi[y[0]]])]
        print("pairs", len(pidx), flush=True)
    print(f"voices {len(tr)} utts {len(xs)} frames {sum(x.shape[1] for x in xs)}", flush=True)
    dev = "cuda"
    sig_t = torch.from_numpy(sig).to(dev)
    Ms, tv = torch.from_numpy(warp_mats(a.amin, a.amax)).to(dev), torch.from_numpy(tilt_vec()).to(dev)
    eqb = torch.from_numpy(eq_basis()).to(dev)
    m = CPC().to(dev)
    if a.init:
        st0 = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)
        m.load_state_dict(st0["cpc"])
        sig = np.asarray(st0["sig"], np.float32)
        sig_t = torch.from_numpy(sig).to(dev)
        print("init", a.init, "step", st0.get("step"), flush=True)
    wh = WHead(a.white).to(dev) if a.white else None
    adv = SpkAdv(len(tr)).to(dev)
    params = list(m.parameters()) + (list(adv.parameters()) if a.spk > 0 else []) + (list(wh.parameters()) if wh is not None else [])
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)
    steps = 200 if a.smoke else a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    rng, g = random.Random(0), torch.Generator(device=dev).manual_seed(0)
    np.save(out / "sig.npy", sig)
    t0 = time.time()
    for step in range(1, steps + 1):
        jb: list = []
        X, M, S = batch(xs, a.bs, rng, lab, jb)
        X, M, S = X.to(dev), M.to(dev), S.to(dev)
        li = torch.zeros((), device=dev)
        if wh is not None:
            picks = [rng.sample([k for k in norm_of[sp] if k != j], N_ENR) for sp, j in zip(S.tolist(), jb)]
            XE, ME = enr_batch(xs, picks, rng)
            XE, ME = XE.to(dev), ME.to(dev)
            B = X.shape[0]
            fx = (torch.randint(0, len(Ms), (B,), generator=g, device=dev), (torch.rand(B, generator=g, device=dev) * 2 - 1) * TILT)
            fxe = (fx[0].repeat_interleave(N_ENR), fx[1].repeat_interleave(N_ENR))
            with torch.no_grad():
                ze, ce = m.encode(x_torch(XE, sig_t))
                _, cae = m.encode(x_torch(augment(XE, Ms, tv, g, fixed=fxe), sig_t))
                pze, pce = wh.proj(ze, ce)
                _, pcae = wh.proj(ze, cae)
                cat = lambda h: h.reshape(B, N_ENR, h.shape[1], ENR).permute(0, 2, 1, 3).reshape(B, h.shape[1], N_ENR * ENR)
                mv = (ME * XE[:, T.D_Y]).reshape(B, N_ENR * ENR)
                sz, sc_, sca = white_stats(cat(pze), mv), white_stats(cat(pce), mv), white_stats(cat(pcae), mv)
            z, c = m.encode(x_torch(X, sig_t))
            _, ca = m.encode(x_torch(augment(X, Ms, tv, g, fixed=fx), sig_t))
            pz, pc = wh.proj(z, c)
            _, pca_ = wh.proj(z, ca)
            zw = white_apply(pz, sz)
            l0, _ = info_nce(zw, white_apply(pc, sc_), M, wh.W, g)
            la, accs = info_nce(zw, white_apply(pca_, sca), M, wh.W, g)
            loss = 0.5 * (l0 + la)
        else:
            z, c = m.encode(x_torch(X, sig_t))
            loss, accs = info_nce(z, c, M, m.W, g)
        if wh is not None:
            pass
        elif a.inv > 0:
            _, c1 = m.encode(x_torch(augment(X, Ms, tv, g, eqb, a.eq), sig_t))
            _, c2 = m.encode(x_torch(augment(X, Ms, tv, g, eqb, a.eq), sig_t))
            l1, accs = info_nce(z, c1, M, m.W, g)
            l2, _ = info_nce(z, c2, M, m.W, g)
            li = inv_loss([c, c1, c2], M)
            loss = (loss + l1 + l2) / 3 + a.inv * li
        elif a.aug:
            _, ca = m.encode(x_torch(augment(X, Ms, tv, g), sig_t))
            la, accs = info_nce(z, ca, M, m.W, g)
            loss = 0.5 * (loss + la)
        lp = torch.zeros((), device=dev)
        if a.pair > 0:
            sel = [j for pr in rng.sample(pidx, N_PAIR) for j in pr]
            Lp = max(xs[j].shape[1] for j in sel)
            XP = torch.zeros(len(sel), T.D_Y + 1, Lp)
            for i, j in enumerate(sel):
                XP[i, :, :xs[j].shape[1]] = torch.from_numpy(xs[j])
            XP = XP.to(dev)
            _, cp = m.encode(x_torch(XP, sig_t))
            vm = XP[:, T.D_Y] > 0
            vm[:, :WARM_INV] = False
            out_ = []
            for i in range(0, len(sel), 2):
                bags = []
                for j in (i, i + 1):
                    f = cp[j].T[vm[j]]
                    bags.append(f[torch.randperm(len(f), generator=g, device=dev)[:MMD_FR]])
                out_.append(P13.mmd2(bags[0], bags[1]))
            lp = torch.stack(out_).mean()
            loss = loss + a.pair * lp
        sacc = 0.0
        if a.spk > 0:
            ls, sacc = spk_loss(adv, c, M, S, a.spk * min(1.0, step / 5000))
            loss = loss + ls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if a.spk > 0:
            torch.nn.utils.clip_grad_norm_(adv.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or a.smoke and step % 50 == 0:
            print(f"step {step} loss {float(loss):.3f} inv {float(li):.3f} pair {float(lp):.4f} spk_acc {sacc:.3f} acc " + " ".join(f"k{k}:{x:.2f}" for k, x in zip(KS, accs)) + f" {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 5000 == 0 or step == steps:
            extra = ({"adv": adv.state_dict()} if a.spk > 0 else {}) | ({"white": wh.state_dict(), "dw": a.white} if wh is not None else {})
            torch.save({"cpc": m.state_dict(), "sig": sig, "step": step} | extra, out / "last.pt")
            if step == steps // 2:
                torch.save({"cpc": m.state_dict(), "sig": sig, "step": step} | extra, out / f"step{step // 1000}k.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
