"""P3(synth_nam_vc §8): 表の型 x の C の演奏を、凍結した自前の因果の CPC の文脈 c(話者の登録の平均を引く)だけにして、自己の reamp で C を学ぶ。
P1 の E(y) は源の声を運び C に写しの道があった(n = 写し + 平均のずらし)。c は凍結 = 勾配で声を運ぶように変わらない = C は声を参照から取るしかない(c に残る声の分だけ写しが残る)。
演奏 a = W(c_t − μ_c)(W は学習する 256 → 64 の線形・μ_c = その話者の登録の有声のフレームの c の平均 = 登録の定数)。C・損失・データ・物差しは P1 の n と同じ(self + rend)。
注意(鍵 = 参照の a・値 = 参照の Y)が one-hot に潰れて参照のフレームの写しになる解は検査 C のずる = 注意のエントロピーと出力の参照のフレームとの一致を記録する。
事前登録 = results/snv_p3/prereg.yaml。
    cd training && uv run --no-sync python train_snv_p3.py --tag snv_p3_x --cpc results/snv_p2b_cpc/last.pt
--cyc w(P4)= 自前の reamp を演奏の入力に: 束の別の声 B の構造で前向きに変換した ŷ_AB = C(a_A, 高さ A, 有声 A, 構造 B)(勾配を止める)を演奏にし、
元の構造 A で本物の y_A を再現する L1 を w 倍で足す(ŷ_AB の演奏の平均 μ_c は切り出しの ŷ_AB の有声のフレームで)。教師は本物のまま・加工は入力側だけ。
--init = 学習済みの P3 から続ける(proj と C)。対照 = --init だけ(同じ step・同じ学習率の予定)。
    cd training && uv run --no-sync python train_snv_p3.py --tag snv_p4_cyc --cpc results/snv_p2b_cpc/last.pt --init results/snv_p3_x/last.pt --steps 20000 --cyc 1.0 --prereg results/snv_p4/prereg.yaml
--circ q(P7)= 番地だけの回路 T.CircQ: 演奏は参照を読む注意の問い(番地)にだけ使い、描く側は読んだ参照の値と高さ・有声だけを見る(演奏 → 出力の直接の道が無い)。
注意は余弦 × 上限 6 の尺度(一つのフレームへ潰れない)。評価ごとに注意の最大の重みと実効のフレーム数(1 / Σw²)を記録し、最大 > 0.5 か実効 < 8 で止める(検索 = 不可)。
    cd training && uv run --no-sync python train_snv_p3.py --tag snv_p7_q --cpc results/snv_p2b_cpc/last.pt --circ q --prereg results/snv_p7/prereg.yaml
--play wst(P8)= 演奏を話者の登録の分布で白くする PlayW: c − μ を固定の PCA 32 次元(results/snv_p8b/pca_c_P2b.npy)へ射影し 0・40・80・120 ms に積んだ 128 次元を、
その話者の登録(学習 = 参照・推論 = 操作者の登録)の有声のフレームの 2 次のモーメント(収縮 0.1)の逆平方根で白くしてから線形 128 → 64(snv_diag/p8a_draw_voice.py の wst と同じ)。
    cd training && uv run --no-sync python train_snv_p3.py --tag snv_p8_wst --cpc results/snv_p2b_cpc/last.pt --play wst --pca results/snv_p8b/pca_c_P2b.npy --prereg results/snv_p8/prereg.yaml
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

import train_snv_p1 as T
import train_snv_p2_cpc as CP

ROOT = Path(__file__).resolve().parent.parent


class Play(nn.Module):
    def __init__(self, ck: Path) -> None:
        super().__init__()
        st = torch.load(ck, map_location="cpu", weights_only=False)
        self.cpc = CP.CPC()
        self.cpc.load_state_dict(st["cpc"])
        self.cpc.requires_grad_(False).eval()
        self.register_buffer("sig", torch.from_numpy(np.asarray(st["sig"], np.float32)))
        self.proj = nn.Linear(CP.CH, T.D_A)

    def train(self, mode: bool = True) -> "Play":
        super().train(mode)
        self.cpc.eval()
        return self

    def ctx(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.cpc.encode(CP.x_torch(torch.cat([y, v[:, None]], 1), self.sig))[1]

    def mean_of(self, y: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        c = self.ctx(y, v)
        w = (v * m.float())[:, None]
        return (c * w).sum(-1) / w.sum(-1).clamp(min=1.0)

    def forward(self, y: torch.Tensor, v: torch.Tensor, mu_c: torch.Tensor) -> torch.Tensor:
        return self.proj((self.ctx(y, v) - mu_c[..., None]).transpose(1, 2)).transpose(1, 2)


class PlayW(Play):
    def __init__(self, ck: Path, pca: Path, k: int = 32, lags: tuple = (0, 8, 16, 24), shrink: float = 0.1, cb: Path | None = None, xc0: bool = False) -> None:
        super().__init__(ck)
        B = np.load(pca)
        k = B.shape[1] if B.shape[1] < B.shape[0] else k
        self.register_buffer("P", torch.from_numpy(np.ascontiguousarray(B[:, :k]).astype(np.float32)))
        self.lags, self.shrink, self.xc0 = lags, shrink, xc0
        self.proj = nn.Linear(k * len(lags) + int(xc0), T.D_A)
        self.cb_s2 = None
        if cb is not None:
            import snv_cmn as CM
            c, self.cb_s2 = CM.load(cb)
            assert c.shape[1] == k * len(lags), (c.shape, k, lags)
            self.register_buffer("cb", c)

    def stack(self, c: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        s = torch.einsum("bct,ck->bkt", c - mu[..., None], self.P)
        return torch.cat([torch.cat([s[..., :1].expand(-1, -1, l), s[..., :s.shape[-1] - l]], -1) if l else s for l in self.lags], 1)

    def mean_of(self, y: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.ctx(y, v)
        w = (v * m.float())[:, None]
        mu = (c * w).sum(-1) / w.sum(-1).clamp(min=1.0)
        s = self.stack(c, mu).double()
        cov = torch.einsum("bit,bjt->bij", s * w.double(), s) / w.double().sum(-1).clamp(min=1.0)[..., None]
        D = cov.shape[-1]
        tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1) / D
        cov = (1 - self.shrink) * cov + self.shrink * tr[:, None, None] * torch.eye(D, dtype=cov.dtype, device=cov.device)
        e, q = torch.linalg.eigh(cov)
        Wm = ((q / e.clamp(min=1e-8).sqrt()[:, None, :]) @ q.transpose(-1, -2)).float()
        st: tuple = (mu, Wm)
        if self.cb_s2 is not None:
            import snv_cmn as CM
            xe = torch.einsum("bij,bjt->bit", Wm, s.float()).transpose(1, 2)
            st = (*st, CM.offsets(xe, w[:, 0], self.cb, self.cb_s2))
        if self.xc0:
            n = w[:, 0].sum(-1).clamp(min=1.0)
            m0 = (y[:, 0] * w[:, 0]).sum(-1) / n
            s0 = (((y[:, 0] - m0[:, None]) ** 2 * w[:, 0]).sum(-1) / n).sqrt().clamp(min=1e-3)
            st = (*st, torch.stack([m0, s0], 1))
        return st

    def white(self, y: torch.Tensor, v: torch.Tensor, st: tuple) -> torch.Tensor:
        x = torch.einsum("bij,bjt->bit", st[1], self.stack(self.ctx(y, v), st[0]))
        if self.cb_s2 is None:
            return x
        import snv_cmn as CM
        return CM.apply(x.transpose(1, 2), self.cb, self.cb_s2, st[2]).transpose(1, 2)

    def feat(self, y: torch.Tensor, v: torch.Tensor, st: tuple) -> torch.Tensor:
        x = self.white(y, v, st)
        return torch.cat([x, (y[:, :1] - st[-1][:, :1, None]) / st[-1][:, 1:, None]], 1) if self.xc0 else x

    def forward(self, y: torch.Tensor, v: torch.Tensor, st: tuple) -> torch.Tensor:
        return self.proj(self.feat(y, v, st).transpose(1, 2)).transpose(1, 2)


def make_play(cpc: Path, play: str = "c", pca: Path | None = None, lags: list | tuple | None = None, shrink: float | None = None, cb: Path | None = None, xc0: bool = False) -> Play:
    if play != "wst":
        return Play(cpc)
    return PlayW(cpc, pca, lags=tuple(lags) if lags is not None else (0, 8, 16, 24), shrink=float(shrink) if shrink is not None else 0.1, cb=cb, xc0=xc0)


def xc0_tests(ck: Path, pca: Path, lags: tuple, shrink: float, cb: Path | None) -> None:
    Pl, P0 = make_play(ck, "wst", pca, lags, shrink, cb, True), make_play(ck, "wst", pca, lags, shrink, cb)
    us = [T.load_utt(T.FEAT / f"e1/ef000/t0{j}.npz") for j in (0, 1, 2, 9)] if (T.FEAT / "e1/ef000").exists() else [T.load_utt(p) for p in sorted((T.FEAT / "e1").glob("*/t0*.npz"))[:4]]
    with torch.no_grad():
        ry = torch.from_numpy(np.concatenate([u["y"] for u in us[:3]], 1))[None]
        rv = torch.from_numpy(np.concatenate([u["v"] for u in us[:3]]))[None]
        rm = torch.ones(rv.shape, dtype=torch.bool)
        y, v = torch.from_numpy(us[3]["y"])[None], torch.from_numpy(us[3]["v"])[None]
        st, s0 = Pl.mean_of(ry, rv, rm), P0.mean_of(ry, rv, rm)
        f, w = Pl.feat(y, v, st), P0.white(y, v, s0)
        c0e = ry[0, 0][rv[0] > 0.5].numpy()
        ref = (us[3]["y"][0] - c0e.mean()) / c0e.std()
        assert f.shape[1] == w.shape[1] + 1 and torch.allclose(f[:, :-1], w, atol=1e-5), "明示の強さを足すと白くした符号が変わる"
        assert np.allclose(f[0, -1].numpy(), ref, atol=1e-3), "明示の強さが登録の有声の c0 の標準化と一致しない"
        ry2, y2 = ry.clone(), y.clone()
        ry2[:, 0] += 0.7
        y2[:, 0] += 0.7
        assert torch.allclose(Pl.feat(y2, v, Pl.mean_of(ry2, rv, rm))[:, -1], f[:, -1], atol=1e-4), "明示の強さが音量の一律の差で変わる"
        y3 = y.clone()
        y3[..., 200:] = torch.randn_like(y3[..., 200:])
        assert torch.allclose(Pl.feat(y3, v, st)[..., :200], f[..., :200], atol=1e-5), "明示の強さを足した演奏が因果でない"
        assert Pl.proj.in_features == P0.proj.in_features + 1
    print("xc0 tests ok(白くした符号は同じ・登録の有声の c0 の標準化と一致・音量の一律の差に不変・因果性)", flush=True)


def cmn_tests(ck: Path, pca: Path, lags: tuple, shrink: float, cb: Path) -> None:
    import snv_cmn as CM
    CM.test()
    Pl, P0 = make_play(ck, "wst", pca, lags, shrink, cb), make_play(ck, "wst", pca, lags, shrink)
    us = [T.load_utt(T.FEAT / f"e1/ef000/t0{j}.npz") for j in (0, 1, 2, 9)] if (T.FEAT / "e1/ef000").exists() else [T.load_utt(p) for p in sorted((T.FEAT / "e1").glob("*/t0*.npz"))[:4]]
    with torch.no_grad():
        ry = torch.from_numpy(np.concatenate([u["y"] for u in us[:3]], 1))[None]
        rv = torch.from_numpy(np.concatenate([u["v"] for u in us[:3]]))[None]
        m = torch.ones(rv.shape, dtype=torch.bool)
        st, s0 = Pl.mean_of(ry, rv, m), P0.mean_of(ry, rv, m)
        assert len(st) == 3 and torch.allclose(st[1], s0[1], atol=1e-5)
        xr = Pl.white(ry, rv, st).transpose(1, 2)
        d0, d1 = float(st[2].norm()), float(CM.offsets(xr, rv, Pl.cb, Pl.cb_s2).norm())
        assert d1 < 0.72 * d0, (d0, d1)
        y, v = torch.from_numpy(us[3]["y"])[None], torch.from_numpy(us[3]["v"])[None]
        a = Pl(y, v, st)
        y2 = y.clone()
        y2[..., 300:] = torch.randn_like(y2[..., 300:])
        assert torch.allclose(a[..., :300], Pl(y2, v, st)[..., :300], atol=1e-5), "内容ごとの平均の正規化が因果でない"
    print(f"cmn tests ok(登録の級ごとのずれ {d0:.3f} → 正規化の後 {d1:.3f}・白くする行列は同じ・因果性)", flush=True)


def structure_of(Pl: Play, C: T.Circ, ry: torch.Tensor, rv: torch.Tensor, rmask: torch.Tensor, mu_c: torch.Tensor) -> dict:
    return C.structure(ry, Pl(ry, rv, mu_c), rmask)


def p_stats(p: torch.Tensor) -> tuple[float, float, float]:
    ent = float(-(p * p.clamp(min=1e-12).log()).sum(-1).mean())
    return ent, float(p.max(-1).values.mean()), float((1.0 / (p * p).sum(-1)).mean())


def attn_stats(C: nn.Module, a: torch.Tensor, lf: torch.Tensor, v: torch.Tensor, s: dict) -> tuple[float, float, float]:
    if isinstance(C, T.CircQ):
        return p_stats(C.weights(a, lf, v, s))
    h = C.inp(torch.cat([a, lf[:, None], v[:, None]], 1))
    for i, blk in enumerate(C.qblocks if isinstance(C, T.CircB) else C.blocks):
        h = blk(h, s["z"])
        if i == 3:
            B, Ch, L = h.shape
            q = C.q(h.transpose(1, 2)).view(B, L, C.heads, C.dk).transpose(1, 2)
            logit = (q @ s["k"].transpose(-1, -2)) / math.sqrt(C.dk)
            return p_stats(logit.masked_fill(~s["rmask"][:, None, None, :], -1e9).softmax(-1))
    return 0.0, 0.0, 0.0


def unit_tests(ck: Path) -> None:
    torch.manual_seed(0)
    Pl = Play(ck)
    y, v = torch.randn(1, T.D_Y, 300), torch.ones(1, 300)
    mu = torch.zeros(1, CP.CH)
    a1 = Pl(y, v, mu)
    y2 = y.clone()
    y2[..., 200:] = torch.randn(1, T.D_Y, 100)
    a2 = Pl(y2, v, mu)
    assert torch.allclose(a1[..., :200], a2[..., :200], atol=1e-5) and not torch.allclose(a1[..., 200:], a2[..., 200:]), "演奏が因果でない"
    assert not any(p.requires_grad for p in Pl.cpc.parameters()), "CPC が凍結されていない"
    print("unit tests ok(演奏の因果性・CPC の凍結)", flush=True)


def playw_tests(ck: Path, pca: Path, lags: list | tuple | None = None, shrink: float | None = None) -> None:
    from snv_diag import p8a_draw_voice as DV
    Pl = make_play(ck, "wst", pca, lags, shrink)
    us = [T.load_utt(T.FEAT / f"e1/ef000/t0{j}.npz") for j in (0, 1, 2, 9)] if (T.FEAT / "e1/ef000").exists() else [T.load_utt(p) for p in sorted((T.FEAT / "e1").glob("*/t0*.npz"))[:4]]
    tn = lambda u: (torch.from_numpy(u["y"])[None], torch.from_numpy(u["v"])[None])
    with torch.no_grad():
        cs = [Pl.ctx(*tn(u))[0].numpy().T for u in us]
        vm = [u["v"] > 0 for u in us]
        mu = np.concatenate([c[m] for c, m in zip(cs[:3], vm[:3])]).mean(0)
        P = Pl.P.numpy()
        stk = lambda c: DV.lagged((c - mu) @ P, Pl.lags)
        W = DV.whiten(np.concatenate([stk(c)[m] for c, m in zip(cs[:3], vm[:3])]), Pl.shrink)
        ref = stk(cs[3]) @ W
        st = (torch.from_numpy(mu)[None].float(), torch.from_numpy(W)[None].float())
        got = Pl.white(*tn(us[3]), st)[0].numpy().T
        ry = torch.from_numpy(np.concatenate([u["y"] for u in us[:3]], 1))[None]
        rv = torch.from_numpy(np.concatenate([u["v"] for u in us[:3]]))[None]
        mu2, W2 = Pl.mean_of(ry, rv, torch.ones(rv.shape, dtype=torch.bool))
    assert np.allclose(got, ref, atol=1e-3 * np.abs(ref).max()), "PlayW の白くする手順が探り(p8a の wst)と一致しない"
    rmu = float(np.abs(mu2[0].numpy() - mu).max() / np.abs(mu).max())
    rel = float(np.abs(W2[0].numpy() - W).max() / np.abs(W).max())
    print(f"PlayW 登録の統計の探りとのずれ(参照を継いで CPC に通す分だけ違う): 平均 {rmu:.4f}・行列 {rel:.4f}", flush=True)
    assert rmu < 0.05 and rel < 0.1, "PlayW の登録の統計が探り(発話ごと)と大きくずれる"
    y, v = torch.randn(1, T.D_Y, 300), torch.ones(1, 300)
    y2 = y.clone()
    y2[..., 200:] = torch.randn(1, T.D_Y, 100)
    with torch.no_grad():
        a1, a2 = Pl(y, v, st), Pl(y2, v, st)
    assert torch.allclose(a1[..., :200], a2[..., :200], atol=1e-4) and not torch.allclose(a1[..., 200:], a2[..., 200:]), "PlayW が因果でない"
    print("PlayW tests ok(同じ統計で探りの wst と一致・登録の統計のずれが小さい・因果性)", flush=True)


def circq_tests() -> None:
    torch.manual_seed(0)
    C = T.CircQ()
    a, lf, v = torch.randn(1, T.D_A, 300), torch.randn(1, 300), torch.ones(1, 300)
    s = C.structure(torch.randn(1, T.D_Y, 80), torch.randn(1, T.D_A, 80), torch.ones(1, 80, dtype=torch.bool))
    a2, lf2 = a.clone(), lf.clone()
    a2[..., 200:], lf2[:, 200:] = torch.randn(1, T.D_A, 100), 0.3
    y1, y2 = C(a, lf, v, s), C(a2, lf2, v, s)
    assert torch.allclose(y1[..., :200], y2[..., :200], atol=1e-5) and not torch.allclose(y1[..., 200:], y2[..., 200:]), "CircQ が因果でない"
    s1 = C.structure(torch.randn(1, T.D_Y, 1), torch.randn(1, T.D_A, 1), torch.ones(1, 1, dtype=torch.bool))
    assert torch.allclose(C(a, lf, v, s1), C(torch.randn_like(a), lf, v, s1), atol=1e-5), "演奏が注意を通らずに出力へ届く"
    p = C.weights(a, lf, v, s)
    bound = math.exp(C.smax) / (math.exp(C.smax) + 79 * math.exp(-C.smax))
    assert float(p.max()) <= bound + 1e-6, "注意の尺度が上限を超える"
    print("CircQ tests ok(因果性・演奏 → 出力の直接の道なし(参照 1 フレームで演奏に不変)・尺度の上限)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cpc", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prereg", default="results/snv_p3/prereg.yaml")
    ap.add_argument("--cyc", type=float, default=0.0)
    ap.add_argument("--init", default="")
    ap.add_argument("--circ", choices=["x", "q"], default="x")
    ap.add_argument("--play", choices=["c", "wst"], default="c")
    ap.add_argument("--pca", default="")
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (ROOT / a.prereg).exists():
        print(f"{a.prereg} が無い: 起動しない", flush=True)
        return 1
    print("prereg", ROOT / a.prereg, "cpc", a.cpc, flush=True)
    torch.manual_seed(a.seed)
    unit_tests(ROOT / a.cpc)
    if a.circ == "q":
        circq_tests()
    if a.play == "wst":
        playw_tests(ROOT / a.cpc, ROOT / a.pca)
    t0 = time.time()
    corpus = T.load_corpus()
    tr, va = T.split(sorted(corpus))
    print(f"voices {len(corpus)} train {len(tr)} val {len(va)} load {time.time() - t0:.0f}s", flush=True)
    (out / "split.json").write_text(json.dumps({"train": tr, "val": va}))
    dev = "cuda"
    Pl, C = make_play(ROOT / a.cpc, a.play, ROOT / a.pca if a.pca else None).to(dev), (T.CircQ() if a.circ == "q" else T.Circ()).to(dev)
    if a.init:
        st0 = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)
        assert st0["cpc"] == a.cpc, "演奏の CPC が初期値と違う"
        Pl.proj.load_state_dict(st0["proj"]), C.load_state_dict(st0["C"])
        print("init", a.init, "step", st0["step"], flush=True)
    params = list(Pl.proj.parameters()) + list(C.parameters())
    print(f"params proj {sum(p.numel() for p in Pl.proj.parameters()) / 1e6:.3f}M C {sum(p.numel() for p in C.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = 200 if a.smoke else a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    trb, vab = T.Batches(corpus, tr, a.seed), T.Batches(corpus, va, 12345, p_nv=0.0)
    vbatches = [vab.sample(a.bs) for _ in range(8)]
    to_dev = lambda b: {k: x.to(dev, non_blocking=True) for k, x in b.items()}

    def pieces(b: dict) -> tuple:
        mu_c = Pl.mean_of(b["ry"], b["rv"], b["rmask"])
        s = structure_of(Pl, C, b["ry"], b["rv"], b["rmask"], mu_c)
        return mu_c, s

    @torch.no_grad()
    def reamp(b: dict, aA: torch.Tensor, sA: dict) -> torch.Tensor:
        perm = torch.roll(torch.arange(aA.shape[0], device=aA.device), 1)
        yAB = C(aA, b["lf"], b["v"], {k: x[perm] for k, x in sA.items()})
        return Pl(yAB, b["v"], Pl.mean_of(yAB, b["v"], b["ok"])), yAB

    @torch.no_grad()
    def evaluate() -> tuple[float, float, float, float, float, float, float]:
        Pl.eval(), C.eval()
        ls, lt, en, mx, fd, lc, ef = [], [], [], [], [], [], []
        for vb in vbatches:
            b = to_dev(vb)
            m = b["ok"].clone()
            m[:, :T.WARM_SELF] = 0
            mu_c, s = pieces(b)
            aA = Pl(b["y"], b["v"], mu_c)
            ls.append(float(T.masked_l1(C(aA, b["lf"], b["v"], s), b["y"], m)))
            rmean = (b["ry"] * b["rmask"][:, None]).sum(-1, keepdim=True) / b["rmask"].sum(-1)[:, None, None].clamp(min=1)
            lt.append(float(T.masked_l1(rmean.expand_as(b["y"]), b["y"], m)))
            e_, x_, f_ = attn_stats(C, aA, b["lf"], b["v"], s)
            en.append(e_), mx.append(x_), ef.append(f_)
            aAB, yAB = reamp(b, aA, s)
            fd.append(float(T.masked_l1(yAB, b["y"], m)))
            lc.append(float(T.masked_l1(C(aAB, b["lf"], b["v"], s), b["y"], m)))
        Pl.train(), C.train()
        return float(np.mean(ls)), float(np.mean(lt)), float(np.mean(en)), float(np.mean(mx)), float(np.mean(fd)), float(np.mean(lc)), float(np.mean(ef))

    v0, triv, _, _, _, _, _ = evaluate()
    print(f"floors: untrained val L1 {v0:.4f} / trivial (参照の平均の Y) {triv:.4f}", flush=True)
    log = {"floors": {"untrained": v0, "trivial": triv}, "evals": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        b = to_dev(trb.sample(a.bs))
        ms = b["ok"].clone()
        ms[:, :T.WARM_SELF] = 0
        mc = b["ok"].clone()
        mc[:, :T.WARM_CYC] = 0
        mu_c, sA = pieces(b)
        aA = Pl(b["y"], b["v"], mu_c)
        yA = C(aA, b["lf"], b["v"], sA)
        l_self = T.masked_l1(yA, b["y"], ms)
        l_rend = T.masked_l1(C(Pl(yA.detach(), b["v"], mu_c), b["lf"], b["v"], sA), b["y"], mc)
        loss = l_self + l_rend
        l_cyc = torch.zeros((), device=dev)
        if a.cyc > 0:
            aAB, _ = reamp(b, aA, sA)
            l_cyc = T.masked_l1(C(aAB, b["lf"], b["v"], sA), b["y"], mc)
            loss = loss + a.cyc * l_cyc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step} self {float(l_self):.4f} rend {float(l_rend):.4f} cyc {float(l_cyc):.4f} gn {float(gn):.2f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 2000 == 0 or step == steps:
            vl, _, ent, amax, fdist, vcyc, eff = evaluate()
            rec = {"step": step, "val_l1": vl, "train_self": float(l_self), "train_rend": float(l_rend), "train_cyc": float(l_cyc), "attn_entropy": ent, "attn_max": amax, "attn_eff_frames": eff, "val_fwd_dist": fdist, "val_cyc_l1": vcyc, "sec": time.time() - t0}
            log["evals"].append(rec)
            print("eval", json.dumps(rec), flush=True)
            ck = {"proj": Pl.proj.state_dict(), "C": C.state_dict(), "cpc": a.cpc, "step": step, "circ": a.circ, "play": a.play, "pca": a.pca}
            torch.save(ck, out / "last.pt")
            if step == steps // 2:
                torch.save(ck, out / f"step{step // 1000}k.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
            if a.circ == "q" and (amax > 0.5 or eff < 8):
                print(f"STOP: 注意が検索に寄る(最大 {amax:.3f}・実効 {eff:.1f} フレーム)", flush=True)
                return 3
            if step == 10000 and vl > 0.8 * triv and not a.init:
                print(f"TRIPWIRE: val L1 {vl:.4f} > 0.8 × trivial {triv:.4f} at 10k → stop", flush=True)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
