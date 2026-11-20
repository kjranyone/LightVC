"""段 P20 補足 2〜(paper_snv rev50 §6): 内容ごとの平均の正規化(閉じた形・登録の定数・因果)。
族の学習の声の白くした符号(PlayW の white の出力)で K 個の級の符号帳(k 平均)を作る。操作者の登録のフレームの級ごとの平均(柔らかい割り当て・事前の重み TAU フレームで族の中心へ寄せる)と
族の級の中心の差 delta を、フレームごとに級の事後確率で重みづけて引く: x' = x − γ(x) · delta・γ_k(x) ∝ exp(−‖x − c_k‖² / (2 s2))。
符号帳は音素でも事前に決めた写像でもなく、合成の族の符号の上の級(学習データ = Irodori の生成だけ)。操作者の側で使うのは登録の統計だけ(白くする式と同じ類)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_cmn.py fit [K=128] [lags=0] [shrink=0.6] [cpc=results/snv_p2b_cpc/last.pt] [pca=results/snv_p8b/pca_c_P2b.npy] [out=results/snv_p21/codebook_k128.npz] [vsel=f]   # E1 の学習の声から符号帳を作る(vsel=f = 段 P70 step b の診断: 男声を除く)
    cd training && uv run --no-sync python snv_cmn.py test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
TAU = 20.0


def posterior(x: torch.Tensor, c: torch.Tensor, s2: float) -> torch.Tensor:
    d = (x * x).sum(-1, keepdim=True) - 2 * x @ c.T + (c * c).sum(-1)
    return torch.softmax(-d / (2 * s2), -1)


def offsets(xe: torch.Tensor, we: torch.Tensor, c: torch.Tensor, s2: float, tau: float = TAU) -> torch.Tensor:
    g = posterior(xe, c, s2) * we[..., None]
    return (g.transpose(-1, -2) @ xe + tau * c) / (g.sum(-2)[..., None] + tau) - c


def apply(x: torch.Tensor, c: torch.Tensor, s2: float, delta: torch.Tensor) -> torch.Tensor:
    return x - posterior(x, c, s2) @ delta


def fit_codebook(X: torch.Tensor, K: int, iters: int = 40, seed: int = 0) -> tuple[torch.Tensor, float]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = X[torch.randperm(len(X), generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for k in range(K):
            m = a == k
            if m.any():
                c[k] = X[m].mean(0)
    return c, float(((X - c[torch.cdist(X, c).argmin(1)]) ** 2).sum(1).mean() / X.shape[1])


def load(path: Path) -> tuple[torch.Tensor, float]:
    z = np.load(path)
    return torch.from_numpy(z["c"].astype(np.float32)), float(z["s2"])


def test() -> None:
    g = torch.Generator().manual_seed(0)
    c = torch.randn(8, 4, generator=g)
    xe = c[torch.randint(0, 8, (400,), generator=g)] + 0.2 * torch.randn(400, 4, generator=g)
    shift = torch.tensor([0.5, -0.3, 0.0, 0.2])
    d0 = offsets(xe, torch.ones(400), c, 0.04, 0.0)
    assert d0.abs().max() < 0.1, d0.abs().max()
    d1 = offsets(xe + shift, torch.ones(400), c, 0.04 * 6, 1.0)
    y = apply(xe + shift, c, 0.04 * 6, d1)
    assert (y - xe).abs().mean() < 0.5 * shift.abs().mean(), (float((y - xe).abs().mean()), float(shift.abs().mean()))
    b = offsets(torch.stack([xe, xe + shift]), torch.ones(2, 400), c, 0.04 * 6, 1.0)
    assert torch.allclose(b[1], d1, atol=1e-5)
    p = posterior(xe[:5], c, 0.04)
    assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-5)
    print("snv_cmn test ok(ずれの無い登録は差 0・一律のずれは半分以上戻る・束の形・事後確率の和 1)")


def fit_main(opt: dict) -> int:
    import train_snv_p1 as T
    import train_snv_p3 as P3
    K, shrink = int(opt.get("K", 128)), float(opt.get("shrink", 0.6))
    lags = tuple(int(x) for x in opt.get("lags", "0").split(","))
    out = ROOT / opt.get("out", f"results/snv_p21/codebook_k{K}.npz")
    split = json.loads((ROOT / "results/snv_p8_wst/split.json").read_text())
    if opt.get("vsel") == "f":
        split["train"] = [v for v in split["train"] if not v.startswith("em")]
    cpc, pca = opt.get("cpc", "results/snv_p2b_cpc/last.pt"), opt.get("pca", "results/snv_p8b/pca_c_P2b.npy")
    Pl = P3.PlayW(ROOT / cpc, ROOT / pca, lags=lags, shrink=shrink).to("cuda").eval()
    xs = []
    with torch.no_grad():
        for v in split["train"]:
            us = [T.load_utt(p) for p in sorted((T.FEAT / "e1" / v).glob("t0*.npz"))]
            ry = torch.from_numpy(np.concatenate([u["y"] for u in us[:3]], 1))[None].cuda()
            rv = torch.from_numpy(np.concatenate([u["v"] for u in us[:3]]))[None].cuda()
            st = Pl.mean_of(ry, rv, torch.ones(rv.shape, dtype=torch.bool, device="cuda"))
            for u in us:
                y, vv = torch.from_numpy(u["y"])[None].cuda(), torch.from_numpy(u["v"])[None].cuda()
                xs.append(Pl.white(y, vv, st)[0].T[vv[0] > 0.5][::16])
    X = torch.cat(xs)
    c, s2 = fit_codebook(X, K)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, c=c.cpu().numpy().astype(np.float32), s2=np.float32(s2), K=K, lags=np.asarray(lags), shrink=np.float32(shrink), n_frames=len(X), n_voices=len(split["train"]), cpc=cpc, pca=pca)
    print("符号帳", out, "符号", cpc, "K", K, "フレーム", len(X), "声", len(split["train"]), "級の中の分散(次元あたり)", round(s2, 4), flush=True)
    return 0


if __name__ == "__main__":
    o = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    if "test" in sys.argv[1:]:
        test()
    elif "fit" in sys.argv[1:]:
        raise SystemExit(fit_main(o))
