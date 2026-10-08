"""D5: 目標の表の推定を統計的に良くできるか(学習なし・評価話者を使わない)。
開発話者 --n の 25s の音声を、前半(≈10s)で表を作り(薄い表)・後半を「真値」の表にして、真値の観測された単位の重みつき二乗誤差を比べる。
方法: raw(現行 = 観測単位の平均+未観測は最近傍の単位から借用)/ mean(未観測と薄い単位を全話者の単位平均 μ_u へ縮約)/ factor(縮約先を μ_u + P z_s:話者の低次元因子 z_s を観測単位から重みつき最小二乗)。
縮約の強さ λ(フレーム数)を掃引。
    uv run python d5_table_shrink.py --n 600 --out ../results/conv_p0/d5.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import json as J
    import random
    import torch
    import c1_content as CC
    import conv_c0 as C0
    import nvoc as N
    import prep_c3_targets as P
    import train_c1 as T1
    import train_rvoc as TR
    P.TGT_S, P.MAX_S, P.AUG = 25.0, 30.0, False
    ex, exs = TR.hi_eval_paths(), TR.held_speakers()
    rows = [r for r in J.loads(P.MANIFEST.read_text())["rows"] if r.get("keep") and not TR.eval_excluded(r, ex, exs) and r["src"].startswith("real")]
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["spk"], []).append(r)
    rng = random.Random(1)
    jobs = []
    for spk, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["wav"])
        rng.shuffle(rs)
        pick, tot = [], 0.0
        for r in rs:
            if tot >= 25.0:
                break
            if tot + r["dur"] > 30.0 and pick:
                continue
            pick.append(r)
            tot += r["dur"]
        if tot >= 22.0:
            jobs.append((("real", spk), pick))
    jobs = jobs[:: max(1, len(jobs) // a.n)][: a.n]
    print("dev speakers", len(jobs), flush=True)
    dev = "cuda"
    c1d = ROOT / "results/c1_1"
    st = torch.load(c1d / "last.pt", map_location="cpu", weights_only=False)
    net = CC.C1(st["cfg"]["k"], st["cfg"]["ch"], tuple(st["cfg"]["dils"])).to(dev).eval()
    net.load_state_dict(st["net"])
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(c1d / "codebook.pt").to(dev)
    K = Cb.shape[0]
    Cn = Cb.cpu().numpy()
    D1, D2 = [], []
    with Pool(a.procs) as pool, torch.no_grad():
        for i, r in enumerate(pool.imap(P.work, jobs, chunksize=1)):
            if "err" in r:
                continue
            x, env = r["x"], r["env"]
            n = len(r["f0"])
            xp = torch.from_numpy(T1.prime(x))[None].to(dev)
            p = net(mfront(xp))[..., -(len(x) // N.HOP):].softmax(1)[0].T.cpu().numpy()
            p = np.pad(p, ((0, max(0, n - len(p))), (0, 0)), mode="edge")[:n]
            u = p.argmax(1)
            c = env[1:25, :n]
            cut = int(n * 0.4)
            def stats(sl):
                M = np.zeros((24, K), np.float32); cnt = np.zeros(K, np.float32)
                for j in range(K):
                    m = u[sl] == j
                    if m.any():
                        M[:, j] = c[:, sl][:, m].mean(1); cnt[j] = m.sum()
                return M, cnt
            D1.append(stats(slice(0, cut))); D2.append(stats(slice(cut, n)))
    S = len(D1)
    print("speakers", S, flush=True)
    M1 = np.stack([d[0] for d in D1]); N1 = np.stack([d[1] for d in D1])
    M2 = np.stack([d[0] for d in D2]); N2 = np.stack([d[1] for d in D2])
    split = S // 2
    dv = np.arange(S) < split
    te = ~dv
    w1 = N1[dv][:, None, :]
    mu_u = (M1[dv] * w1).sum(0) / np.maximum(w1.sum(0), 1)
    Dm = ((M1[dv] - mu_u[None]) * (N1[dv][:, None, :] > 0))
    X = np.transpose(Dm, (0, 2, 1))[N1[dv] > 0]
    U, Sg, Vt = np.linalg.svd(X[np.random.RandomState(0).permutation(len(X))[:200000]], full_matrices=False)
    Pb = Vt[: a.m].T

    def shrink(M, Nn, lam, factor):
        out = np.zeros_like(M)
        for s in range(len(M)):
            n_u = Nn[s]
            if factor:
                r = (M[s] - mu_u) * (n_u[None] > 0)
                z = (Pb.T @ (r * n_u[None]).sum(1)) / max(n_u.sum(), 1.0)
                prior = mu_u + (Pb @ z)[:, None]
            else:
                prior = mu_u
            out[s] = (M[s] * n_u[None] + lam * prior) / (n_u[None] + lam) if lam > 0 else np.where(n_u[None] > 0, M[s], prior)
        return out

    def raw_borrow(M, Nn):
        out = M.copy()
        sim = Cn @ Cn.T
        for s in range(len(M)):
            has = Nn[s] > 0
            if (~has).any():
                sm = sim.copy(); sm[:, ~has] = -9
                out[s][:, ~has] = M[s][:, sm[~has].argmax(1)]
        return out

    def err(T):
        w = N2[te][:, None, :]
        d = (T[te] - M2[te]) ** 2
        return float(np.sqrt((d * w).sum() / (w.sum() * 24)))

    rep = {"n_dev": int(dv.sum()), "n_test": int(te.sum()), "rms_between_spk_truth": float(np.sqrt((((M2[te] - mu_u) ** 2) * N2[te][:, None, :]).sum() / (N2[te].sum() * 24))),
           "raw_borrow": err(raw_borrow(M1, N1)), "unit_mean_only": err(np.broadcast_to(mu_u, M1.shape))}
    for factor in (False, True):
        for lam in (0, 2, 5, 10, 20, 50, 100):
            rep[f"{'factor' if factor else 'mean'}_lam{lam}"] = err(shrink(M1, N1, lam, factor))
    print(json.dumps(rep, indent=1))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
