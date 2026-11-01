"""D21(artic_vc §8e の (3)・GPU の学習なし): 参照 20s の要約が、評価発話の単位の中の細かい動きを、別の話者の要約より良く予測するか。
動きの要約 D_s(u, p) = 参照の残差(包絡 − 参照の単位平均)を 単位 u × 単位の連続の中の位置 p(0 / 1 / 2 / 3 / 4+ フレーム目)で平均し、観測数で 0 へ縮約(λ)。
予測の対象 = 評価発話の 包絡 − その発話の単位平均(INS)= 単位の中の動き。予測 = D(u_t, p_t) をその発話の中で単位ごとに中心化したもの。
指標 = 説明率 R² = 1 − Σ(r − pred)² / Σ r²(有声・活性のフレーム)。自分の要約 / 母集団平均の要約 / 別の話者 10 人の要約 を比べ、目標ごとの差で符号検定。
c0 の動きも同様(1 次元)。
    uv run python d21_dyn_predict.py --out ../results/conv_p0/d21.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import diag_pool as P

NP = 5


def runpos(u: np.ndarray) -> np.ndarray:
    p = np.zeros(len(u), int)
    for t in range(1, len(u)):
        p[t] = p[t - 1] + 1 if u[t] == u[t - 1] else 0
    return np.minimum(p, NP - 1)


def unit_means(x: np.ndarray, u: np.ndarray, K: int) -> np.ndarray:
    s = np.zeros((x.shape[0], K)); c = np.zeros(K)
    np.add.at(s.T, u, x.T); np.add.at(c, u, 1.0)
    return s / np.maximum(c, 1)


def dyn_summary(x: np.ndarray, u: np.ndarray, K: int, lam: float) -> np.ndarray:
    r = x - unit_means(x, u, K)[:, u]
    p = runpos(u)
    s = np.zeros((x.shape[0], K, NP)); c = np.zeros((K, NP))
    np.add.at(s, (slice(None), u, p), r)
    np.add.at(c, (u, p), 1.0)
    return s / (c[None] + lam)


def predict(D: np.ndarray, u: np.ndarray, K: int) -> np.ndarray:
    pr = D[:, u, runpos(u)]
    return pr - unit_means(pr, u, K)[:, u]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--n_other", type=int, default=10)
    a = ap.parse_args()
    import pae as PA
    K = 200
    pool = P.load_pool(150, 3)
    refs = P.load_refs(150, 3)
    S = len(pool)
    Denv = [dyn_summary(r["env"].astype(np.float64), r["u"].astype(int), K, a.lam) for r in refs]
    Dc0 = [dyn_summary(r["c0"][None].astype(np.float64), r["u"].astype(int), K, a.lam) for r in refs]
    Penv = np.mean(Denv, 0); Pc0 = np.mean(Dc0, 0)
    rng = np.random.default_rng(0)
    rep: dict = {"n_targets": S, "lam": a.lam}
    for name, Ds, Pop, getx in (("env", Denv, Penv, lambda ut: ut["er"]), ("c0", Dc0, Pc0, lambda ut: ut["c0"][None])):
        own, pop, oth = [], [], []
        for ti, t in enumerate(pool):
            others = rng.choice([j for j in range(S) if j != ti], a.n_other, replace=False)
            so, sp, ss = [], [], []
            for ut in t["utts"]:
                u = ut["u"].astype(int)
                x = getx(ut).astype(np.float64)
                act = ((ut["c0"] - PA.C0_SIL) / 10 > 1.0) & (ut["f0"] > 0)
                if act.sum() < 50:
                    continue
                r = (x - unit_means(x, u, K)[:, u])[:, act]
                den = float((r ** 2).sum())
                r2 = lambda D: 1 - float(((r - predict(D, u, K)[:, act]) ** 2).sum()) / den
                so.append(r2(Ds[ti])); sp.append(r2(Pop)); ss.append(np.mean([r2(Ds[j]) for j in others]))
            if so:
                own.append(np.mean(so)); pop.append(np.mean(sp)); oth.append(np.mean(ss))
        own, pop, oth = np.array(own), np.array(pop), np.array(oth)
        from scipy.stats import wilcoxon
        rep[name] = {"r2_own": round(float(own.mean()), 4), "r2_population": round(float(pop.mean()), 4), "r2_other_mean": round(float(oth.mean()), 4),
                     "own_minus_other": round(float((own - oth).mean()), 4), "own_minus_pop": round(float((own - pop).mean()), 4),
                     "targets_own>other": int((own > oth).sum()), "targets_own>pop": int((own > pop).sum()), "n": int(len(own)),
                     "p_own>other": float(wilcoxon(own, oth, alternative="greater").pvalue), "p_own>pop": float(wilcoxon(own, pop, alternative="greater").pvalue)}
        print(name, rep[name], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
