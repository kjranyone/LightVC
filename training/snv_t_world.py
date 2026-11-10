"""段 T の合成の世界(paper_snv §6・diag・製品の判断に引かない): 演奏と構造が既知で、正解の reamp F(u_o; φ_t) を作れる試験の世界。
演奏 u(8 次元)= 内容(状態の原型 40 個の並び)を、発話ごとの区間の長さの揺れ・目標の揺れ・慣性(一次の遅れ)・雑音で演じたもの = 内容を与えたときの分布は話者に依らない(仮定 A2)。
構造 φ_s = 演奏へのアフィン(A_s・c_s)→ 共通の非線形の基底 B → 周波数の軸の一律の伸縮 α_s(声道の長さ・女 1.15 / 男 1.00 の 2 群)+ 演奏に依る小さな偏り + 傾き → 29 次元の「包絡」+ 観測の雑音。
話者の個人差は全て 6 個の潜在の因子 z_s(と性別)× 共通の基底で作る(解剖は少ない自由度 = 仮定 A1 が試験の世界で成り立つ・results/snv_t/prereg.yaml amendment_2)。
rot > 0(T9・paper_snv 系 2e): 演奏に話者ごとの大きな回転 Q_s = expm(rot · Σ_k z_k G_k / √6)(G_k = 固定の乱数の歪対称)を A_s の前に掛ける = 同じ演奏が声ごとに別の音になる(回転も 6 個の因子から)。
手で決めたこの写像は試験の世界にだけ使い、模型にも製品の学習のデータにも入れない(paper_snv §9 O2)。
    cd training && uv run --no-sync python snv_t_world.py   # 世界の性質の自己検査
"""
from __future__ import annotations

import numpy as np

D_U, D_Y, N_PROTO = 8, 29, 40
SEG_FR, SEG_SD = 12.0, 0.3
JIT, KAPPA, U_NOISE, Y_NOISE = 0.15, 0.25, 0.02, 0.02
N_TRAIN, N_TEST, N_UTT = 180, 20, 40
N_Z = 6


class World:
    def __init__(self, n_spk: int = N_TRAIN + N_TEST, seed: int = 0, rot: float = 0.0) -> None:
        g = np.random.default_rng(seed)
        self.proto = g.normal(size=(N_PROTO, D_U))
        self.W1, self.b1 = g.normal(size=(D_U, 64)) / np.sqrt(D_U), g.normal(size=64) * 0.3
        self.W2, self.b2 = g.normal(size=(64, 64)) / 8.0, g.normal(size=64) * 0.3
        self.W3 = g.normal(size=(64, D_Y)) / 8.0
        half = n_spk // 2
        self.sex = np.array([0] * half + [1] * (n_spk - half))
        self.z = g.normal(size=(n_spk, N_Z))
        Rk = g.normal(size=(N_Z, D_U, D_U)) / np.sqrt(D_U)
        ck = g.normal(size=(N_Z, D_U))
        self.V = g.normal(size=(N_Z, D_U, 16)) / np.sqrt(D_U)
        self.P = g.normal(size=(N_Z, 16, D_Y)) / 4.0
        self.alpha = np.where(self.sex == 0, 1.15, 1.0) + 0.04 * self.z[:, 0]
        self.A = np.eye(D_U)[None] + 0.3 * np.einsum("sk,kij->sij", self.z, Rk) / np.sqrt(N_Z)
        if rot > 0:
            from scipy.linalg import expm
            gr = np.random.default_rng(seed + 77)
            G = gr.normal(size=(N_Z, D_U, D_U))
            G = G - G.transpose(0, 2, 1)
            G = G / np.linalg.norm(G, axis=(1, 2), keepdims=True) * np.sqrt(D_U)
            Q = np.stack([expm(rot * np.einsum("k,kij->ij", self.z[s], G) / np.sqrt(N_Z)) for s in range(n_spk)])
            self.A = np.einsum("sij,sjk->sik", self.A, Q)
        self.c = 0.3 * self.z @ ck / np.sqrt(N_Z)
        self.tilt = 0.3 * self.z[:, 1]
        self.lin = np.linspace(-1.0, 1.0, D_Y)
        self.grid = np.arange(D_Y, dtype=np.float64)
        perm = g.permutation(n_spk)
        f, m = [i for i in perm if self.sex[i] == 0], [i for i in perm if self.sex[i] == 1]
        nt = N_TEST // 2
        self.test = f[:nt] + m[:nt]
        self.train = [i for i in perm if i not in self.test]

    def base(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        return h @ self.W3

    def warp(self, e: np.ndarray, a: float) -> np.ndarray:
        return np.stack([np.interp(self.grid / a, self.grid, row) for row in e])

    def render(self, u: np.ndarray, s: int, rng: np.random.Generator | None = None) -> np.ndarray:
        e = self.warp(self.base(u @ self.A[s].T + self.c[s]), self.alpha[s])
        dev = sum(self.z[s, k] * (np.tanh(u @ self.V[k]) @ self.P[k]) for k in range(N_Z)) * 0.3 / np.sqrt(N_Z)
        e = e + dev + self.tilt[s] * self.lin
        return e + (rng.normal(0, Y_NOISE, e.shape) if rng is not None else 0.0)

    def content(self, rng: np.random.Generator) -> np.ndarray:
        n = int(rng.integers(6, 13))
        seq = [int(rng.integers(N_PROTO))]
        while len(seq) < n:
            k = int(rng.integers(N_PROTO))
            if k != seq[-1]:
                seq.append(k)
        return np.array(seq)

    def playing(self, content: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        tg = []
        for k in content:
            d = max(3, int(round(np.exp(rng.normal(np.log(SEG_FR), SEG_SD)))))
            tg.append(np.repeat((self.proto[k] + rng.normal(0, JIT, D_U))[None], d, 0))
        tg = np.concatenate(tg)
        u = np.zeros_like(tg)
        u[0] = tg[0]
        for t in range(1, len(tg)):
            u[t] = u[t - 1] + KAPPA * (tg[t] - u[t - 1])
        return u + rng.normal(0, U_NOISE, u.shape)


def design(w: World, K: int, cross_sex: bool = True, seed: int = 0, n_utt: int = N_UTT) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    rng = np.random.default_rng(1000 + seed)
    groups = [w.train] if cross_sex else [[s for s in w.train if w.sex[s] == 0], [s for s in w.train if w.sex[s] == 1]]
    contents, pairs = [], []
    for grp in groups:
        cap = {s: n_utt for s in grp}
        while True:
            avail = [s for s in grp if cap[s] > 0]
            if len(avail) < K:
                break
            ci = len(contents)
            contents.append(w.content(rng))
            for s in rng.choice(avail, K, replace=False):
                pairs.append((int(s), ci))
                cap[int(s)] -= 1
    return contents, pairs


def selftest() -> None:
    w = World()
    rng = np.random.default_rng(7)
    assert len(w.test) == N_TEST and len(set(w.test) & set(w.train)) == 0 and sum(w.sex[w.test]) == N_TEST // 2
    e = w.base(rng.normal(size=(5, D_U)))
    assert np.allclose(w.warp(e, 1.0), e), "伸縮 1.0 が恒等でない"
    us = {s: [w.playing(w.content(rng), rng) for _ in range(6)] for s in w.test}
    gap, noise = [], []
    for o in w.test:
        for t in w.test:
            if o != t:
                u = us[o][0]
                gap.append(np.abs(w.render(u, o) - w.render(u, t)).mean())
        noise.append(np.abs(w.render(us[o][0], o, rng) - w.render(us[o][0], o)).mean())
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    from sklearn.linear_model import LogisticRegression
    X_u, X_y, lab = [], [], []
    for i, s in enumerate(w.test):
        for u in us[s]:
            X_u.append(u), X_y.append(w.render(u, s, rng)), lab.append(np.full(len(u), i))
    X_u, X_y, lab = np.concatenate(X_u), np.concatenate(X_y), np.concatenate(lab)
    idx = rng.permutation(len(lab))
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
    acc_u = QDA(reg_param=0.1).fit(X_u[tr], lab[tr]).score(X_u[te], lab[te])
    acc_y = LogisticRegression(max_iter=2000).fit(X_y[tr], lab[tr]).score(X_y[te], lab[te])
    c, p = design(w, 4)
    print(f"world ok: copy gap |F(u;φ_o) − F(u;φ_t)| {np.mean(gap):.3f} / obs noise {np.mean(noise):.3f} / y std {X_y.std():.3f} / u std {X_u.std():.3f}")
    print(f"speaker id (20 声・偶然 0.05): true u QDA {acc_u:.3f} / y logistic {acc_y:.3f}")
    print(f"design K=4: contents {len(c)} utts {len(p)} speakers {len({s for s, _ in p})} utt frames ~{np.mean([len(x) for x in c]) * SEG_FR:.0f}")


if __name__ == "__main__":
    selftest()
