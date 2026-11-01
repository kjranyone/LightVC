"""A3d の事前検査(学習なしの探り): 差し替えた E の入力から元の声 A を当てるよう学ぶ探り = G が使える残りの声の量。
探り = フレームだけの MLP(状態に条件づいた形)と時間の文脈つきの畳み込み(受容野 約 0.75 s)。差し替え = なし / 全体の統計(A3c)/ 状態ごとの写し。"""
from __future__ import annotations
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from sklearn.mixture import GaussianMixture
from train_snv_a2 import VAL_VOICES
from train_snv_a3 import envelope, load_corpus, sqrtm_pair

DEV = "cuda"
K_STATES = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N_A, N_POOL, TR_UTT = 60, 200, 7
rng = np.random.default_rng(0)
torch.manual_seed(0)
voices = load_corpus()
train = sorted(voices)[:-VAL_VOICES]
pool = sorted(rng.choice(train, N_POOL, replace=False).tolist())
A = pool[:N_A]
with torch.no_grad():
    env = {n: [envelope(torch.from_numpy(np.asarray(w, dtype=np.float32))[None].to(DEV))[0].T.contiguous() for _, _, w in voices[n]] for n in pool}
gst = {}
for n in pool:
    X = torch.cat(env[n])
    cs, ci = sqrtm_pair(torch.cov(X.T))
    gst[n] = (X.mean(0), cs, ci)
white = {n: [(x - gst[n][0]) @ gst[n][2] for x in env[n]] for n in pool}
Z = torch.cat([torch.cat(white[n]) for n in pool]).cpu().numpy()
sub = Z[rng.choice(len(Z), min(200000, len(Z)), replace=False)]
gmm = GaussianMixture(K_STATES, covariance_type="diag", random_state=0, max_iter=200).fit(sub)
gm_mu = torch.tensor(gmm.means_, dtype=torch.float32, device=DEV)
gm_var = torch.tensor(gmm.covariances_, dtype=torch.float32, device=DEV)
gm_lw = torch.tensor(np.log(gmm.weights_), dtype=torch.float32, device=DEV)


def post(z: torch.Tensor) -> torch.Tensor:
    ll = -0.5 * (((z[:, None] - gm_mu[None]) ** 2) / gm_var[None] + gm_var[None].log()).sum(-1) + gm_lw[None]
    return ll.softmax(-1)


sst = {}
for n in pool:
    z = torch.cat(white[n])
    g = post(z)
    w = g.sum(0)[:, None] + 1e-3
    m = (g.T @ z) / w
    v = (g.T @ (z ** 2)) / w - m ** 2
    lam = 20.0
    tot = g.sum(0)[:, None]
    sst[n] = (m * tot / (tot + lam), ((v * tot + lam) / (tot + lam)).clamp(min=1e-3))
print("states", K_STATES, "frames/voice/state min", float(min(post(torch.cat(white[n])).sum(0).min() for n in A)), flush=True)


def do_swap(x: torch.Tensor, a: str, b: str, mode: str) -> torch.Tensor:
    if mode == "raw":
        return x
    z = (x - gst[a][0]) @ gst[a][2]
    if mode in ("state", "state_cov"):
        g = post(z)
        ma, va = sst[a]
        mb, vb = sst[b]
        if mode == "state":
            z = z + g @ (mb - ma)
        else:
            z = g @ mb + (z * (g @ (vb / va).sqrt())) - (g @ (ma * (vb / va).sqrt()))
    return z @ gst[b][1] + gst[b][0]


allx = torch.cat([torch.cat(env[n]) for n in A])
MU, SD = allx.mean(0), allx.std(0)


class Frame(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(80, 512, 1), nn.ReLU(), nn.Conv1d(512, 512, 1), nn.ReLU(), nn.Conv1d(512, N_A, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


class Ctx(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        ls = [nn.Conv1d(80, 256, 5, padding=2), nn.ReLU()]
        for d in (2, 4, 8, 16):
            ls += [nn.Conv1d(256, 256, 3, padding=d, dilation=d), nn.ReLU()]
        self.f = nn.Sequential(*ls, nn.Conv1d(256, N_A, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


def others(a: str, r: np.random.Generator) -> str:
    while True:
        b = pool[int(r.integers(len(pool)))]
        if b != a:
            return b


def batch(mode: str, r: np.random.Generator, bs: int = 32, L: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for _ in range(bs):
        i = int(r.integers(N_A))
        u = int(r.integers(TR_UTT))
        x = env[A[i]][u]
        s = int(r.integers(max(1, len(x) - L)))
        xs.append(do_swap(x[s:s + L], A[i], others(A[i], r), mode))
        ys.append(i)
    L2 = min(len(x) for x in xs)
    return torch.stack([((x[:L2] - MU) / SD).T for x in xs]), torch.tensor(ys, device=DEV)


def run(mode: str, kind: str, steps: int = 3000) -> dict:
    r = np.random.default_rng(1)
    net = (Frame() if kind == "frame" else Ctx()).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(steps):
        x, y = batch(mode, r)
        lg = net(x)
        loss = F.cross_entropy(lg, y[:, None].expand(-1, lg.shape[-1]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    rt = np.random.default_rng(2)
    utt, fr = [], []
    with torch.no_grad():
        for i, a in enumerate(A):
            for u in range(TR_UTT, len(env[a])):
                for _ in range(2):
                    x = do_swap(env[a][u], a, others(a, rt), mode)
                    lp = net(((x - MU) / SD).T[None])[0].log_softmax(0)
                    utt.append(int(lp.mean(1).argmax()) == i)
                    fr.append(float((lp.argmax(0) == i).float().mean()))
    return {"utt": round(float(np.mean(utt)), 3), "frame": round(float(np.mean(fr)), 3)}


with torch.no_grad():
    agree = []
    for a in A[:20]:
        b = others(a, rng)
        for x in env[a][TR_UTT:]:
            g0 = post((x - gst[a][0]) @ gst[a][2]).argmax(-1)
            for mode in ("global", "state", "state_cov"):
                y = do_swap(x, a, b, mode)
                g1 = post((y - gst[b][0]) @ gst[b][2]).argmax(-1)
                agree.append((mode, float((g0 == g1).float().mean())))
    for mode in ("global", "state", "state_cov"):
        print("state agreement(差し替えの前後で状態が同じ率)", mode, round(float(np.mean([v for m, v in agree if m == mode])), 3), flush=True)
print("chance", round(1 / N_A, 3), flush=True)
for mode in ("raw", "global", "state", "state_cov"):
    for kind in ("frame", "ctx"):
        print(mode, kind, run(mode, kind), flush=True)
