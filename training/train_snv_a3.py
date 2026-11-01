"""A3(synth_nam_vc §5・diag): 自前の因果の演奏の信号。A2 の x の G の演奏の入力のうち、調音の部分(SPARC の EMA 12 次元)を自前の因果の符号器 E(12 次元)に替える。
E の入力 = 左寄せの窓(2048 点・hop 512・先読み 0)の対数振幅をケプストラムで 2.5 ms に切った包絡(倍音の櫛を除く)を mel の間隔の 80 点で読んだもの。
声源(log f0・大きさ・周期性 = SPARC)と描画(BigVGAN)は A2 と同じ代役なので diag。
--perturb p = E の入力にだけ発話ごとの周波数の伸縮(α)・なめらかな EQ・利得を掛ける(目標の側は加工しない)/ n = 掛けない(対照)。
--adv λ(A3b)= E の出力から学習の声(570)を当てる識別器 D(9 フレームの窓の畳み込み → 切り出しの全体で平均と標準偏差を束ねる)を別に学び、
E には D の事後確率を一様へ寄せる損失を λ 倍で足す(不変性の正則化・分布の段)。声は発話の全体で一定なので、フレームの窓だけの識別器は騙せても発話で束ねると残る(煙テスト)。
--swap(A3c)= E の入力の包絡を、学習の声ごとの平均・共分散で白色化し、発話ごとに乱数で選んだ別の学習の声 B の平均・共分散で着色し直す(話者の統計の差し替え)。
目標・参照・声源は元の声 A のまま = 入力側だけの加工。E に残る声の手がかりは B のもの = 目標と無関係になり、参照の鍵(A)と質問(B 風)が話者を跨ぐ。
--swap_mode state / state_cov(A3d)= 全体の差し替えに加えて、白色化した空間で調音の状態(学習の声の白色化した包絡に当てはめた共通の GMM・--states 成分)ごとに、
声 A の状態ごとの平均(state)と標準偏差(state_cov)を B のものへ写す(状態の事後確率で重みづけ)。全体の差し替えが残す状態ごとの包絡の形を B へ替える。
LEAK(検証の 30 声・前半の発話で当てはめ後半で当てる・E の出力の線形と QDA)と XFER(検証の声 A の発話を次の検証の声 B の参照で変換し、
本物の mel で当てはめた QDA が A と B のどちらと読むか)と XFER_ECAPA(同じ変換を BigVGAN で描き、本物の 3 本の ECAPA + LDA の 30 声中の 1 位で読む = 評価の主の物差しと同じ系統)
を step 5k・10k・最後に印字し、10k で XFER の A と読む率が --leak_trip 以上か XFER_ECAPA の A と読む率が --ecapa_trip 以上なら中止。
事前登録 = results/<tag>/prereg.yaml(L/I/C)。
    cd training && uv run --no-sync python train_snv_a3.py --perturb p --tag diag_snv_a3_p
    cd training && uv run --no-sync python train_snv_a3.py --perturb n --swap --leak_trip 0.5 --tag diag_snv_a3c_w
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import random
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_snv_a2 import CROP, FEAT, N_MEL, P_DIM, REF_UTT, ROOT, VAL_VOICES, WARM, G, l1, playing

SR = 44100
HOP = 512
NFFT = 2048
LIFT = 110
N_ENV = 80
D_E = 12
ENV_F = librosa.mel_frequencies(n_mels=N_ENV, fmin=60.0, fmax=12000.0)
ALPHA = (0.8, 1.25)
EQ_AMP = 0.3
GAIN = 0.7


def envelope(y: torch.Tensor, alpha: torch.Tensor | None = None, left_pad: bool = True) -> torch.Tensor:
    if left_pad:
        y = F.pad(y, (NFFT - HOP, 0))
    X = torch.stft(y, NFFT, HOP, window=torch.hann_window(NFFT, device=y.device), center=False, return_complex=True)
    c = torch.fft.irfft(torch.log(X.abs() + 1e-4), n=NFFT, dim=1)
    w = torch.zeros(NFFT, device=y.device)
    w[:LIFT] = 1.0
    w[-LIFT + 1:] = 1.0
    S = torch.fft.rfft(c * w[None, :, None], dim=1).real
    f = torch.as_tensor(ENV_F, dtype=torch.float32, device=y.device)[None].expand(y.shape[0], -1)
    if alpha is not None:
        f = f / alpha[:, None]
    idx = f * NFFT / SR
    i0 = idx.floor().long().clamp(0, S.shape[1] - 2)
    fr = (idx - i0.float())[..., None]
    g0 = S.gather(1, i0[..., None].expand(-1, -1, S.shape[2]))
    g1 = S.gather(1, (i0 + 1)[..., None].expand(-1, -1, S.shape[2]))
    return g0 * (1 - fr) + g1 * fr


def perturb_params(B: int, device: str, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.rand(B, generator=gen, device=device)
    alpha = torch.exp(math.log(ALPHA[0]) + u * (math.log(ALPHA[1]) - math.log(ALPHA[0])))
    m = torch.arange(N_ENV, device=device, dtype=torch.float32)[None, :]
    eq = torch.zeros(B, N_ENV, device=device)
    for j in (1, 2, 3):
        a = (torch.rand(B, 1, generator=gen, device=device) * 2 - 1) * EQ_AMP
        ph = torch.rand(B, 1, generator=gen, device=device) * 2 * math.pi
        eq = eq + a * torch.cos(math.pi * j * m / (N_ENV - 1) + ph)
    eq = eq + (torch.rand(B, 1, generator=gen, device=device) * 2 - 1) * GAIN
    return alpha, eq


class EBlock(nn.Module):
    def __init__(self, ch: int, dil: int) -> None:
        super().__init__()
        self.dil = dil
        self.conv = nn.Conv1d(ch, 2 * ch, 3, dilation=dil)
        self.out = nn.Conv1d(ch, ch, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a, b = self.conv(F.pad(h, (2 * self.dil, 0))).chunk(2, 1)
        return h + self.out(torch.tanh(a) * torch.sigmoid(b))


class E(nn.Module):
    def __init__(self, ch: int = 256) -> None:
        super().__init__()
        self.inp = nn.Conv1d(N_ENV, ch, 1)
        self.blocks = nn.ModuleList([EBlock(ch, d) for d in (1, 2, 4, 8, 1, 2)])
        self.out = nn.Conv1d(ch, D_E, 1)

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        h = self.inp(env)
        for b in self.blocks:
            h = b(h)
        return self.out(h).transpose(1, 2)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e = E()
        self.g = G("x")
        self.register_buffer("env_mu", torch.zeros(N_ENV, 1))
        self.register_buffer("env_sd", torch.ones(N_ENV, 1))

    def encode(self, env: torch.Tensor) -> torch.Tensor:
        return self.e((env - self.env_mu) / self.env_sd)

    def play(self, env: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encode(env), src], -1)

    def structure(self, rmel: torch.Tensor, renv: torch.Tensor, rsrc: torch.Tensor, rmask: torch.Tensor) -> dict:
        return self.g.structure(rmel, self.play(renv, rsrc), rmask)

    def forward(self, env: torch.Tensor, src: torch.Tensor, s: dict, return_attn: bool = False):
        return self.g(self.play(env, src), s, return_attn=return_attn)


class Spk(nn.Module):
    def __init__(self, n: int, ch: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(D_E, ch, 5, padding=2), nn.GELU(), nn.Conv1d(ch, ch, 5, padding=2), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * ch, ch), nn.GELU(), nn.Linear(ch, n))

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.conv(z.transpose(1, 2))
        m = mask[:, None]
        k = m.sum(-1).clamp(min=1.0)
        mu = (h * m).sum(-1) / k
        sd = ((((h - mu[..., None]) ** 2) * m).sum(-1) / k + 1e-5).sqrt()
        return self.head(torch.cat([mu, sd], -1))


def confusion(logits: torch.Tensor) -> torch.Tensor:
    return -logits.log_softmax(-1).mean(-1).mean() - math.log(logits.shape[-1])


def spk_ce(logits: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, spk)


def sqrtm_pair(C: torch.Tensor, shrink: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    d = C.shape[-1]
    C = C + shrink * C.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] / d * torch.eye(d, device=C.device)
    w, U = torch.linalg.eigh(C)
    w = w.clamp(min=1e-6)
    return (U * w.sqrt()[..., None, :]) @ U.transpose(-1, -2), (U * w.rsqrt()[..., None, :]) @ U.transpose(-1, -2)


def swap_stats(voices: dict, names: list[str], dev: str) -> dict:
    mu, cs, ci = [], [], []
    with torch.no_grad():
        for n in names:
            X = torch.cat([envelope(torch.from_numpy(np.asarray(w, dtype=np.float32))[None].to(dev))[0] for _, _, w in voices[n]], 1).T
            m = X.mean(0)
            s_, i_ = sqrtm_pair(torch.cov(X.T))
            mu.append(m), cs.append(s_), ci.append(i_)
    return {"mu": torch.stack(mu), "cs": torch.stack(cs), "ci": torch.stack(ci)}


def state_post(z: torch.Tensor, st: dict) -> torch.Tensor:
    d = z[..., None, :] - st["gm_mu"]
    return (-0.5 * (d * d / st["gm_var"] + st["gm_var"].log()).sum(-1) + st["gm_lw"]).softmax(-1)


def state_stats(voices: dict, names: list[str], sw: dict, dev: str, k: int, lam: float = 20.0) -> dict:
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(0)

    def white(i: int, n: str) -> torch.Tensor:
        x = torch.cat([envelope(torch.from_numpy(np.asarray(w, dtype=np.float32))[None].to(dev))[0] for _, _, w in voices[n]], 1).T
        return (x - sw["mu"][i]) @ sw["ci"][i]

    with torch.no_grad():
        sub = []
        for i, n in enumerate(names):
            z = white(i, n).cpu().numpy()
            sub.append(z[rng.choice(len(z), min(len(z), 200000 // len(names) + 1), replace=False)])
        gm = GaussianMixture(k, covariance_type="diag", random_state=0, max_iter=200).fit(np.concatenate(sub))
        st = {"gm_mu": torch.tensor(gm.means_, dtype=torch.float32, device=dev), "gm_var": torch.tensor(gm.covariances_, dtype=torch.float32, device=dev),
              "gm_lw": torch.tensor(np.log(gm.weights_), dtype=torch.float32, device=dev)}
        sm, sv = [], []
        for i, n in enumerate(names):
            z = white(i, n)
            g = state_post(z, st)
            tot = g.sum(0)[:, None]
            m = (g.T @ z) / (tot + 1e-3)
            v = (g.T @ (z * z)) / (tot + 1e-3) - m * m
            sm.append(m * tot / (tot + lam))
            sv.append(((v * tot + lam) / (tot + lam)).clamp(min=1e-3))
    st.update({"sm": torch.stack(sm), "sv": torch.stack(sv)})
    return st


def swap(env: torch.Tensor, st: dict, a: torch.Tensor, b: torch.Tensor, mode: str = "global") -> torch.Tensor:
    z = (env.transpose(1, 2) - st["mu"][a][:, None]) @ st["ci"][a]
    if mode != "global":
        g = state_post(z, st)
        ma, mb = st["sm"][a], st["sm"][b]
        if mode == "state":
            z = z + g @ (mb - ma)
        else:
            r = (st["sv"][b] / st["sv"][a]).sqrt()
            z = g @ mb + z * (g @ r) - g @ (ma * r)
    return (z @ st["cs"][b] + st["mu"][b][:, None]).transpose(1, 2)


def xfer_outputs(model: "Model", voices: dict, val: list[str], mu: np.ndarray, sd: np.ndarray, dev: str) -> list[tuple]:
    st = {}
    for n in val:
        f = np.concatenate([np.exp(s_[s_[:, 2] > 0.5, 0]) for _, s_, _ in voices[n]])
        st[n] = (float(f.mean()), float(f.std()))
    model.eval()
    outs = []
    with torch.no_grad():
        for i, a_ in enumerate(val):
            j = (i + 1) % len(val)
            b_ = val[j]
            refs = voices[b_][:REF_UTT]
            rm = np.concatenate([r[0] for r in refs], 1)
            rs = (np.concatenate([r[1] for r in refs], 0) - mu) / sd
            rw = np.concatenate([np.asarray(r[2], dtype=np.float32) for r in refs])
            s = model.structure(torch.from_numpy(rm)[None].to(dev), envelope(torch.from_numpy(rw)[None].to(dev)), torch.from_numpy(rs)[None].to(dev),
                                torch.ones(1, rm.shape[1], dtype=torch.bool, device=dev))
            for mel, src, w in voices[a_][REF_UTT + 3:]:
                q = src.copy()
                f0 = (np.exp(q[:, 0]) - st[a_][0]) / st[a_][1] * st[b_][1] + st[b_][0]
                q[:, 0] = np.log(np.maximum(f0, 1.0))
                out = model(envelope(torch.from_numpy(np.asarray(w, dtype=np.float32))[None].to(dev)), torch.from_numpy((q - mu) / sd)[None].to(dev), s)[0].cpu().numpy()
                outs.append((i, j, mel, out))
    model.train()
    return outs


def xfer_val(model: "Model", voices: dict, val: list[str], mu: np.ndarray, sd: np.ndarray, dev: str) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([m.T for n in val for m, _, _ in voices[n][REF_UTT:REF_UTT + 3]])
    y = np.concatenate([np.full(m.shape[1], i) for i, n in enumerate(val) for m, _, _ in voices[n][REF_UTT:REF_UTT + 3]])
    clf = make_pipeline(StandardScaler(), PCA(40), QuadraticDiscriminantAnalysis(reg_param=0.05)).fit(X, y)
    pred = lambda m: int(clf.classes_[np.argmax(clf.predict_log_proba(m.T).sum(0))])
    real, to_b, to_a = [], [], []
    for i, j, mel, out in xfer_outputs(model, voices, val, mu, sd, dev):
        k = pred(out)
        real.append(pred(mel) == i), to_b.append(k == j), to_a.append(k == i)
    return {"real_acc": float(np.mean(real)), "to_target": float(np.mean(to_b)), "read_as_source": float(np.mean(to_a))}


def xfer_ecapa(model: "Model", voices: dict, val: list[str], mu: np.ndarray, sd: np.ndarray, dev: str) -> dict:
    from speechbrain.inference.speaker import EncoderClassifier
    from snv_a2_eval import vocoder
    T = np.load(FEAT / "ecapa_lda_T.npz")
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": dev})

    def emb(y: np.ndarray) -> np.ndarray:
        y16 = librosa.resample(y.astype(np.float32), orig_sr=SR, target_sr=16000)
        e = F.normalize(ec.encode_batch(torch.from_numpy(y16).float()[None].to(dev)).reshape(-1), dim=0).cpu().numpy()
        z = (e - T["mu"]) @ T["scalings"]
        return z / np.linalg.norm(z)

    outs = xfer_outputs(model, voices, val, mu, sd, dev)
    voc = vocoder()
    with torch.no_grad():
        anc = np.stack([emb(np.concatenate([np.asarray(r[2], dtype=np.float32) for r in voices[n][REF_UTT:REF_UTT + 3]])) for n in val])
        to_b, to_a = [], []
        for i, j, _, out in outs:
            k = int(np.argmax(anc @ emb(voc(torch.from_numpy(out)[None].to(dev)).squeeze().cpu().numpy())))
            to_b.append(k == j), to_a.append(k == i)
    del voc, ec
    torch.cuda.empty_cache()
    return {"to_target": float(np.mean(to_b)), "read_as_source": float(np.mean(to_a))}


def leak_val(model: "Model", voices: dict, val: list[str], dev: str) -> dict:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model.eval()
    with torch.no_grad():
        Z = {n: [model.encode(envelope(torch.from_numpy(np.asarray(w, dtype=np.float32))[None].to(dev)))[0].cpu().numpy() for _, _, w in voices[n]] for n in val}
    model.train()
    h = min(len(v) for v in Z.values()) // 2
    Xtr = np.concatenate([z[::2] for n in val for z in Z[n][:h]])
    ytr = np.concatenate([np.full(len(z[::2]), i) for i, n in enumerate(val) for z in Z[n][:h]])
    te = [(i, z) for i, n in enumerate(val) for z in Z[n][h:]]
    out = {}
    for name, clf in (("qda", QuadraticDiscriminantAnalysis(reg_param=0.05)), ("linear", LogisticRegression(C=1.0, max_iter=1000))):
        c = make_pipeline(StandardScaler(), clf).fit(Xtr, ytr)
        out[name] = float(np.mean([c.classes_[np.argmax(c.predict_log_proba(z).sum(0))] == i for i, z in te]))
    return out


def load_corpus() -> dict:
    man = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
    voices = {}
    for v in man["voices"]:
        utts = []
        for f in v["files"]:
            mp = FEAT / f"mel/e1/{v['voice']}/{f['t']}.npy"
            sp = FEAT / f"sparc/e1/{v['voice']}/{f['t']}.npz"
            wp = FEAT / f"wav44/e1/{v['voice']}/{f['t']}.npy"
            if mp.exists() and sp.exists() and wp.exists():
                mel = np.load(mp).astype(np.float32)
                utts.append((mel, playing(sp, mel.shape[1])[:, 12:15], np.load(wp, mmap_mode="r")))
        if len(utts) >= REF_UTT + 1:
            voices[v["voice"]] = utts
    return voices


class Batches:
    def __init__(self, voices: dict, names: list[str], mu: np.ndarray, sd: np.ndarray, seed: int) -> None:
        self.v, self.names, self.mu, self.sd = voices, names, mu, sd
        self.rng = random.Random(seed)

    def sample(self, bs: int) -> dict:
        tm, tw, ts, mask, rm, rw, rs, rl, sk = [], [], [], [], [], [], [], [], []
        W = (NFFT - HOP) + CROP * HOP
        for _ in range(bs):
            name = self.rng.choice(self.names)
            sk.append(self.names.index(name))
            us = self.v[name]
            idx = list(range(len(us)))
            self.rng.shuffle(idx)
            mel, src, wav = us[idx[0]]
            T = mel.shape[1]
            s = self.rng.randint(0, max(0, T - CROP))
            n = min(CROP, T - s)
            m = np.zeros((N_MEL, CROP), np.float32)
            q = np.zeros((CROP, 3), np.float32)
            k = np.zeros(CROP, np.float32)
            m[:, :n] = mel[:, s:s + n]
            q[:n] = (src[s:s + n] - self.mu) / self.sd
            k[WARM:n] = 1.0
            w = np.zeros(W, np.float32)
            a0 = s * HOP - (NFFT - HOP)
            seg = np.asarray(wav[max(0, a0):(s + n) * HOP], dtype=np.float32)
            w[max(0, -a0):max(0, -a0) + len(seg)] = seg
            tm.append(m), ts.append(q), mask.append(k), tw.append(w)
            refs = [us[i] for i in idx[1:1 + REF_UTT]]
            rm.append(np.concatenate([r[0] for r in refs], 1))
            rs.append((np.concatenate([r[1] for r in refs], 0) - self.mu) / self.sd)
            rw.append(np.concatenate([np.asarray(r[2], dtype=np.float32) for r in refs]))
            rl.append(rm[-1].shape[1])
        L = max(rl)
        RM = np.zeros((bs, N_MEL, L), np.float32)
        RS = np.zeros((bs, L, 3), np.float32)
        RW = np.zeros((bs, L * HOP), np.float32)
        RK = np.zeros((bs, L), bool)
        for i, (a, b, c, l) in enumerate(zip(rm, rs, rw, rl)):
            RM[i, :, :l], RS[i, :l], RW[i, :l * HOP], RK[i, :l] = a, b, c, True
        return {"mel": torch.from_numpy(np.stack(tm)), "wav": torch.from_numpy(np.stack(tw)), "src": torch.from_numpy(np.stack(ts)),
                "mask": torch.from_numpy(np.stack(mask)), "rmel": torch.from_numpy(RM), "rwav": torch.from_numpy(RW),
                "rsrc": torch.from_numpy(RS), "rmask": torch.from_numpy(RK), "spk": torch.tensor(sk)}


class Prefetch:
    def __init__(self, b: Batches, bs: int, depth: int = 8) -> None:
        self.q: queue.Queue = queue.Queue(depth)
        threading.Thread(target=self.fill, args=(b, bs), daemon=True).start()

    def fill(self, b: Batches, bs: int) -> None:
        while True:
            self.q.put({k: v.pin_memory() for k, v in b.sample(bs).items()})

    def next(self) -> dict:
        return self.q.get()


def real_wavs(k: int) -> list[np.ndarray]:
    ps = sorted((FEAT / "wav44/e1").glob("e*/t0*.npy"))[:k]
    return [np.load(p).astype(np.float32) for p in ps]


def unit_tests() -> dict:
    y, z = real_wavs(2)
    n = min(len(y), len(z)) // HOP * HOP
    y, z = torch.from_numpy(y[:n])[None], torch.from_numpy(z[:n])[None]
    full = envelope(y)
    assert full.shape[-1] == n // HOP, "包絡のフレーム数 != 長さ // 512"
    s = 37
    crop = envelope(y[:, s * HOP - (NFFT - HOP):(s + 100) * HOP], left_pad=False)
    assert torch.allclose(full[..., s:s + 100], crop, atol=1e-4), "切り出しの包絡が全体の包絡とずれる"
    assert torch.allclose(envelope(y, torch.ones(1)), full, atol=1e-5), "α = 1 の伸縮が恒等でない"
    m = Model().eval()
    src = torch.randn(1, n // HOP, 3)
    s_ = m.structure(torch.randn(1, N_MEL, 60), envelope(z[:, :60 * HOP]), torch.randn(1, 60, 3), torch.ones(1, 60, dtype=torch.bool))
    P = (n // HOP // 2) * HOP + 200
    y2 = y.clone()
    y2[:, P:] = z[:, P:]
    with torch.no_grad():
        o1, o2 = m(envelope(y), src, s_), m(envelope(y2), src, s_)
    safe = (P - HOP) // HOP
    assert torch.allclose(o1[..., :safe], o2[..., :safe], atol=1e-5), "E + G が未来の入力で変わる(未来不変性 FAIL)"
    assert not torch.allclose(o1[..., safe + 1:], o2[..., safe + 1:]), "編集が出力を動かさない(INCONCLUSIVE)"
    from scipy.signal import lfilter
    t = np.arange(SR) / SR

    def voiced(f0: float, poles: list[tuple[float, float]]) -> torch.Tensor:
        e = np.zeros(SR)
        e[(np.arange(0, 1, 1 / f0) * SR).astype(int)] = 1.0
        a = np.array([1.0])
        for fc, bw in poles:
            r = np.exp(-math.pi * bw / SR)
            a = np.convolve(a, [1.0, -2 * r * math.cos(2 * math.pi * fc / SR), r * r])
        return torch.from_numpy(lfilter([1.0], a, e).astype(np.float32))[None]

    va = [(700, 120), (1200, 120), (2600, 200)]
    vi = [(300, 100), (2300, 150), (3000, 200)]
    mid = slice(20, 70)
    d_pitch = float((envelope(voiced(200, va)) - envelope(voiced(320, va)))[..., mid].abs().mean())
    d_form = float((envelope(voiced(200, va)) - envelope(voiced(200, vi)))[..., mid].abs().mean())
    assert d_pitch < 0.5 * d_form, f"包絡が高さを運びすぎる(高さの差 {d_pitch:.3f} / 母音の差 {d_form:.3f})"
    g0 = torch.Generator().manual_seed(0)
    lg = torch.randn(2, 7, generator=g0)
    assert abs(float(confusion(torch.zeros(2, 7)))) < 1e-5 and float(confusion(lg)) > 0, "一様への損失が一様で 0・それ以外で正にならない"
    assert abs(float(spk_ce(torch.zeros(2, 7), torch.tensor([1, 3]))) - math.log(7)) < 1e-5, "話者の交差エントロピーが一様で log K にならない"
    with torch.random.fork_rng(devices=[]):
        d = Spk(7)
    zz = torch.randn(2, 40, D_E, generator=g0)
    mk = torch.zeros(2, 40)
    mk[:, 10:30] = 1.0
    z2 = zz.clone()
    z2[:, 35:] = torch.randn(2, 5, D_E, generator=g0)
    with torch.no_grad():
        assert torch.allclose(d(zz, mk), d(z2, mk), atol=1e-6), "識別器が切り出しの外(窓の届かない所)を見る"
        assert not torch.allclose(d(zz, mk), d(z2.index_fill(1, torch.tensor([20]), 3.0), mk)), "識別器が切り出しの内を見ない"
    Xa = (torch.randn(3000, 6, generator=g0) @ (0.5 * torch.randn(6, 6, generator=g0) + torch.eye(6)) + 2.0).double()
    Xb = (torch.randn(3000, 6, generator=g0) @ (0.5 * torch.randn(6, 6, generator=g0) + 2.0 * torch.eye(6)) - 1.0).double()
    sa, ia = sqrtm_pair(torch.cov(Xa.T), 0.0)
    sb, ib = sqrtm_pair(torch.cov(Xb.T), 0.0)
    st = {"mu": torch.stack([Xa.mean(0), Xb.mean(0)]), "cs": torch.stack([sa, sb]), "ci": torch.stack([ia, ib])}
    e0 = Xa.T[None]
    assert torch.allclose(swap(e0, st, torch.tensor([0]), torch.tensor([0])), e0, atol=1e-3), "同じ声への差し替えが恒等でない"
    xs = swap(e0, st, torch.tensor([0]), torch.tensor([1]))[0].T
    assert torch.allclose(xs.mean(0), Xb.mean(0), atol=1e-3) and float((torch.cov(xs.T) - torch.cov(Xb.T)).abs().max() / torch.cov(Xb.T).abs().max()) < 1e-6, "差し替えた包絡の平均・共分散が B に一致しない"
    st.update({"gm_mu": torch.randn(3, 6, generator=g0).double(), "gm_var": (0.5 + torch.rand(3, 6, generator=g0)).double(), "gm_lw": torch.log(torch.full((3,), 1 / 3)).double(),
               "sm": torch.randn(2, 3, 6, generator=g0).double(), "sv": (0.3 + torch.rand(2, 3, 6, generator=g0)).double()})
    assert torch.allclose(state_post(torch.randn(5, 6, generator=g0).double(), st).sum(-1), torch.ones(5, dtype=torch.float64)), "状態の事後確率の和が 1 でない"
    for md in ("state", "state_cov"):
        assert torch.allclose(swap(e0, st, torch.tensor([0]), torch.tensor([0]), md), e0, atol=1e-8), f"同じ声への状態ごとの差し替え({md})が恒等でない"
    eq_st = dict(st, sm=st["sm"][[0, 0]], sv=st["sv"][[0, 0]])
    for md in ("state", "state_cov"):
        assert torch.allclose(swap(e0, eq_st, torch.tensor([0]), torch.tensor([1]), md), swap(e0, st, torch.tensor([0]), torch.tensor([1])), atol=1e-8), f"状態の統計が同じ声どうしで {md} が全体の差し替えと一致しない"
    assert not torch.allclose(swap(e0, st, torch.tensor([0]), torch.tensor([1]), "state"), swap(e0, st, torch.tensor([0]), torch.tensor([1]))), "状態ごとの差し替えが全体の差し替えから動かない"
    print(f"unit tests ok(話者の統計の差し替え = 恒等と B への一致・状態ごとの差し替え = 恒等と状態の統計が同じなら全体に一致・識別器の束ね・包絡のフレーム数・切り出しの整合・α=1 の恒等・E + G の未来不変性 = 実音声で編集 {P} 以降・包絡の高さの頑健さ 高さの差 {d_pitch:.3f} / 母音の差 {d_form:.3f})", flush=True)
    return {"pitch_vs_vowel": d_pitch / d_form}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturb", choices=["p", "n"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adv", type=float, default=0.0)
    ap.add_argument("--leak_trip", type=float, default=1.01)
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--swap_mode", choices=["global", "state", "state_cov"], default="global")
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--ecapa_trip", type=float, default=1.01)
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print("prereg", out / "prereg.yaml", "args", vars(a), flush=True)
    print("ledger: E の入力 = 左寄せの窓(フレーム t は標本 t*512+512 まで)= 先読み 0・E と G は因果の畳み込み = 先読み 0・塊 = hop 11.6 ms。声源(SPARC)と描画(BigVGAN)は非因果の代役 = diag", flush=True)
    torch.manual_seed(a.seed)
    ut = unit_tests()
    voices = load_corpus()
    names = sorted(voices)
    val, tr = names[-VAL_VOICES:], names[:-VAL_VOICES]
    S = np.concatenate([s for n in tr for _, s, _ in voices[n]], 0)
    mu, sd = S.mean(0).astype(np.float32), (S.std(0) + 1e-4).astype(np.float32)
    print("voices", len(voices), "train", len(tr), "val", len(val), "utts", sum(len(voices[n]) for n in tr), flush=True)
    dev = "cuda"
    model = Model().to(dev)
    rng = random.Random(1)
    smp = [np.asarray(voices[n][rng.randrange(len(voices[n]))][2], dtype=np.float32) for n in rng.sample(tr, 200)]
    with torch.no_grad():
        env = torch.cat([envelope(torch.from_numpy(w)[None].to(dev))[0] for w in smp], 1)
    model.env_mu.copy_(env.mean(1, keepdim=True))
    model.env_sd.copy_(env.std(1, keepdim=True) + 1e-3)
    np.savez(out / "norm.npz", mu=mu, sd=sd)
    print("params E", sum(p.numel() for p in model.e.parameters()) / 1e6, "G", sum(p.numel() for p in model.g.parameters()) / 1e6, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    vab = Batches(voices, val, mu, sd, 12345)
    vbs = [vab.sample(a.bs) for _ in range(8)]
    trb = Prefetch(Batches(voices, tr, mu, sd, a.seed), a.bs)
    mel_mean = torch.from_numpy(np.concatenate([m for n in tr for m, _, _ in voices[n]], 1).mean(1)).to(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)

    sw = swap_stats(voices, tr, dev) if a.swap else None
    if sw is not None:
        if a.swap_mode != "global":
            sw.update(state_stats(voices, tr, sw, dev, a.states))
            g_ = torch.cat([state_post((torch.cat([envelope(torch.from_numpy(np.asarray(w, dtype=np.float32))[None].to(dev))[0] for _, _, w in voices[n]], 1).T - sw["mu"][i]) @ sw["ci"][i], sw) for i, n in enumerate(tr[:20])]).sum(0)
            print("state stats", a.states, "states・学習の声の先頭 20 声の状態ごとのフレーム数", [int(x) for x in g_.tolist()], flush=True)
        torch.save({k: v.cpu() for k, v in sw.items()}, out / "swap_stats.pt")
        print("swap stats", len(tr), "voices", a.swap_mode, flush=True)
    spk = Spk(len(tr)).to(dev) if a.adv > 0 else None
    sopt = torch.optim.AdamW(spk.parameters(), lr=1e-3, weight_decay=0.0) if spk is not None else None

    def step_loss(b: dict, pert: bool, sw_on: bool = False) -> tuple[torch.Tensor, torch.Tensor, dict]:
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        env = envelope(b["wav"], left_pad=False) if not pert else None
        if pert:
            alpha, eq = perturb_params(b["wav"].shape[0], dev, gen)
            env = envelope(b["wav"], alpha, left_pad=False) + eq[..., None]
        if sw_on:
            other = (b["spk"] + torch.randint(1, len(tr), b["spk"].shape, generator=gen, device=dev)) % len(tr)
            env = swap(env, sw, b["spk"], other, a.swap_mode)
        s = model.structure(b["rmel"], envelope(b["rwav"]), b["rsrc"], b["rmask"])
        z = model.encode(env)
        return l1(model.g(torch.cat([z, b["src"]], -1), s), b["mel"], b["mask"]), z, b

    def evaluate() -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            ls = [float(step_loss(b, False)[0]) for b in vbs]
            triv = [float(l1(mel_mean[None, :, None].expand_as(b["mel"]).to(dev), b["mel"].to(dev), b["mask"].to(dev))) for b in vbs]
        model.train()
        return float(np.mean(ls)), float(np.mean(triv))

    v0, triv = evaluate()
    print(f"L-4 floors: untrained val L1 {v0:.4f} / trivial mean-mel L1 {triv:.4f}", flush=True)
    log = []
    t0 = time.time()
    leaks = {}
    for step in range(1, a.steps + 1):
        loss, z, bd = step_loss(trb.next(), a.perturb == "p", sw is not None)
        tot, conf = loss, None
        if spk is not None:
            lam = a.adv * min(1.0, max(0.0, (step - 1000) / 4000))
            conf = confusion(spk(z, bd["mask"]))
            tot = loss + lam * conf
        opt.zero_grad(set_to_none=True)
        tot.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            print("non-finite grad, skip", step, flush=True)
            continue
        opt.step()
        sched.step()
        ce = None
        if spk is not None:
            ce = spk_ce(spk(z.detach(), bd["mask"]), bd["spk"])
            sopt.zero_grad(set_to_none=True)
            ce.backward()
            sopt.step()
        if step % 500 == 0:
            vl, _ = evaluate()
            rec = {"step": step, "train_l1": float(loss.detach()), "val_l1": vl, "sec": time.time() - t0}
            if spk is not None:
                rec.update({"lam": lam, "conf": float(conf.detach()), "spk_ce": float(ce.detach()), "log_k": math.log(len(tr))})
            log.append(rec)
            print(json.dumps(rec), flush=True)
            if step == 10000 and vl >= 0.8 * triv:
                print(f"TRIPWIRE: step 10k の検証 L1 {vl:.4f} が自明な予測 × 0.8 = {0.8 * triv:.4f} を下回らない → 中止", flush=True)
                (out / "train_log.json").write_text(json.dumps({"floors": {"untrained": v0, "trivial": triv}, "unit": ut, "log": log, "aborted": "tripwire"}, indent=1))
                return 2
        if step in (5000, 10000) or step == a.steps:
            leaks[step] = {"leak": leak_val(model, voices, val, dev), "xfer": xfer_val(model, voices, val, mu, sd, dev),
                           "xfer_ecapa": xfer_ecapa(model, voices, val, mu, sd, dev)}
            print(json.dumps({"step": step, "val30": leaks[step]}), flush=True)
            if step == 10000 and leaks[step]["xfer_ecapa"]["read_as_source"] >= a.ecapa_trip:
                print(f"ECAPA_TRIPWIRE: step 10k の検証 30 声の変換が ECAPA で源と読まれる率 {leaks[step]['xfer_ecapa']['read_as_source']:.3f} ≥ {a.ecapa_trip} → 中止", flush=True)
                torch.save({"model": model.state_dict(), "perturb": a.perturb, "adv": a.adv, "step": step}, out / "last.pt")
                (out / "train_log.json").write_text(json.dumps({"floors": {"untrained": v0, "trivial": triv}, "unit": ut, "log": log, "leak": leaks, "aborted": "ecapa_tripwire"}, indent=1))
                return 4
            if step == 10000 and leaks[step]["xfer"]["read_as_source"] >= a.leak_trip:
                print(f"LEAK_TRIPWIRE: step 10k の検証 30 声の変換が源と読まれる率 {leaks[step]['xfer']['read_as_source']:.3f} ≥ {a.leak_trip} → 中止", flush=True)
                torch.save({"model": model.state_dict(), "perturb": a.perturb, "adv": a.adv, "step": step}, out / "last.pt")
                (out / "train_log.json").write_text(json.dumps({"floors": {"untrained": v0, "trivial": triv}, "unit": ut, "log": log, "leak": leaks, "aborted": "leak_tripwire"}, indent=1))
                return 3
        if step % 10000 == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "perturb": a.perturb, "adv": a.adv, "step": step,
                        "spk": spk.state_dict() if spk is not None else None}, out / "last.pt")
    (out / "train_log.json").write_text(json.dumps({"floors": {"untrained": v0, "trivial": triv}, "unit": ut, "log": log, "leak": leaks}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
