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
--xview(段 P24 step b・paper_snv §5 M30)= 視点を 1 つ足す: 同じ発話の擬似の操作者の版(data/snv_p_feat_x16 = WORLD の高さと声道の長さの下げ + 録音の鎖 v2・フレームは元の音と揃っている)に
同じ伸縮・傾きの乱しを掛けた入力の文脈 c から、加工しない入力の未来 z を当てる InfoNCE を、元と乱しの 2 つと平均する。当てる先は加工しない音のまま(加工は入力側だけ)。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p24_xcpc --aug --xview --prereg results/snv_p24/prereg.yaml
--rview(段 P25 step b・paper_snv §5 M31)= さらに視点を 1 つ: 同じ発話の「本物の非音声の音源を足した版」(data/snv_p_feat_rec2・M18 の入力 = 本物の雑音と歌の無い音楽・残響・音量。版 k0 / k1 を試料ごとに半々)に
同じ伸縮・傾きの乱しを掛けた入力の文脈 c から、加工しない入力の未来 z を当てる InfoNCE を足す(--xview と組む = 4 視点の平均)。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p25_xrcpc --aug --xview --rview --prereg results/snv_p25/prereg.yaml
--dview(段 P30・paper_snv §5 M33)= 視点を 1 つ足す: 段 D の、同じ文を別の声が読んだ発話の組(data/snv_p_feat_dalign = snv_p30_align.py の DTW の時刻の対応)。
声 A の発話(伸縮・傾きの乱しを掛ける)の文脈 c から、声 B の発話の加工しない z の「対応する位置の未来」を当てる InfoNCE(負例は B の同じ窓の別の時刻・正例から D_GAP フレーム以上離す)を、元・乱し・擬似の操作者の 3 つと平均する。
当てる先は B の加工しない音の z(音響の回帰の対にしない・対応は正例の位置にだけ使う)。--xview と組む。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p30_dcpc --aug --xview --dview --prereg results/snv_p30/prereg.yaml
--rdir <dir>(段 P57 step b・paper_snv §5 M42)= --rview の版の置き場を替える(既定 data/snv_p_feat_rec2)。data/snv_p_feat_fr57 = 族の混ぜを研究の形の前段に通した特徴(snv_p57_feat.py・k0 = Irodori の声・k1 = 擬似の操作者の版)。
--dview と --rview を組むと 5 視点(元・乱し・擬似の操作者・別の声・前段を通った入力)の平均。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p57_rcpc --aug --xview --dview --rview --rdir data/snv_p_feat_fr57 --prereg results/snv_p57/prereg.yaml
--vsel f / q(段 P70・paper_snv 系 2x・診断)= 学習の声を選ぶ: f = 男声(em*)を全て除く / q = 男声と同じ数の女声を乱数(種 70)で除く(数を揃えた対照)。全ての視点(元・擬似の操作者・別の声の組)に同じ選びが掛かる。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag diag_snv_p70_fcpc --aug --xview --dview --vsel f --prereg results/snv_p70/prereg.yaml
--jview g(段 P86・paper_snv 系 2x (xviii))= 視点を 1 つ足す: 伸縮・傾きの乱しを掛けた入力の包絡(c0〜c24)に、フレームと次元ごとに独立な正規の雑音を足す。
係数ごとの標準偏差 = results/snv_p85/jitter_shape.json の s(本物の男声の崩れる源の、包絡のフレームごとの余分な揺れの見積もり)× 発話ごとの倍率 U(0, g)。
その文脈 c から、加工しない入力の未来 z を当てる InfoNCE を他の視点と平均する(--xview と組む)。当てる先は加工しない音のまま。
--seed n = 重みの初めの値・束の並び・乱しの乱数の種(渡さないと、これまでどおり束と乱しは 0・重みは種なし)。
--ctc λ(段 P81・paper_snv 系 2x (xvii))= 補助の損失: 普通の発話の全長(伸縮・傾きの乱しを掛ける)の文脈 c を 4 フレームずつ平均して(50 fps)2 層の頭を置き、台詞の音素の列(results/snv_p81/phones.json = 合成の台詞の文を辞書の G2P に通したもの)を CTC で当てる(1 歩に N_CTC 本)。
音素の無い発声には掛けない。頭は符号に入れない(演奏は c のまま・推論で使わない)。
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag diag_snv_p81_ccpc --aug --xview --dview --ctc 0.5 --init results/snv_p30_dcpc/last.pt --steps 10000 --prereg results/snv_p81/prereg.yaml
    cd training && uv run --no-sync python train_snv_p2_cpc.py --tag snv_p10_wcpc --aug --white 64 --init results/snv_p2b_cpc/last.pt --steps 20000 --prereg results/snv_p10/prereg.yaml
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

import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
XFEAT = ROOT / os.environ.get("SNV_XFEAT", "data/snv_p_feat_x16")
RFEAT = ROOT / "data/snv_p_feat_rec2"
D_IN = 30
CH = 256
KS = (1, 2, 4, 8, 12, 16, 24)
N_NEG = 16
CROP = 512
MIN_GAP = 5
ENR = 600
N_PAIR = 4
N_CTC = 8
CTC_POOL = 4
DALIGN = ROOT / "data/snv_p_feat_dalign/align.npz"
D_BS = 16
D_LB = 768
D_GAP = 12
D_PAD = 48
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


def info_nce_x(zb: torch.Tensor, ca: torch.Tensor, ma: torch.Tensor, nb: torch.Tensor, jb: torch.Tensor, W: nn.ModuleList, g: torch.Generator) -> tuple[torch.Tensor, list[float]]:
    B, C, L = ca.shape
    zt, ct = zb.transpose(1, 2), ca.transpose(1, 2)
    losses, accs = [], []
    for k, Wk in zip(KS, W):
        n = L - k
        sc = Wk(ct[:, :n]) @ zt.transpose(1, 2) / math.sqrt(C)
        tp = jb[:, k:]
        valid = ((ma[:, :n] * ma[:, k:]) > 0) & (tp >= 0) & (tp < nb[:, None])
        span = (nb - 2 * D_GAP).clamp(min=1)[:, None, None]
        off = D_GAP + (torch.rand(B, n, N_NEG, generator=g, device=ca.device) * span).long().clamp(max=int(zb.shape[2]))
        pos = tp.clamp(min=0)[..., None]
        idx = torch.cat([pos, (pos + off) % nb[:, None, None]], 2)
        lv = torch.gather(sc, 2, idx)[valid]
        losses.append(F.cross_entropy(lv, torch.zeros(len(lv), dtype=torch.long, device=ca.device)))
        accs.append(float((lv.argmax(1) == 0).float().mean()))
    return torch.stack(losses).mean(), accs


def load_d(names: list[str]) -> tuple[dict, list[tuple], dict]:
    tr = set(names)
    al = np.load(DALIGN)
    keys = [tuple(k.split("|")) for k in al.files]
    keys = [k for k in keys if k[0] in tr and k[2] in tr]
    need = sorted({(k[0], k[1]) for k in keys} | {(k[2], k[3]) for k in keys})
    us = {}
    for v, s in need:
        u = T.load_utt(T.FEAT / "d" / v / f"{s}.npz")
        us[(v, s)] = np.concatenate([u["y"], u["v"][None]], 0)
    return us, keys, {k: al["|".join(k)].astype(np.int64) for k in keys}


def d_batch(us: dict, keys: list[tuple], al: dict, bs: int, rng: random.Random) -> tuple:
    XA = np.zeros((bs, T.D_Y + 1, CROP), np.float32)
    XB = np.zeros((bs, T.D_Y + 1, D_LB), np.float32)
    MA = np.zeros((bs, CROP), np.float32)
    JB = np.full((bs, CROP), -1, np.int64)
    NB = np.ones(bs, np.int64)
    for i in range(bs):
        k = keys[rng.randrange(len(keys))]
        a, b, j = us[(k[0], k[1])], us[(k[2], k[3])], al[k]
        L = min(a.shape[1], len(j))
        s = rng.randint(0, max(0, L - CROP))
        n = min(CROP, L - s)
        b0 = max(0, int(j[s]) - D_PAD)
        nb = min(D_LB, b.shape[1] - b0)
        XA[i, :, :n], MA[i, :n] = a[:, s:s + n], 1.0
        XB[i, :, :nb], NB[i] = b[:, b0:b0 + nb], nb
        jj = j[s:s + n] - b0
        JB[i, :n] = np.where((jj >= 0) & (jj < nb), jj, -1)
    return torch.from_numpy(XA), torch.from_numpy(MA), torch.from_numpy(XB), torch.from_numpy(NB), torch.from_numpy(JB)


def load(names: list[str], sig: np.ndarray | None = None) -> tuple[list[np.ndarray], np.ndarray, list[int]]:
    us, lab = [], []
    for i, v in enumerate(names):
        for p in sorted((T.FEAT / "e1" / v).glob("t*.npz")) + sorted((T.FEAT / "e1nv" / v).glob("*.npz")):
            us.append(T.load_utt(p)), lab.append(i)
    if sig is None:
        sig = corpus_scale(us[::7])
    return [np.concatenate([u["y"], u["v"][None]], 0) for u in us], sig, lab


def batch(ys: list[np.ndarray], bs: int, rng: random.Random, lab: list[int] | None = None, js: list | None = None, xs2: list[np.ndarray] | None = None, xs3: tuple | None = None) -> tuple:
    X = np.zeros((bs, T.D_Y + 1, CROP), np.float32)
    X2 = np.zeros((bs, T.D_Y + 1, CROP), np.float32)
    X3 = np.zeros((bs, T.D_Y + 1, CROP), np.float32)
    M = np.zeros((bs, CROP), np.float32)
    S = np.zeros(bs, np.int64)
    for i in range(bs):
        j = rng.randrange(len(ys))
        if js is not None:
            js.append(j)
        x = ys[j]
        S[i] = lab[j] if lab is not None else 0
        x3 = xs3[rng.randrange(len(xs3))][j] if xs3 is not None else None
        L = min([x.shape[1]] + ([xs2[j].shape[1]] if xs2 is not None else []) + ([x3.shape[1]] if x3 is not None else []))
        s = rng.randint(0, max(0, L - CROP))
        n = min(CROP, L - s)
        X[i, :, :n], M[i, :n] = x[:, s:s + n], 1.0
        if xs2 is not None:
            X2[i, :, :n] = xs2[j][:, s:s + n]
        if x3 is not None:
            X3[i, :, :n] = x3[:, s:s + n]
    out = (torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(S))
    return out + ((torch.from_numpy(X2),) if xs2 is not None else ()) + ((torch.from_numpy(X3),) if xs3 is not None else ())


def load_x(names: list[str], root: Path = XFEAT, suffix: str = "") -> list[np.ndarray]:
    us = []
    for v in names:
        for p in sorted((T.FEAT / "e1" / v).glob("t*.npz")) + sorted((T.FEAT / "e1nv" / v).glob("*.npz")):
            us.append(T.load_utt(root / p.parent.parent.name / v / f"{p.stem}{suffix}.npz"))
    return [np.concatenate([u["y"], u["v"][None]], 0) for u in us]


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
    ap.add_argument("--xview", action="store_true")
    ap.add_argument("--rview", action="store_true")
    ap.add_argument("--dview", action="store_true")
    ap.add_argument("--rdir", default="data/snv_p_feat_rec2")
    ap.add_argument("--vsel", default="", choices=("", "f", "q"))
    ap.add_argument("--ctc", type=float, default=0.0)
    ap.add_argument("--jview", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--jshape", default="results/snv_p85/jitter_shape.json")
    ap.add_argument("--phones", default="results/snv_p81/phones.json")
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (ROOT / a.prereg).exists():
        print(f"{a.prereg} が無い: 起動しない", flush=True)
        return 1
    print("prereg", ROOT / a.prereg, "aug", a.aug, "inv", a.inv, "alpha", (a.amin, a.amax), "eq", a.eq, "spk", a.spk, "xview", a.xview, flush=True)
    assert not a.xview or (a.aug and not a.white and a.inv == 0), "--xview は --aug と組む(--white・--inv とは組まない)"
    assert not a.rview or a.xview, "--rview は --xview と組む"
    assert not a.dview or a.xview, "--dview は --xview と組む"
    unit_test()
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    tr = split["train"][:40] if a.smoke else split["train"][:: a.subset]
    if a.vsel:
        males = [v for v in tr if v.startswith("em")]
        drop = set(males) if a.vsel == "f" else set(random.Random(70).sample([v for v in tr if v.startswith("ef")], len(males)))
        tr = [v for v in tr if v not in drop]
        print("vsel", a.vsel, "除いた声", len(drop), "残り 女 / 男", sum(v.startswith("ef") for v in tr), sum(v.startswith("em") for v in tr), flush=True)
    xs, sig, lab = load(tr)
    xx = load_x(tr) if a.xview else None
    if xx is not None:
        assert len(xx) == len(xs) and max(abs(p.shape[1] - q.shape[1]) for p, q in zip(xs, xx)) <= 2, "擬似の操作者の版の発話が元と揃っていない"
        print("xview utts", len(xx), flush=True)
    xr = tuple(load_x(tr, ROOT / a.rdir, f"_k{k}") for k in (0, 1)) if a.rview else None
    if xr is not None:
        assert all(len(q) == len(xs) and max(abs(p.shape[1] - r.shape[1]) for p, r in zip(xs, q)) <= 2 for q in xr), "本物の非音声の音源を足した版の発話が元と揃っていない"
        print("rview", a.rdir, "utts", len(xr[0]), "× 版 2", flush=True)
    dd = load_d(tr) if a.dview else None
    if dd is not None:
        print("dview utts", len(dd[0]), "組", len(dd[1]), flush=True)
    norm_of: dict = {}
    if a.white or a.pair > 0 or a.ctc > 0:
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
    cidx: list = []
    if a.ctc > 0:
        ph = json.loads((ROOT / a.phones).read_text())
        for i, v in enumerate(tr):
            for t, y in ph["labels"][v].items():
                if int(t[1:]) < len(norm_of[i]) and 0 < len(y) * 2 * CTC_POOL <= xs[norm_of[i][int(t[1:])]].shape[1]:
                    cidx.append((norm_of[i][int(t[1:])], y))
        chead = nn.Sequential(nn.Linear(CH, CH), nn.ReLU(), nn.Linear(CH, len(ph["vocab"]) + 1)).to(dev)
        print("ctc utts", len(cidx), "vocab", len(ph["vocab"]), flush=True)
    sig_t = torch.from_numpy(sig).to(dev)
    Ms, tv = torch.from_numpy(warp_mats(a.amin, a.amax)).to(dev), torch.from_numpy(tilt_vec()).to(dev)
    eqb = torch.from_numpy(eq_basis()).to(dev)
    jit = torch.tensor(json.loads((ROOT / a.jshape).read_text())["s"], dtype=torch.float32, device=dev) if a.jview > 0 else None
    accj: list = []
    if a.seed is not None:
        torch.manual_seed(a.seed)
    m = CPC().to(dev)
    if a.init:
        st0 = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)
        m.load_state_dict(st0["cpc"])
        sig = np.asarray(st0["sig"], np.float32)
        sig_t = torch.from_numpy(sig).to(dev)
        print("init", a.init, "step", st0.get("step"), flush=True)
    wh = WHead(a.white).to(dev) if a.white else None
    adv = SpkAdv(len(tr)).to(dev)
    params = list(m.parameters()) + (list(adv.parameters()) if a.spk > 0 else []) + (list(wh.parameters()) if wh is not None else []) + (list(chead.parameters()) if a.ctc > 0 else [])
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)
    steps = 200 if a.smoke else a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    rng, g = random.Random(a.seed or 0), torch.Generator(device=dev).manual_seed(a.seed or 0)
    np.save(out / "sig.npy", sig)
    t0 = time.time()
    for step in range(1, steps + 1):
        jb: list = []
        bt = batch(xs, a.bs, rng, lab, jb, xx, xr)
        X, M, S = bt[0].to(dev), bt[1].to(dev), bt[2].to(dev)
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
            if a.xview:
                _, cx = m.encode(x_torch(augment(bt[3].to(dev), Ms, tv, g), sig_t))
                lx, accx = info_nce(z, cx, M, m.W, g)
                views = [loss, la, lx]
                if a.dview:
                    XA, MA, XB, NB, JB = (q.to(dev) for q in d_batch(*dd, D_BS, rng))
                    zb, _ = m.encode(x_torch(XB, sig_t))
                    _, cd = m.encode(x_torch(augment(XA, Ms, tv, g), sig_t))
                    ld, accd = info_nce_x(zb, cd, MA, NB, JB, m.W, g)
                    views.append(ld)
                if a.rview:
                    _, cr = m.encode(x_torch(augment(bt[4].to(dev), Ms, tv, g), sig_t))
                    lr_, accr = info_nce(z, cr, M, m.W, g)
                    views.append(lr_)
                if a.jview > 0:
                    XJ = augment(X, Ms, tv, g).clone()
                    XJ[:, :25] = XJ[:, :25] + jit[None, :, None] * (a.jview * torch.rand(X.shape[0], 1, 1, generator=g, device=dev)) * torch.randn(X.shape[0], 25, X.shape[2], generator=g, device=dev)
                    _, cj = m.encode(x_torch(XJ, sig_t))
                    lj, accj = info_nce(z, cj, M, m.W, g)
                    views.append(lj)
                loss = sum(views) / len(views)
            else:
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
        lc = torch.zeros((), device=dev)
        if a.ctc > 0:
            sel = rng.sample(cidx, N_CTC)
            Lc = max(xs[j].shape[1] for j, _ in sel)
            XC = torch.zeros(len(sel), T.D_Y + 1, Lc)
            for i, (j, _) in enumerate(sel):
                XC[i, :, :xs[j].shape[1]] = torch.from_numpy(xs[j])
            _, cc = m.encode(x_torch(augment(XC.to(dev), Ms, tv, g), sig_t))
            lp_ = F.log_softmax(chead(F.avg_pool1d(cc, CTC_POOL).transpose(1, 2)), -1).transpose(0, 1)
            lc = F.ctc_loss(lp_, torch.cat([torch.tensor(y) for _, y in sel]).to(dev), torch.tensor([xs[j].shape[1] // CTC_POOL for j, _ in sel]), torch.tensor([len(y) for _, y in sel]), blank=0, zero_infinity=True)
            loss = loss + a.ctc * lc
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
            print(f"step {step} loss {float(loss):.3f} inv {float(li):.3f} pair {float(lp):.4f} ctc {float(lc):.3f} spk_acc {sacc:.3f} acc " + " ".join(f"k{k}:{x:.2f}" for k, x in zip(KS, accs)) + (" xacc " + " ".join(f"k{k}:{x:.2f}" for k, x in zip(KS, accx)) if a.xview else "") + (" racc " + " ".join(f"k{k}:{x:.2f}" for k, x in zip(KS, accr)) if a.rview else "") + (" dacc " + " ".join(f"k{k}:{x:.2f}" for k, x in zip(KS, accd)) if a.dview else "") + (" jacc " + " ".join(f"k{k}:{x:.2f}" for k, x in zip(KS, accj)) if a.jview > 0 else "")
                  + f" {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 5000 == 0 or step == steps:
            extra = ({"adv": adv.state_dict()} if a.spk > 0 else {}) | ({"white": wh.state_dict(), "dw": a.white} if wh is not None else {}) | ({"ctc": chead.state_dict(), "ctc_w": a.ctc} if a.ctc > 0 else {})
            torch.save({"cpc": m.state_dict(), "sig": sig, "step": step} | extra, out / "last.pt")
            if step == steps // 2 or a.xview and step % 10000 == 0 and step < steps:
                torch.save({"cpc": m.state_dict(), "sig": sig, "step": step} | extra, out / f"step{step // 1000}k.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
