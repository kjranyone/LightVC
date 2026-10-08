"""ZS-VC: 因果ゼロショット VC の音響モデル。入力声の因果 log-mel(内容)+ 参照音声(話者)+ 目標音域の f0 → nvoc 用の 128 帯 log-mel。

  内容符号器: 入力 mel(学習時は周波数伸縮 α・利得の摂動)→ 因果 8 層 → 192 次元(ContentVec への補助頭)→ 因果正規化
  話者符号器: 参照の mel(≤3s)→ 平均・標準偏差のプーリング → 256 次元(発話全体を見るのは参照だけ・推論では登録時に 1 回)
  復号器   : [内容, f0 特徴, レベル] → 因果 10 層。各層で因果 AdaIN(累積統計で正規化 → 話者の γ, β を注入)→ 128 帯 log-mel
  時刻規約 : 出力フレーム t は入力フレーム ≤ t だけで決まる(先読み 0)。nvoc がさらに DELAY を足す(合計アルゴリズム遅延 10ms)。
  加算 FiLM ではなく AdaIN(記憶 m2-clone-adain-grl: 加算 FiLM ではクローンが効かず、AdaIN で目標に追随した)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvoc as N

N_MEL = N.N_MEL
D_CONTENT = 192
D_SPK = 256
D_HID = 384
NORM_PRIOR = 100.0


def mel_centers() -> torch.Tensor:
    import librosa
    return torch.from_numpy(librosa.mel_frequencies(N_MEL + 2, fmin=0.0, fmax=N.FMAX, htk=False)[1:-1]).float()


def warp_mel(m: torch.Tensor, alpha: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    lc = torch.log(centers)
    src = (lc[None] - torch.log(alpha)[:, None]).contiguous()
    lo = (torch.searchsorted(lc, src, right=True) - 1).clamp(0, len(lc) - 2)
    fr = ((src - lc[lo]) / (lc[lo + 1] - lc[lo])).clamp(0, 1)[..., None]
    g0 = torch.gather(m, 1, lo[..., None].expand(-1, -1, m.shape[-1]))
    g1 = torch.gather(m, 1, (lo + 1)[..., None].expand(-1, -1, m.shape[-1]))
    return g0 * (1 - fr) + g1 * fr


def dct_mat(n: int) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64)[:, None]
    i = torch.arange(n, dtype=torch.float64)[None]
    m = torch.cos(math.pi / n * (i + 0.5) * k) * math.sqrt(2.0 / n)
    m[0] /= math.sqrt(2.0)
    return m.float()


def lifter_env(mel: torch.Tensor, D: torch.Tensor, keep: int) -> torch.Tensor:
    """log-mel [B,M,T] の周波数方向 DCT の低次 keep 個だけを残して戻す = 倍音の縞(ピッチ)を消した包絡。"""
    c = torch.einsum("km,bmt->bkt", D, mel)
    c[:, keep:] = 0
    return torch.einsum("km,bkt->bmt", D, c)


RIP_F0 = (50.0, 1200.0, 600)


def ripple_table(keep: int = 12) -> torch.Tensor:
    """f0 格子(対数等間隔 RIP_F0)ごとに、平坦な倍音列(k·f0 < 22kHz)を nvoc と同じ解析窓(Hann 1024・nfft 2048)と mel フィルタに
    通した log-mel から、DCT 低次 keep の包絡を引いた「倍音の縞」だけを返す [n_f0, N_MEL]。"""
    import numpy as np
    fb = N.mel_fb().numpy()
    w = np.hanning(N.WIN + 1)[:-1]
    big = 1 << 17
    W = np.abs(np.fft.rfft(w, n=big)) / w.sum()
    df = N.SR / big
    fj = np.arange(N.NFFT // 2 + 1) * N.SR / N.NFFT
    D = dct_mat(N_MEL).numpy()
    lo, hi, n = RIP_F0
    out = np.zeros((n, N_MEL), np.float32)
    for i, f0 in enumerate(np.exp(np.linspace(np.log(lo), np.log(hi), n))):
        k = np.arange(1, int(N.H_MAX // f0) + 1)
        d = np.abs(fj[:, None] - k[None] * f0)
        idx = np.minimum((d / df).astype(np.int64), len(W) - 1)
        S = W[idx].sum(1)
        lm = np.log(np.maximum(fb @ S, 1e-5))
        c = D @ lm
        c[keep:] = 0
        out[i] = lm - D.T @ c
    return torch.from_numpy(out)


def ripple(f0: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """f0 [B,T] → 倍音の縞 [B,N_MEL,T](無声は 0)。対数 f0 で表を線形補間。"""
    lo, hi, n = RIP_F0
    x = ((torch.log(f0.clamp(min=lo)) - math.log(lo)) / (math.log(hi) - math.log(lo)) * (n - 1)).clamp(0, n - 1 - 1e-4)
    i0 = x.floor().long()
    fr = (x - i0)[..., None]
    r = table[i0] * (1 - fr) + table[i0 + 1] * fr
    return (r * (f0 > 0).float()[..., None]).transpose(1, 2)


def causal_norm(c: torch.Tensor, prior: float = NORM_PRIOR) -> torch.Tensor:
    """[B,C,T] の各チャネルを時刻 t までの累積平均・分散で正規化(事前値 平均 0・分散 1 に重み prior フレーム)。"""
    n = torch.arange(1, c.shape[-1] + 1, device=c.device, dtype=c.dtype) + prior
    m1 = torch.cumsum(c, -1) / n
    m2 = (torch.cumsum(c * c, -1) + prior) / n
    return (c - m1) / (m2 - m1 * m1).clamp(min=1e-4).sqrt()


def f0_features(f0: torch.Tensor) -> torch.Tensor:
    """f0 [B,T] Hz(0=無声)→ [B,10,T]: log2(f0/200)・有声・sin/cos(2^k π log2(f0/200)) k=0..3。"""
    v = (f0 > 0).float()
    lf = torch.log2(f0.clamp(min=1.0) / 200.0) * v
    feats = [lf, v]
    for k in range(4):
        feats += [torch.sin(math.pi * 2 ** k * lf) * v, torch.cos(math.pi * 2 ** k * lf) * v]
    return torch.stack(feats, 1)


class CRes(nn.Module):
    def __init__(self, c: int, d: int):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, dilation=d)
        self.g = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.pw = nn.Conv1d(c, c, 1)
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(F.pad(x, (2 * self.d, 0)))
        h = F.layer_norm(h.transpose(1, 2), (h.shape[1],), self.g, self.b).transpose(1, 2)
        return x + self.pw(F.gelu(h))


class ContentEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Conv1d(N_MEL, D_HID, 1)
        self.blocks = nn.ModuleList(CRes(D_HID, d) for d in (1, 2, 4, 8, 1, 2, 4, 8))
        self.out = nn.Conv1d(D_HID, D_CONTENT, 1)
        self.aux = nn.Linear(D_CONTENT, 768)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        h = self.inp(mel)
        for b in self.blocks:
            h = b(h)
        return self.out(h)


class SpkEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(N_MEL, 256, 5, padding=2), nn.Conv1d(256, 256, 5, padding=2), nn.Conv1d(256, 256, 5, padding=2)])
        self.out = nn.Linear(512, D_SPK)

    def forward(self, mel: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = mel
        for c in self.convs:
            h = F.gelu(c(h))
        if mask is None:
            mask = torch.ones_like(h[:, :1])
        else:
            mask = mask[:, None]
        n = mask.sum(-1).clamp(min=1)
        mu = (h * mask).sum(-1) / n
        sd = (((h - mu[..., None]) ** 2 * mask).sum(-1) / n).clamp(min=1e-5).sqrt()
        return self.out(torch.cat([mu, sd], -1))


class AdaRes(nn.Module):
    """因果 conv → 因果インスタンス正規化 → 話者の γ, β を注入(AdaIN)→ GELU → 1x1 → 残差。"""

    def __init__(self, c: int, d: int):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, dilation=d)
        self.gb = nn.Linear(D_SPK, 2 * c)
        nn.init.zeros_(self.gb.weight)
        nn.init.zeros_(self.gb.bias)
        self.pw = nn.Conv1d(c, c, 1)
        self.d = d

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = causal_norm(self.conv(F.pad(x, (2 * self.d, 0))))
        g, b = self.gb(s).chunk(2, -1)
        h = h * (1 + g[..., None]) + b[..., None]
        return x + self.pw(F.gelu(h))


class Dec(nn.Module):
    def __init__(self, d_in: int = D_CONTENT):
        super().__init__()
        self.inp = nn.Conv1d(d_in + 10 + 1, D_HID, 1)
        self.blocks = nn.ModuleList(AdaRes(D_HID, d) for d in (1, 2, 4, 8, 1, 2, 4, 8, 1, 1))
        self.out = nn.Conv1d(D_HID, N_MEL, 1)

    def forward(self, c: torch.Tensor, f0: torch.Tensor, lev: torch.Tensor, s: torch.Tensor, return_h: bool = False):
        h = self.inp(torch.cat([c, f0_features(f0), lev[:, None]], 1))
        for b in self.blocks:
            h = b(h, s)
        return (self.out(h), h) if return_h else self.out(h)


class ZSVC(nn.Module):
    """cv=True: 内容 = ContentVec の予測(768・200fps)を勾配停止して復号器へ(内容側のピッチ・話者を ContentVec の分布で断つ)。"""

    def __init__(self, cv: bool = False, env_keep: int = 0, rip: bool = False):
        super().__init__()
        self.cv = cv
        self.env_keep = env_keep
        self.rip = rip
        if rip:
            self.register_buffer("rtab", ripple_table())
            self.gain = nn.Conv1d(D_HID, N_MEL, 1)
            nn.init.zeros_(self.gain.weight)
            nn.init.constant_(self.gain.bias, -3.0)
        self.register_buffer("dct", dct_mat(N_MEL), persistent=False)
        self.mel = N.CausalMel()
        self.register_buffer("centers", mel_centers())
        self.content = ContentEnc()
        self.spk = SpkEnc()
        self.dec = Dec(768 if cv else D_CONTENT)

    @staticmethod
    def level(mel: torch.Tensor) -> torch.Tensor:
        return (mel.mean(1) + 5.0) / 4.0

    def forward(self, mel_in: torch.Tensor, lev: torch.Tensor, f0: torch.Tensor, s: torch.Tensor, return_parts: bool = False):
        if self.env_keep > 0:
            mel_in = lifter_env(mel_in, self.dct, self.env_keep)
        c = self.content(mel_in)
        if self.cv:
            p = self.content.aux(c.transpose(1, 2)).transpose(1, 2)
            cn = p.detach()
        else:
            p = None
            cn = causal_norm(c)
        if self.rip:
            y, h = self.dec(cn, f0, lev, s, return_h=True)
            y = y + 1.5 * torch.sigmoid(self.gain(h)) * ripple(f0, self.rtab)
        else:
            y = self.dec(cn, f0, lev, s)
        return (y, {"content": c, "content_n": cn, "cvpred": p}) if return_parts else y


def _selftest() -> None:
    torch.manual_seed(0)
    m = ZSVC().eval()
    print("params (M): content", round(sum(p.numel() for p in m.content.parameters()) / 1e6, 2), "spk",
          round(sum(p.numel() for p in m.spk.parameters()) / 1e6, 2), "dec", round(sum(p.numel() for p in m.dec.parameters()) / 1e6, 2))
    T = 80
    mel = torch.randn(2, N_MEL, T)
    assert (warp_mel(mel, torch.ones(2), m.centers) - mel).abs().max() < 1e-5, "warp_mel α=1 は恒等"
    f0 = torch.full((2, T), 220.0)
    f0[:, 30:40] = 0
    s = torch.randn(2, D_SPK)
    lev = m.level(mel)
    with torch.no_grad():
        y = m(mel, lev, f0, s)
        for cut in (20, 50):
            mel2, f02 = mel.clone(), f0.clone()
            mel2[..., cut:] = torch.randn_like(mel2[..., cut:])
            f02[:, cut:] = 150.0
            y2 = m(mel2, m.level(mel2), f02, s)
            d = (y2 - y).abs().amax((0, 1))
            first = int((d > 1e-6).nonzero()[0])
            assert first >= cut, (cut, first)
            print(f"future invariance: edit at frame {cut} → first change at frame {first}")
    print("zsvc selftest OK")


if __name__ == "__main__":
    _selftest()
