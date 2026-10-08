"""DDSP-VC: 学習ありの DDSP によるゼロショット VC(48kHz・因果・アルゴリズム遅延 10ms)。設計 current/ddsp_vc.md。

  入力 x(48k)─因果 log-mel(左寄せ 1024・hop 240=200fps)─→ content 符号器 C ─┐
  目標の参照音声 ─ log-mel ─→ 話者符号器 S(平均+標準偏差プーリング)──── FiLM ─┤
  明示 f0(因果 YIN・目標レジスタへ写像)・レベル(入力 mel の平均)─────────────┤
                                                                               ↓
                       生成器 G(因果畳み込み 200fps)→ 調波包絡・雑音包絡(メル 80 帯の対数振幅)+ 後段条件
                                                                               ↓
  DDSP 骨格: 調波の加算合成(位相 = f0 の積分・k·f0 の包絡値で振幅)+ 雑音の帯域整形 → s0
  神経後段 P: 48k 因果畳み込み(A2 型・残差形・出力 0 初期化)→ y = s0 + P(s0, 調波, 雑音, 条件)

時刻規約: フレーム t のパラメータは入力 x[:(t+1)·HOP] だけで決まり、出力サンプル [(t+1)·HOP, (t+2)·HOP) を
フレーム t−1 → t の直線補間で描く。出力 y[n] は入力時刻 n − DELAY を表す(学習の教師は x を DELAY だけ遅らせたもの)。
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 48000
HOP = 240
DELAY = 2 * HOP
N_MEL = 80
MEL_NFFT = 1024
K_HARM = 160
F_MAX = 22000.0
NOISE_NFFT = 512


def mel_fb(n_fft: int = MEL_NFFT, n_mels: int = N_MEL, fmin: float = 40.0, fmax: float = 16000.0) -> torch.Tensor:
    import librosa
    return torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)).float()


def mel_centers(n_mels: int = N_MEL, fmin: float = 40.0, fmax: float = 16000.0) -> torch.Tensor:
    import librosa
    return torch.from_numpy(librosa.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)[1:-1]).float()


class Front(nn.Module):
    """因果 log-mel。フレーム t = x[(t+1)·HOP − MEL_NFFT : (t+1)·HOP](左 0 詰め)。T = N // HOP。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("fb", mel_fb())
        self.register_buffer("win", torch.hann_window(MEL_NFFT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1] // HOP
        xp = F.pad(x[..., :T * HOP], (MEL_NFFT - HOP, 0))
        S = torch.stft(xp, MEL_NFFT, HOP, MEL_NFFT, self.win, center=False, return_complex=True)[..., :T]
        return torch.log(torch.clamp(self.fb @ S.abs().pow(2), min=1e-8))


def warp_mel(m: torch.Tensor, alpha: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """log-mel [B,80,T] の周波数軸を α 倍に伸縮(フォルマントと倍音を同時に動かす=性別軸の入力摂動)。
    目標の帯 i(中心 f_i)に元の f_i/α の値を置く。中心は非等間隔なので searchsorted で内挿(α=1 で厳密に恒等)。"""
    lc = torch.log(centers)
    src = (lc[None] - torch.log(alpha)[:, None]).contiguous()
    lo = (torch.searchsorted(lc, src, right=True) - 1).clamp(0, len(lc) - 2)
    fr = ((src - lc[lo]) / (lc[lo + 1] - lc[lo])).clamp(0, 1)[..., None]
    g0 = torch.gather(m, 1, lo[..., None].expand(-1, -1, m.shape[-1]))
    g1 = torch.gather(m, 1, (lo + 1)[..., None].expand(-1, -1, m.shape[-1]))
    return g0 * (1 - fr) + g1 * fr


NORM_PRIOR = 100.0


def causal_norm(c: torch.Tensor, prior: float = NORM_PRIOR) -> torch.Tensor:
    """c [B,C,T] を各チャネルの累積平均・分散で正規化(因果な走行推定・事前値 平均 0/分散 1 に重み prior フレーム)。"""
    n = torch.arange(1, c.shape[-1] + 1, device=c.device, dtype=c.dtype) + prior
    m1 = torch.cumsum(c, -1) / n
    m2 = (torch.cumsum(c * c, -1) + prior) / n
    return (c - m1) / (m2 - m1 * m1).clamp(min=1e-4).sqrt()


class CRes(nn.Module):
    """因果残差ブロック(kernel 3・dilation d)+ 任意の FiLM。時間方向の統計は使わない(チャネル方向の LayerNorm のみ)。"""

    def __init__(self, ch: int, d: int, cond: int = 0):
        super().__init__()
        self.d = d
        self.conv = nn.Conv1d(ch, ch, 3, dilation=d)
        self.norm = nn.LayerNorm(ch)
        self.pw = nn.Conv1d(ch, ch, 1)
        self.film = nn.Linear(cond, 2 * ch) if cond else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        u = self.conv(F.pad(h, (2 * self.d, 0)))
        u = self.norm(u.transpose(1, 2)).transpose(1, 2)
        if self.film is not None:
            g, b = self.film(c).chunk(2, -1)
            u = u * (1 + g[..., None]) + b[..., None]
        return h + self.pw(F.gelu(u))


class ContentEnc(nn.Module):
    def __init__(self, ch: int = 256, out: int = 192):
        super().__init__()
        self.inp = nn.Conv1d(N_MEL, ch, 1)
        self.blocks = nn.ModuleList([CRes(ch, d) for d in (1, 2, 4, 8, 1, 2, 4, 8)])
        self.out = nn.Conv1d(ch, out, 1)
        self.aux = nn.Conv1d(out, 768, 1)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        h = self.inp((m + 5.0) / 4.0)
        for b in self.blocks:
            h = b(h)
        return self.out(h)


class SpkEnc(nn.Module):
    def __init__(self, ch: int = 256, out: int = 256):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv1d(N_MEL, ch, 5, padding=2), nn.GELU(), nn.Conv1d(ch, ch, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(ch, ch, 5, padding=2), nn.GELU())
        self.proj = nn.Linear(2 * ch, out)

    def forward(self, m: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.convs((m + 5.0) / 4.0)
        w = torch.ones_like(h[:, :1]) if mask is None else mask[:, None]
        n = w.sum(-1).clamp(min=1)
        mu = (h * w).sum(-1) / n
        sd = (((h - mu[..., None]) ** 2 * w).sum(-1) / n).clamp(min=1e-6).sqrt()
        return self.proj(torch.cat([mu, sd], -1))


class Gen(nn.Module):
    def __init__(self, cin: int = 192, ch: int = 384, spk: int = 256, pc: int = 32):
        super().__init__()
        self.inp = nn.Conv1d(cin + 3, ch, 1)
        self.blocks = nn.ModuleList([CRes(ch, d, spk) for d in (1, 2, 4, 8, 16, 1, 2, 4, 8, 16)])
        self.harm = nn.Conv1d(ch, N_MEL, 1)
        self.noise = nn.Conv1d(ch, N_MEL, 1)
        self.pcond = nn.Conv1d(ch, pc, 1)
        nn.init.constant_(self.harm.bias, -4.0)
        nn.init.constant_(self.noise.bias, -7.0)

    def forward(self, c, s, lf0, vuv, lev):
        h = self.inp(torch.cat([c, lf0[:, None], vuv[:, None], lev[:, None]], 1))
        for b in self.blocks:
            h = b(h, s)
        return self.harm(h), self.noise(h), self.pcond(h)


def frames_to_samples(p: torch.Tensor) -> torch.Tensor:
    """[B,C,T] フレーム値 → [B,C,T·HOP] サンプル値。サンプル [(t+1)H,(t+2)H) は p[t−1]→p[t] の直線補間(先頭 1 フレームは p[0] 保持)。"""
    B, C, T = p.shape
    prev = torch.cat([p[..., :1], p[..., :-1]], -1)
    w = (torch.arange(HOP, device=p.device, dtype=p.dtype) + 1) / HOP
    seg = prev[..., None] + (p - prev)[..., None] * w
    seg = seg.reshape(B, C, T * HOP)
    return torch.cat([seg[..., :HOP].new_zeros(B, C, HOP) + p[..., :1], seg[..., :-HOP]], -1) if T > 0 else seg


def harmonic_synth(f0s: torch.Tensor, env: torch.Tensor, centers: torch.Tensor, vuv_s: torch.Tensor) -> torch.Tensor:
    """f0s [B,N] Hz(サンプル毎・0=無声)・env [B,80,T] 対数振幅(フレーム)→ 調波の和 [B,N]。
    振幅はフレーム毎に k·f0 の包絡値を取り(対数メル周波数で内挿)、サンプルへ直線補間。位相は float64 で積算。"""
    B, N = f0s.shape
    T = env.shape[-1]
    f0f = f0s[:, HOP - 1::HOP][:, :T]
    k = torch.arange(1, K_HARM + 1, device=env.device, dtype=env.dtype)
    fk = f0f[..., None] * k
    lc = torch.log(centers)
    idx = ((torch.log(fk.clamp(min=1.0)) - lc[0]) / (lc[-1] - lc[0]) * (len(lc) - 1)).clamp(0, len(lc) - 1 - 1e-6)
    lo = idx.floor().long()
    fr = idx - lo
    e = env.transpose(1, 2)
    a = torch.gather(e, 2, lo) * (1 - fr) + torch.gather(e, 2, (lo + 1).clamp(max=len(lc) - 1)) * fr
    amp = torch.exp(a) * ((fk < F_MAX) & (f0f[..., None] > 0)).float()
    amp_s = frames_to_samples(amp.transpose(1, 2))
    ph = torch.cumsum(f0s.double() / SR, -1)
    ph = torch.remainder(ph, 1.0).float() * (2 * math.pi)
    out = torch.zeros(B, N, device=env.device, dtype=env.dtype)
    for k0 in range(0, K_HARM, 32):
        kk = k[k0:k0 + 32]
        out = out + (torch.sin(ph[:, None] * kk[:, None]) * amp_s[:, k0:k0 + 32]).sum(1)
    return out * vuv_s


def noise_synth(env: torch.Tensor, n: int, centers: torch.Tensor, gen: torch.Generator | None = None,
                wn: torch.Tensor | None = None) -> torch.Tensor:
    """雑音の帯域整形。フレーム t の包絡で窓 [(t+1)H, (t+1)H+NOISE_NFFT) の白色雑音を整形して重畳加算(因果)。"""
    B, _, T = env.shape
    dev = env.device
    win = torch.hann_window(NOISE_NFFT, device=dev)
    fbin = torch.fft.rfftfreq(NOISE_NFFT, 1 / SR).to(dev)
    lc = torch.log(centers)
    idx = ((torch.log(fbin.clamp(min=1.0)) - lc[0]) / (lc[-1] - lc[0]) * (len(lc) - 1)).clamp(0, len(lc) - 1 - 1e-6)
    lo = idx.floor().long()
    fr = idx - lo
    e = env.transpose(1, 2)
    g = torch.exp(e[..., lo] * (1 - fr) + e[..., (lo + 1).clamp(max=len(lc) - 1)] * fr)
    wn = (torch.randn(B, T, NOISE_NFFT, device=dev, generator=gen) if wn is None else wn[:, :T]) * win
    Sp = torch.fft.rfft(wn, dim=-1) * g
    fr_t = torch.fft.irfft(Sp, n=NOISE_NFFT, dim=-1) * win
    L = (T - 1) * HOP + NOISE_NFFT
    out = F.fold(fr_t.transpose(1, 2), output_size=(1, L), kernel_size=(1, NOISE_NFFT), stride=(1, HOP))[:, 0, 0]
    out = F.pad(out, (HOP, 0))
    if out.shape[-1] < n:
        out = F.pad(out, (0, n - out.shape[-1]))
    return out[:, :n] / 1.5


class Post(nn.Module):
    def __init__(self, ch: int = 24, pc: int = 32):
        super().__init__()
        self.inp = nn.Conv1d(3 + pc, ch, 1)
        self.dils = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        self.convs = nn.ModuleList([nn.Conv1d(ch, ch, 3, dilation=d) for d in self.dils])
        self.mix = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in self.dils])
        self.out = nn.Conv1d(ch, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, s0, harm, noise, pc_s):
        h = self.inp(torch.cat([s0[:, None], harm[:, None], noise[:, None], pc_s], 1))
        for c, m, d in zip(self.convs, self.mix, self.dils):
            h = h + m(F.leaky_relu(c(F.pad(h, (2 * d, 0))), 0.1))
        return s0 + self.out(F.leaky_relu(h, 0.1))[:, 0]


class DDSPVC(nn.Module):
    def __init__(self, norm: bool = False):
        super().__init__()
        self.norm = norm
        self.front = Front()
        self.register_buffer("centers", mel_centers())
        self.content = ContentEnc()
        self.spk = SpkEnc()
        self.gen = Gen()
        self.post = Post()

    def forward(self, x_mel_in: torch.Tensor, lev: torch.Tensor, f0: torch.Tensor, spk_emb: torch.Tensor,
                n: int, gen: torch.Generator | None = None, return_parts: bool = False, noise: torch.Tensor | None = None):
        """x_mel_in [B,80,T](content 用・摂動済みでもよい)・lev [B,T]・f0 [B,T](Hz・0=無声・フレーム t は入力 ≤(t+1)H)
        ・spk_emb [B,256] → y [B,n](n = T·HOP)。noise [B,T,512] を与えると雑音枝はその白色雑音を使う(Rust parity 用)。"""
        c = self.content(x_mel_in)
        cn = causal_norm(c) if self.norm else c
        vuv = (f0 > 0).float()
        lf0 = torch.where(f0 > 0, torch.log(f0.clamp(min=1.0) / 200.0), torch.zeros_like(f0))
        he, ne, pc = self.gen(cn, spk_emb, lf0, vuv, lev)
        f0s = frames_to_samples(f0[:, None])[:, 0]
        vuv_s = frames_to_samples(vuv[:, None])[:, 0]
        f0s = torch.where(vuv_s > 0.5, f0s, torch.zeros_like(f0s))
        harm = harmonic_synth(f0s, he, self.centers, vuv_s)[:, :n]
        noise = noise_synth(ne, n, self.centers, gen, wn=noise)
        s0 = harm + noise
        y = self.post(s0, harm, noise, frames_to_samples(pc)[..., :n])
        if return_parts:
            return y, {"harm": harm, "noise": noise, "s0": s0, "content": c, "content_n": cn}
        return y

    def level(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel.mean(1) + 5.0) / 4.0


def _selftest() -> None:
    torch.manual_seed(0)
    c = mel_centers()
    mm = torch.randn(2, N_MEL, 7)
    print("warp_mel α=1 恒等 max|Δ|", float((warp_mel(mm, torch.ones(2), c) - mm).abs().max()))
    ramp = torch.log(c)[None, :, None].expand(1, -1, 3).clone()
    w = warp_mel(ramp, torch.tensor([1.2]), c)
    inner = (c > 100) & (c < 12000)
    print("warp_mel α=1.2 で log f の傾きが −log1.2 だけずれる: max|Δ−(−ln1.2)|",
          float((w - ramp)[0, inner, 0].sub(-np.log(1.2)).abs().max()))
    m = DDSPVC()
    print("params M", round(sum(p.numel() for p in m.parameters()) / 1e6, 3),
          {k: round(sum(p.numel() for p in getattr(m, k).parameters()) / 1e6, 3) for k in ("content", "spk", "gen", "post")})
    n = 48000
    x = torch.randn(1, n) * 0.1
    mel = m.front(x)
    T = mel.shape[-1]
    f0 = torch.full((1, T), 220.0)
    f0[:, :20] = 0
    s = m.spk(m.front(torch.randn(1, 96000) * 0.1))
    g = torch.Generator().manual_seed(1)
    y = m(mel, m.level(mel), f0, s, n, gen=g)
    print("out", tuple(y.shape), "finite", bool(torch.isfinite(y).all()), "rms", float(y.pow(2).mean().sqrt()))
    with torch.no_grad():
        for p in m.post.out.parameters():
            p.normal_(0, 0.1)
    worst = -1
    for c in (20000, 30000, 40000):
        x2 = x.clone()
        x2[:, c:] = torch.randn(1, n - c) * 0.1
        mel2 = m.front(x2)
        f2 = f0.clone()
        f2[:, c // HOP:] = 300.0
        y1 = m(mel, m.level(mel), f0, s, n, gen=torch.Generator().manual_seed(1))
        y2 = m(mel2, m.level(mel2), f2, s, n, gen=torch.Generator().manual_seed(1))
        d = torch.nonzero((y1 - y2).abs()[0] > 1e-7)
        first = int(d[0]) if len(d) else None
        print(f"edit at {c}: first changed output sample {first}", "(先読み", (c - first) if first is not None else None, "samples)")
        if first is not None:
            worst = max(worst, c - first)
    print("worst lookahead samples", worst, "→", "PASS" if worst <= 0 else "FAIL")


if __name__ == "__main__":
    _selftest()
