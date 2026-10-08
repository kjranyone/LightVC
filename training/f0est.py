"""F1: 因果な f0・有声推定器(converter.md §3b)。

特徴(200fps・フレーム t は入力の (t+1)·240 サンプル(48kHz)までだけで決まる):
  CMNDF(48kHz のまま・窓 W = 480(10ms・速い)と 1200(25ms・低い f0 に強い)の 2 本): フレーム t の区間 x[(t+1)·240 − W − TMAX, (t+1)·240) を
         時間反転し、最新の W を基準窓として遅れ τ ≤ TMAX = 960(50Hz)の YIN の累積平均正規化差分を計算。遅れ 20〜960(2400〜50Hz)を対数に 160 点へ補間(rev2: 1kHz 超の叫び・裏返りまで・current/f0_range.md。rev1 は 48〜960・128 点)。
         段差の f0 で、閾値 0.2 を下回る谷が新しい音高へ移るのは W 1200 で教師の +4 フレーム・W 480 で +2 以内(selftest)。
         基準窓のエネルギーが床未満なら CMNDF = 1(無音を周期的と見せない)。
         (rev0 は区間の最も古い窓を基準にしていて教師より 30〜40ms 遅れた: F1 起動前レビュー重大 1)
  log エネルギー(各 W の最新の平均二乗・床 1e-10)/10。
  因果 log-mel 128(nvoc の規約・窓は (t+1)·240 で終わる)を (mel − log 1e-5)/10。
ネット: 因果な拡張畳み込み(k 3・dil 1,2,4,8 × 2・192ch・残差・受容野 62 フレーム)→ 有声のロジット + log f0 の 20 cent 刻みの分類(50〜2200Hz・329 bin・rev1 は 50〜1100Hz・269 bin)。
f0 = 最大の bin の ±4 bin の重み付き平均(log 領域)。有声 = p > 0.5(固定)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvoc as N

WS = (480, 1200)
TMAX = 960
CTX48 = max(WS) + TMAX
RF = 62
N_LAG = 160
LAG_MIN = 20.0
F_LO, F_HI = 50.0, 2200.0
CENTS = 20.0
N_BIN = int(round(1200 * math.log2(F_HI / F_LO) / CENTS)) + 1
D_IN = len(WS) * (N_LAG + 1) + N.N_MEL
E_FLOOR = 1e-10


def bin_hz() -> torch.Tensor:
    return F_LO * 2.0 ** (torch.arange(N_BIN, dtype=torch.float64) * CENTS / 1200)


def lag_grid(dev=None) -> torch.Tensor:
    return torch.exp(torch.linspace(math.log(LAG_MIN), math.log(TMAX - 1), N_LAG, device=dev, dtype=torch.float64))


def cmndf_feats(x48: torch.Tensor, T: int) -> torch.Tensor:
    """x48 [B, CTX48 + T·240](先頭 CTX48 は左文脈)→ [B, len(WS)·(N_LAG + 1), T]。フレーム t の区間 = x48[(t+1)·240, (t+1)·240 + CTX48)(左文脈込みの添字)。"""
    L = CTX48
    if x48.shape[-1] < L + T * N.HOP:
        raise ValueError("x48 が短い")
    s = x48.double().unfold(-1, L, N.HOP)[:, 1:T + 1].flip(-1)
    nfft = 4096
    Fs = torch.fft.rfft(s, n=nfft)
    cs = torch.cat([torch.zeros_like(s[..., :1]), torch.cumsum(s * s, -1)], -1)
    tau = torch.arange(TMAX + 1, device=x48.device)
    lags = lag_grid(x48.device)
    i0 = lags.floor().long().clamp(max=TMAX - 1)
    w = lags - i0
    outs = []
    for W in WS:
        r = torch.fft.irfft(torch.conj(torch.fft.rfft(s[..., :W], n=nfft)) * Fs, n=nfft)[..., :TMAX + 1]
        e0 = cs[..., W:W + 1]
        et = cs[..., W + tau] - cs[..., tau]
        d = (e0 + et - 2 * r).clamp(min=0)
        d[..., 0] = 0
        cm = torch.cumsum(d[..., 1:], -1) / torch.arange(1, TMAX + 1, device=x48.device, dtype=torch.float64)
        dn = torch.ones_like(d)
        dn[..., 1:] = d[..., 1:] / cm.clamp(min=1e-12)
        dn = torch.where((e0 / W) < E_FLOOR, torch.ones_like(dn), dn)
        feat = dn[..., i0] * (1 - w) + dn[..., i0 + 1] * w
        en = torch.log((e0[..., 0] / W).clamp(min=E_FLOOR)) / 10
        outs.append(torch.cat([feat, en[..., None]], -1))
    return torch.cat(outs, -1).transpose(1, 2).float()


class Front(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = N.CausalMel()

    def forward(self, x48: torch.Tensor, T: int) -> torch.Tensor:
        """x48 [B, CTX48 + T·240](左文脈つき)→ [B, D_IN, T]。"""
        seg = x48[:, CTX48 - (N.WIN - N.HOP):]
        mel = N.NVoc.mel_ctx(self, seg)[..., :T]
        return torch.cat([cmndf_feats(x48, T), (mel - math.log(1e-5)) / 10], 1)


class CBlock(nn.Module):
    def __init__(self, ch: int, d: int):
        super().__init__()
        self.d = d
        self.c1 = nn.Conv1d(ch, ch, 3, dilation=d)
        self.c2 = nn.Conv1d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(F.gelu(self.c1(F.pad(F.gelu(x), (2 * self.d, 0)))))


class F0Est(nn.Module):
    def __init__(self, ch: int = 192, dils: tuple = (1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.cfg = {"ch": ch, "dils": list(dils)}
        self.inp = nn.Conv1d(D_IN, ch, 3)
        self.blocks = nn.ModuleList(CBlock(ch, d) for d in dils)
        self.out = nn.Conv1d(ch, 1 + N_BIN, 1)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.inp(F.pad(feat, (2, 0)))
        for b in self.blocks:
            x = b(x)
        o = self.out(F.gelu(x))
        return o[:, 0], o[:, 1:]


def decode(v_logit: torch.Tensor, p_logit: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """→ f0 [B, T](Hz・無声 0)。"""
    hz = torch.log(bin_hz().to(p_logit.device)).float()
    pr = p_logit.softmax(1)
    k = pr.argmax(1, keepdim=True)
    idx = (k + torch.arange(-4, 5, device=p_logit.device)[None, :, None]).clamp(0, N_BIN - 1)
    w = torch.gather(pr, 1, idx)
    lf = (w * hz[idx]).sum(1) / w.sum(1).clamp(min=1e-9)
    return torch.where(torch.sigmoid(v_logit) > thr, lf.exp(), torch.zeros_like(lf))


def target_bins(f0: torch.Tensor, sigma_cents: float = 25.0) -> torch.Tensor:
    """f0 [B, T] → ぼかした分類の正解 [B, N_BIN, T](無声は 0)。"""
    c = 1200 * torch.log2(f0.clamp(min=1.0) / F_LO)
    grid = torch.arange(N_BIN, device=f0.device, dtype=torch.float32) * CENTS
    g = torch.exp(-0.5 * ((grid[None, :, None] - c[:, None, :]) / sigma_cents) ** 2)
    g = g / g.sum(1, keepdim=True).clamp(min=1e-9)
    return torch.where((f0 > 0)[:, None, :], g, torch.zeros_like(g))


def first_dip_hz(feat: torch.Tensor, thr: float = 0.2) -> torch.Tensor:
    """CMNDF の特徴 [N_LAG, T] から、閾値を下回る最初の谷の音高(Hz・無ければ 0)。整列の単体試験用。"""
    lags = lag_grid().float()
    out = torch.zeros(feat.shape[-1])
    for t in range(feat.shape[-1]):
        c = feat[:N_LAG, t]
        idx = (c < thr).nonzero()
        if len(idx) == 0:
            continue
        j = int(idx[0])
        while j + 1 < N_LAG and c[j + 1] < c[j]:
            j += 1
        out[t] = N.SR / float(lags[j])
    return out


def _selftest() -> None:
    torch.manual_seed(0)
    T = 200
    n = CTX48 + T * N.HOP
    t48 = torch.arange(n, dtype=torch.float64) / N.SR

    def harm(f0, ph=None):
        f = torch.full_like(t48, f0) if ph is None else ph
        p = 2 * math.pi * torch.cumsum(f, 0) / N.SR
        return sum(torch.sin(k * p) / k for k in range(1, 20)) * 0.1

    x = harm(180.0)[None].float()
    fe = Front()(x, T)
    assert fe.shape == (1, D_IN, T), fe.shape
    for wi, W in enumerate(WS):
        hz = first_dip_hz(fe[0, wi * (N_LAG + 1):])[20:]
        print(f"一定 180Hz・W {W}: CMNDF の最初の谷 = {float(hz.median()):.1f}Hz")
        assert abs(float(hz.median()) - 180.0) < 6.0
    k = 100
    f = torch.where(torch.arange(n) < CTX48 + k * N.HOP, torch.tensor(150.0, dtype=torch.float64), torch.tensor(250.0, dtype=torch.float64))
    xs = harm(0.0, f)[None].float()
    fs = Front()(xs, T)[0]
    sws = []
    for wi, W in enumerate(WS):
        hz = first_dip_hz(fs[wi * (N_LAG + 1):])
        sw = int(((hz - 250.0).abs() < 15).nonzero()[0])
        sws.append(sw)
        print(f"整列・W {W}: f0 の段差はフレーム {k} の中心(教師の切り替わり = フレーム {k})→ 特徴の切り替わり = フレーム {sw}(+{sw - k})")
    assert k <= sws[0] <= k + 2 and sws[1] <= k + 5, sws
    z = torch.zeros(1, n)
    fz = Front()(z, T)
    for wi in range(len(WS)):
        assert float((fz[0, wi * (N_LAG + 1):wi * (N_LAG + 1) + N_LAG] - 1).abs().max()) == 0.0
    print("無音: CMNDF = 1(周期的に見せない)")
    m = F0Est().eval()
    print("params (M)", round(sum(p.numel() for p in m.parameters()) / 1e6, 3), "bins", N_BIN)
    with torch.no_grad():
        v, p = m(fe)
        for cut in (77, 151):
            x2 = x.clone()
            x2[:, CTX48 + cut * N.HOP:] = torch.randn_like(x2[:, CTX48 + cut * N.HOP:]) * 0.1
            v2, p2 = m(Front()(x2, T))
            dd = (p2 - p).abs().amax(1)[0] + (v2 - v).abs()[0]
            first = int((dd > 0).nonzero()[0])
            assert first >= cut, (cut, first)
            print(f"未来不変性: サンプル {cut * N.HOP} 以降の書き換え → 最初に変わるフレーム {first}(フレーム t は (t+1)·240 まで)")
    tb = target_bins(torch.tensor([[220.0, 0.0]]))
    assert abs(float(bin_hz()[tb[0, :, 0].argmax()]) - 220.0) < 220 * (2 ** (10 / 1200) - 1) + 1e-6
    print("f0est selftest OK")


if __name__ == "__main__":
    _selftest()
