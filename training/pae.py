"""ピッチ適応の包絡と周期性(出力部の条件・変換器の正解・登録の表の分析)。ストリームの経路には入らない = 中心揃えの窓で可。

包絡: WORLD CheapTrick(f0 適応の窓 + 幅 f0 の平滑化 + 補償リフタ・修正 BSD)のパワー包絡 → 振幅 → 128 帯 mel → log(床 1e-5)→ DCT c0..c24。
  因果 mel の DCT24 は f0 300〜450Hz で +4 半音に低域が 19〜28dB 動く(倍音の縞)= ピッチを運ぶ。CheapTrick は 1.7〜5.6dB(合成・selftest)。
周期性: 中心揃え Hann 2048 のパワーで、帯域(0.3–1・1–2.5・2.5–5・5–10kHz)ごとに 倍音の山(k·f0 ± 0.2 f0 の最大)と
  谷((k + 0.4〜0.6) f0 の最小)のパワー比(dB)/ 30、[0, 1.5]。無声・無音は 0。
時刻: フレーム t の中心 = xa のサンプル A + tH(xa は左右に A = 1024 の余白)= harvest のフレーム t と同じ時刻。
"""
from __future__ import annotations

import math

import numpy as np
import torch

import nvoc as N

NFFT = 2048
A = NFFT // 2
HB = ((300, 1000), (1000, 2500), (2500, 5000), (5000, 10000))
C0_SIL = math.log(1e-5) * math.sqrt(N.N_MEL)
_NP: dict = {}
_CACHE: dict = {}


def _np_consts() -> tuple:
    if not _NP:
        import zsvc as Z
        _NP["fb"] = N.mel_fb().numpy().astype(np.float64)
        _NP["dct"] = Z.dct_mat(N.N_MEL).numpy().astype(np.float64)
    return _NP["fb"], _NP["dct"]


def envelope(xa: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """xa [A + T·H + A](余白つき)・f0 [T](Hz・0 = 無声)→ c [25, T](CheapTrick の mel 包絡の DCT)。"""
    import pyworld
    fb, dct = _np_consts()
    T = len(f0)
    tt = (A + np.arange(T) * N.HOP) / N.SR
    sp = pyworld.cheaptrick(xa.astype(np.float64), f0.astype(np.float64), tt, N.SR, fft_size=NFFT)
    mel = np.log(np.maximum(np.sqrt(sp) @ fb.T, 1e-5)).T
    return (dct[:25] @ mel).astype(np.float32)


def _consts(dev) -> torch.Tensor:
    if dev not in _CACHE:
        _CACHE[dev] = torch.hann_window(NFFT, periodic=False, device=dev, dtype=torch.float64)
    return _CACHE[dev]


def _interp_bins(v: torch.Tensor, fbin: torch.Tensor) -> torch.Tensor:
    nb = v.shape[-1]
    fb = fbin.clamp(0, nb - 1 - 1e-6)
    i0 = fb.floor().long()
    w = fb - i0
    return torch.gather(v, -1, i0) * (1 - w) + torch.gather(v, -1, i0 + 1) * w


def periodicity(xa: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """xa [B, A + T·H + A]・f0 [B, T] → per [B, 4, T]。"""
    dev = xa.device
    win = _consts(dev)
    B, T = f0.shape
    fr = xa.double().unfold(-1, NFFT, N.HOP)[:, :T]
    P = torch.fft.rfft(fr * win, n=NFFT).abs().pow(2)
    df = N.SR / NFFT
    f0d = f0.double()
    fv = torch.where(f0d > 0, f0d, torch.full_like(f0d, 500.0))
    kmax = int(HB[-1][1] / 60.0) + 1
    k = torch.arange(1, kmax + 1, device=dev, dtype=torch.float64)
    fk = fv[..., None] * k
    pk = torch.stack([_interp_bins(P, (fk + o * fv[..., None]) / df) for o in (-0.2, -0.1, 0.0, 0.1, 0.2)], -1).amax(-1)
    vl = torch.stack([_interp_bins(P, (fk + o * fv[..., None]) / df) for o in (0.4, 0.5, 0.6)], -1).amin(-1)
    out = []
    for lo, hi in HB:
        m = ((fk >= lo) & (fk < hi) & (fk < N.SR / 2 - 200)).double()
        num, den = (pk * m).sum(-1), (vl * m).sum(-1)
        hnr = 10 * torch.log10((num + 1e-20) / (den + 1e-20))
        ok = (m.sum(-1) > 0) & (f0d > 0) & (num > 1e-12)
        out.append(torch.where(ok, (hnr / 30).clamp(0, 1.5), torch.zeros_like(hnr)))
    return torch.stack(out, 1).float()


def _selftest() -> None:
    import zsvc as Z
    rng = np.random.default_rng(0)
    T = 120
    t = np.arange(T * N.HOP + 2 * A) / N.SR
    lowb = Z.mel_centers().numpy() < 1500
    fb, dct = _np_consts()
    fmt = ((700, 90), (1200, 110), (2600, 160), (3500, 200))

    def env_amp(f):
        r = np.ones_like(f)
        for fc, bw in fmt:
            r = r / np.sqrt((1 - (f / fc) ** 2) ** 2 + (f * bw / fc ** 2) ** 2)
        return r * (f / 1000 + 0.3) ** -1.0

    def harm(f0):
        x = np.zeros_like(t)
        for kk in range(1, int(20000 / f0)):
            x += env_amp(np.array([kk * f0]))[0] * np.sin(2 * np.pi * kk * f0 * t + 0.37 * kk * kk)
        x = x / np.abs(x).max() * 0.3
        return x + rng.standard_normal(len(x)) * x.std() * 1e-3

    def old(x):
        fr = np.lib.stride_tricks.sliding_window_view(x, 1024)[::N.HOP][:T]
        return dct[:25] @ np.log(np.maximum(np.abs(np.fft.rfft(fr * np.hanning(1024), n=NFFT)) @ fb.T, 1e-5)).T

    db = lambda a, b: float(np.sqrt(((dct[:25].T @ (a - b))[lowb] ** 2).mean()) * 20 / math.log(10))
    print("+4 半音の包絡の変化(合成・鋭いフォルマント・雑音の床 −60dB・1.5kHz 未満の mel 帯・DCT24 で戻した包絡の RMS dB)")
    for f0 in (110, 220, 300, 400, 450):
        e, o = [], []
        for ff in (f0, f0 * 2 ** (4 / 12)):
            x = harm(ff)
            e.append(envelope(x, np.full(T, float(ff)))[:, 10:-10].mean(1))
            o.append(old(x)[:, 10:-10].mean(1))
        dn, do = db(e[0], e[1]), db(o[0], o[1])
        print(f"  f0 {f0:4d}Hz: CheapTrick {dn:5.2f} dB(旧 因果 mel {do:5.2f} dB)")
        assert dn < 6.0 and (f0 < 300 or dn < do / 3), (f0, dn, do)
    c = envelope(np.zeros(T * N.HOP + 2 * A), np.zeros(T))
    assert abs(float(c[0].mean()) - C0_SIL) < 1e-3 and float(np.abs(c[1:]).max()) < 1e-3, (c[0].mean(), C0_SIL)
    p = periodicity(torch.zeros(1, T * N.HOP + 2 * A), torch.zeros(1, T))
    assert float(p.abs().max()) == 0.0
    print("無音: c0 = √128·log 1e-5・c1.. = 0・周期性 0")
    x = torch.from_numpy(harm(250.0)).float()[None]
    f0f = torch.full((1, T), 250.0)
    prev = None
    for snr in (40, 20, 10, 0, -10):
        nz = torch.from_numpy(rng.standard_normal(x.shape[-1])).float()[None] * x.std() * 10 ** (-snr / 20)
        pp = periodicity(x + nz, f0f)[0, :, 10:-10].mean(1).numpy()
        print(f"  SNR {snr:4d}dB: 周期性 {np.round(pp, 2)}")
        if prev is not None:
            assert (pp <= prev + 0.05).all(), (snr, pp, prev)
        prev = pp
    print("pae selftest OK")


if __name__ == "__main__":
    _selftest()
