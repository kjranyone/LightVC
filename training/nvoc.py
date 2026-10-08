"""因果 NSF 型ボコーダ nvoc: 因果 log-mel(128 帯・48kHz)+ 明示 f0 の調波源 → 48kHz 波形。

時刻規約(Rust crates/lightvc-core/src/nvoc.rs と同一):
  フレーム t = x[(t+1)·HOP − WIN : (t+1)·HOP](左寄せ・先読み 0)。T = N // HOP。
  出力ブロック t = y[t·HOP : (t+1)·HOP] はフレーム ≤ t だけで決まり、x[t·HOP − DELAY : (t+1)·HOP − DELAY] を再構成する。
  アルゴリズム遅延 = HOP + DELAY = 10ms。
  調波源のブロック t は f0[t−1] → f0[t] の直線補間(t=0 は f0[0] 保持)。有声の境は f0 を持ち越し、振幅を直線で上げ下げする。
  調波の振幅は k·f0 ≤ H_MAX − H_ROLL で 1、H_MAX で 0 の直線(帯域制限パルス列)。単位 RMS に正規化。位相は float64 で積算。

生成器: 因果 conv_pre → 4 段 [LeakyReLU → 因果 ConvTranspose(2r, r) → + 調波源/雑音の strided 因果 conv → HiFi-GAN ResBlock1 の平均]
→ LeakyReLU → 因果 conv_post。全ての畳み込みは左詰め(未来を見ない)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

SR = 48000
HOP = 240
DELAY = 240
N_MEL = 128
NFFT = 2048
WIN = 1024
FMAX = 24000.0
H_MAX = 22000.0
H_ROLL = 2000.0
F0_MIN = 50.0
K_MAX = int(H_MAX // F0_MIN)
UPS = (5, 4, 4, 3)


def mel_fb() -> torch.Tensor:
    import librosa
    return torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=N_MEL, fmin=0.0, fmax=FMAX)).float()


class CausalMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fb", mel_fb())
        self.register_buffer("win", torch.hann_window(WIN))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-1] // HOP
        xp = F.pad(x[..., :T * HOP], (WIN - HOP, 0))
        fr = xp.unfold(-1, WIN, HOP)[..., :T, :] * self.win
        mag = torch.fft.rfft(fr, n=NFFT).abs()
        return torch.log(torch.clamp(mag @ self.fb.T, min=1e-5)).transpose(-1, -2)


def frame_interp(a: torch.Tensor) -> torch.Tensor:
    B, T = a.shape
    prev = torch.cat([a[:, :1], a[:, :-1]], -1)
    w = (torch.arange(HOP, device=a.device, dtype=a.dtype) + 1) / HOP
    return (prev[..., None] + (a - prev)[..., None] * w).reshape(B, T * HOP)


def harmonic_source(f0: torch.Tensor) -> torch.Tensor:
    """f0 [B,T] Hz(0=無声・フレーム)→ [B, T·HOP] 単位 RMS の帯域制限パルス列(無声は 0)。"""
    f0 = f0.float()
    prev = torch.cat([f0[:, :1], f0[:, :-1]], -1)
    fa = torch.where(prev > 0, prev, f0)
    fb = torch.where(f0 > 0, f0, prev)
    B, T = f0.shape
    w = (torch.arange(HOP, device=f0.device, dtype=torch.float64) + 1) / HOP
    f = (fa.double()[..., None] + (fb - fa).double()[..., None] * w).reshape(B, T * HOP)
    v = frame_interp((f0 > 0).float())
    ph = torch.remainder(torch.cumsum(f / SR, -1), 1.0)
    f32 = f.float()
    out = torch.zeros_like(f32)
    nrm = torch.zeros_like(f32)
    kmax = int(min(K_MAX, math.ceil(H_MAX / max(float(f32[f32 > 0].min()) if (f32 > 0).any() else H_MAX, 1.0))))
    for k0 in range(1, kmax + 1, 16):
        k = torch.arange(k0, min(k0 + 16, kmax + 1), device=f0.device, dtype=torch.float64)
        a = ((H_MAX - k.float()[:, None, None] * f32[None]) / H_ROLL).clamp(0, 1)
        s = torch.sin((torch.remainder(ph[None] * k[:, None, None], 1.0) * (2 * math.pi)).float())
        out = out + (a * s).sum(0)
        nrm = nrm + (a * a).sum(0)
    return v * out / torch.sqrt(nrm / 2).clamp(min=1e-3)


class CConv(nn.Module):
    def __init__(self, ci: int, co: int, k: int, d: int = 1, stride: int = 1, lpad: int | None = None, causal: bool = True):
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(ci, co, k, stride=stride, dilation=d))
        nn.init.normal_(self.conv.parametrizations.weight.original1, 0.0, 0.01)
        full = (k - 1) * d if lpad is None else lpad
        self.pad = (full, 0) if causal or lpad is not None else (full // 2, full - full // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.pad))


class RB(nn.Module):
    def __init__(self, c: int, k: int, dils: tuple[int, ...] = (1, 3, 5), causal: bool = True):
        super().__init__()
        self.c1 = nn.ModuleList(CConv(c, c, k, d, causal=causal) for d in dils)
        self.c2 = nn.ModuleList(CConv(c, c, k, 1, causal=causal) for _ in dils)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for a, b in zip(self.c1, self.c2):
            x = x + b(F.leaky_relu(a(F.leaky_relu(x, 0.1)), 0.1))
        return x


class NVoc(nn.Module):
    def __init__(self, ch: int = 256, kernels: tuple = ((3, 7, 11),) * 4, dils: tuple[int, ...] = (1, 3, 5), causal: bool = True):
        super().__init__()
        self.cfg = {"ch": ch, "kernels": [list(k) for k in kernels], "dils": list(dils), "ups": list(UPS)}
        self.causal = causal
        self.mel = CausalMel()
        self.pre = CConv(N_MEL, ch, 7, causal=causal)
        self.up = nn.ModuleList()
        self.src = nn.ModuleList()
        self.res = nn.ModuleList()
        self.strides = []
        tot = 1
        for i, r in enumerate(UPS):
            ci, co = ch >> i, ch >> (i + 1)
            u = weight_norm(nn.ConvTranspose1d(ci, co, 2 * r, r))
            nn.init.normal_(u.parametrizations.weight.original1, 0.0, 0.01)
            self.up.append(u)
            tot *= r
            s = HOP // tot
            self.strides.append(s)
            self.src.append(CConv(2, co, 2 * s, stride=s, lpad=s) if s > 1 else CConv(2, co, 3))
            self.res.append(nn.ModuleList(RB(co, k, dils, causal) for k in kernels[i]))
        self.post = CConv(ch >> len(UPS), 1, 7, causal=causal)

    def generate(self, mel: torch.Tensor, exc: torch.Tensor) -> torch.Tensor:
        """mel [B,N_MEL,T]・exc [B,2,T·HOP](調波源・雑音)→ y [B, T·HOP]。"""
        T = mel.shape[-1]
        x = self.pre(mel)
        n = T
        for i, r in enumerate(UPS):
            n *= r
            o = 0 if self.causal else r // 2
            x = self.up[i](F.leaky_relu(x, 0.1))[..., o:o + n]
            x = x + self.src[i](exc)
            x = sum(rb(x) for rb in self.res[i]) / len(self.res[i])
        return self.post(F.leaky_relu(x, 0.01)).squeeze(1)

    def forward(self, x_in: torch.Tensor, f0: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """x_in [B, (WIN−HOP) + T·HOP](先頭 WIN−HOP は左文脈)・f0 [B,T] → y [B, T·HOP] ≈ x[tH − DELAY ...]。"""
        mel = self.mel_ctx(x_in)
        T = mel.shape[-1]
        h = harmonic_source(f0[:, :T])
        nz = torch.randn_like(h) if noise is None else noise
        return self.generate(mel, torch.stack([h, nz], 1))

    def mel_ctx(self, x_in: torch.Tensor) -> torch.Tensor:
        """先頭 WIN−HOP サンプルを左文脈として使う log-mel(フレーム t の窓 = x_in[tH : tH + WIN])。"""
        T = (x_in.shape[-1] - (WIN - HOP)) // HOP
        fr = x_in.unfold(-1, WIN, HOP)[..., :T, :] * self.mel.win
        mag = torch.fft.rfft(fr, n=NFFT).abs()
        return torch.log(torch.clamp(mag @ self.mel.fb.T, min=1e-5)).transpose(-1, -2)


def macs_per_second(m: NVoc) -> float:
    ch, ks, ds = m.cfg["ch"], m.cfg["kernels"], m.cfg["dils"]
    rate = SR / HOP
    tot = rate * 7 * N_MEL * ch
    for i, r in enumerate(UPS):
        ci, co = ch >> i, ch >> (i + 1)
        rate *= r
        tot += rate * 2 * ci * co
        tot += rate * 2 * (2 * m.strides[i] if m.strides[i] > 1 else 3)
        tot += rate * sum(2 * len(ds) * k for k in ks[i]) * co * co
    tot += SR * 7 * (ch >> len(UPS))
    return tot


def export_tensors(m: NVoc) -> dict[str, torch.Tensor]:
    """weight_norm を畳んだ推論用テンソル(キー名は Rust と一致)。"""
    out = {"mel.fb": m.mel.fb, "mel.win": m.mel.win}

    def cc(name: str, c: CConv):
        out[f"{name}.w"] = c.conv.weight.detach()
        out[f"{name}.b"] = c.conv.bias.detach()
    cc("pre", m.pre)
    for i in range(len(UPS)):
        out[f"up{i}.w"] = m.up[i].weight.detach()
        out[f"up{i}.b"] = m.up[i].bias.detach()
        cc(f"src{i}", m.src[i])
        for j, rb in enumerate(m.res[i]):
            for q, (a, b) in enumerate(zip(rb.c1, rb.c2)):
                cc(f"res{i}.{j}.c1.{q}", a)
                cc(f"res{i}.{j}.c2.{q}", b)
    cc("post", m.post)
    return {k: v.float().contiguous() for k, v in out.items()}


def _selftest() -> None:
    torch.manual_seed(0)
    m = NVoc().eval()
    print("MAC/s (G)", round(macs_per_second(m) / 1e9, 2), "params (M)", round(sum(p.numel() for p in m.parameters()) / 1e6, 2))
    T = 60
    x = torch.randn(1, WIN - HOP + T * HOP) * 0.1
    f0 = torch.full((1, T), 220.0)
    f0[:, 20:25] = 0
    mel = m.mel_ctx(x)
    mel2 = m.mel(x[..., WIN - HOP:])
    assert mel.shape == (1, N_MEL, T)
    xp = torch.cat([torch.zeros(1, WIN - HOP), x[..., WIN - HOP:]], -1)
    assert (m.mel_ctx(xp) - mel2).abs().max() < 1e-5, "mel_ctx と CausalMel(0 左文脈)が一致しない"
    h = harmonic_source(f0)
    v = h[0, 30 * HOP:40 * HOP]
    assert abs(float(v.pow(2).mean().sqrt()) - 1.0) < 0.05, float(v.pow(2).mean().sqrt())
    assert float(h[0, 21 * HOP:24 * HOP].abs().max()) == 0.0
    ac = torch.stack([(v[:-L] * v[L:]).mean() for L in range(100, 400)])
    lag = 100 + int(ac.argmax())
    assert abs(SR / lag - 220.0) < 2, SR / lag
    noise = torch.randn(1, T * HOP)
    with torch.no_grad():
        y = m(x, f0, noise)
        for cut in (17, 33):
            x2 = x.clone()
            x2[..., WIN - HOP + cut * HOP:] = torch.randn_like(x2[..., WIN - HOP + cut * HOP:])
            f2 = f0.clone()
            f2[:, cut:] = 150.0
            n2 = noise.clone()
            n2[:, cut * HOP:] = torch.randn_like(n2[:, cut * HOP:])
            y2 = m(x2, f2, n2)
            d = (y2 - y).abs()[0]
            first = int((d > 1e-7).nonzero()[0]) if (d > 1e-7).any() else None
            assert first is not None and first >= cut * HOP, (cut, first)
            print(f"future invariance (random weights): edit at frame {cut} → first change at sample {first} (block start {cut * HOP})")
    print("nvoc selftest OK")


if __name__ == "__main__":
    _selftest()
