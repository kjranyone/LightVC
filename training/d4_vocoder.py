"""D4: 条件付きNAM vocoder — mel+lf0+energy+spk(100fps) → 48kHz波形AR。

定理M1検証用(v1=同話者再構成)。codec潜在をスキップし、位相をサンプルレート
逐次で運ぶ。因果conv=訓練並列・推論逐次。

    CUDA_VISIBLE_DEVICES= uv run python d4_vocoder.py   # RTFプローブ
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 48000
HOP = 480
COND_DIM_IN = 274          # mel80 + lf0 + en + spk192
COND_CH = 256
SIG_CH = 48
N_LAYERS = 16
DILS = (1, 2, 4, 8, 16, 32, 64, 128) * 2
MEL_SCALE = 8.0


def causal_mel48(wav: torch.Tensor, n_fft: int = 1024, n_mels: int = 80) -> torch.Tensor:
    """wav [B,N]@48k -> mel [B,n_mels,N//480]。trailing窓=先読み0。"""
    import librosa
    B, N = wav.shape
    T = N // HOP
    n = T * HOP
    x = wav[:, :n]
    w = torch.hann_window(n_fft, device=wav.device)
    xp = F.pad(x[:, None, :], (n_fft - HOP, 0))[:, 0, :]
    S = torch.stft(xp, n_fft, HOP, n_fft, w, center=False,
                   return_complex=True).abs()
    fb = torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=n_fft,
                                              n_mels=n_mels)).to(wav.device)
    m = fb @ (S ** 2)
    return torch.log(torch.clamp(m, min=1e-5))


def tent_up(c: torch.Tensor, n_out: int) -> torch.Tensor:
    """c [B,C,T]@100fps -> [B,C,n_out]@48k。過去2フレーム補間=因果。"""
    B, C, T = c.shape
    t = (torch.arange(n_out, device=c.device, dtype=torch.float64) + 1.0) \
        / HOP - 2.0
    i = t.floor().clamp(0, T - 2)
    fr = (t - i).clamp(0.0, 1.0).to(c.dtype)
    j = i.long()
    return c[:, :, j] * (1 - fr) + c[:, :, j + 1] * fr


def excitation_from_cond(cond: torch.Tensor, n: int) -> torch.Tensor:
    """cond [B,274,T]のlf0(行80)から位相積算sin/cos励起 [B,2,n] を生成。
    f0>50.5Hz(有声)区間のみ非ゼロ。位相=Σf0/sr(定理P:位相は積分量)。
    crop先頭を位相0とする(生成は発話先頭=ストリーム開始と一致)。"""
    lf0 = cond[:, 80:81]                                  # [B,1,T]
    f0 = 200.0 * torch.exp(lf0.clamp(-20, 20))
    voiced = (f0 > 50.5).float()
    fu = tent_up(f0 * voiced, n)                          # [B,1,n] Hz
    vm = tent_up(voiced, n)
    ph = torch.cumsum(fu.squeeze(1), -1) * (2 * 3.141592653589793 / SR)
    e = torch.stack([torch.sin(ph), torch.cos(ph)], 1) * vm * 0.5
    return e.to(cond.dtype)


class Layer(nn.Module):
    def __init__(self, ch: int, dil: int, cond_ch: int):
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, 7, dilation=dil)
        self.mix = nn.Conv1d(cond_ch, ch, 1)
        self.pw = nn.Conv1d(ch, ch, 1)
        self.dil = dil

    def forward(self, h: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
        x = F.pad(h, (6 * self.dil, 0))
        a = F.leaky_relu(self.dw(x) + self.mix(cu), 0.01)
        return h + self.pw(a)


class D4Voc(nn.Module):
    def __init__(self, sig_ch: int = SIG_CH, layers: int = N_LAYERS,
                 excitation: bool = False):
        super().__init__()
        self.excitation = excitation
        self.enc = nn.Sequential(
            nn.Conv1d(COND_DIM_IN, COND_CH, 7),
            nn.LeakyReLU(0.01),
            nn.Conv1d(COND_CH, COND_CH, 7),
            nn.LeakyReLU(0.01),
            nn.Conv1d(COND_CH, sig_ch, 7),
        )
        self.inp = nn.Conv1d(3 if excitation else 1, sig_ch, 7)
        self.blocks = nn.ModuleList(
            [Layer(sig_ch, DILS[i % len(DILS)], sig_ch) for i in range(layers)])
        self.head = nn.Conv1d(sig_ch, 1, 1)
        self.scale = 0.01

    def _cond(self, cond: torch.Tensor, n: int) -> torch.Tensor:
        x = F.leaky_relu(self.enc[0](F.pad(cond, (6, 0))), 0.01)
        x = F.leaky_relu(self.enc[2](F.pad(x, (6, 0))), 0.01)
        c = self.enc[4](F.pad(x, (6, 0)))
        return tent_up(c, n)

    def forward(self, wav: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """wav [B,N](教師過去含む入力)・cond [B,274,T]@100fps → [B,N]。
        excitation=True時、wavが[B,3,N]ならそれをそのまま入力スタックに使う
        (サンプリング側で全発話分を事前計算する用)。"""
        n = (min(wav.shape[-1], cond.shape[-1] * HOP)) // HOP * HOP
        if self.excitation:
            if wav.dim() == 3 and wav.shape[1] == 3:
                x = wav[:, :, :n]
            else:
                x = torch.cat([wav[:, None, :n],
                               excitation_from_cond(cond, n)], 1)
        else:
            x = wav[:, None, :n]
        cu = self._cond(cond, n)
        h = self.inp(F.pad(x, (6, 0)))
        for b in self.blocks:
            h = b(h, cu)
        return torch.tanh(self.head(h)).squeeze(1)


if __name__ == "__main__":
    import time
    torch.set_num_threads(1)
    m = D4Voc().eval()
    T = 200
    wav = torch.randn(1, T * HOP)
    cond = torch.randn(1, COND_DIM_IN, T)
    with torch.no_grad():
        y = m(wav, cond)
        t0 = time.perf_counter()
        for _ in range(3):
            y = m(wav, cond)
        dt = (time.perf_counter() - t0) / 3
    n_par = sum(p.numel() for p in m.parameters())
    print(f"D4Voc: out {tuple(y.shape)}  {dt*1000:.1f} ms/{T}frames "
          f"= {dt*1000/T:.2f} ms/frame (budget ~3.5-6)  params {n_par/1e6:.2f}M")
