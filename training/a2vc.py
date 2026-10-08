"""A2-VC の回路(current/a2vc.md)。48kHz のまま・アップサンプリングなし・因果の拡張畳み込み網(NAM A2 の思想)。

  搬送波(音声レート・3ch): c_in(入力の声・学習時はピッチを嘘にする)・c_pulse(目標 f0 の帯域制限パルス列)・c_noise(白色雑音)
  制御(フレームレート): 声道包絡(log-mel の DCT 低次・128)と話者(256)→ 1x1 → 各層の γ, β → 直線補間で音声レートへ
  層: 因果拡張畳み込み k3 → (1 + γ)·h + β → LeakyReLU → 1x1 → 残差。出力 1x1。
  時刻規約: 出力ブロック t = y[tH:(t+1)H] はフレーム ≤ t と、搬送波の m ≤ (t+1)H − 1 だけで決まる。制御はフレーム t−1 → t を直線補間(nvoc と同じ)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvoc as N

DILS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 1, 3, 8, 21)


def frame_interp(c: torch.Tensor) -> torch.Tensor:
    """[B,C,T] → [B,C,T·HOP]: ブロック t は c[t−1] → c[t] の直線補間(t=0 は c[0])。"""
    B, C, T = c.shape
    prev = torch.cat([c[..., :1], c[..., :-1]], -1)
    w = (torch.arange(N.HOP, device=c.device, dtype=c.dtype) + 1) / N.HOP
    return (prev[..., None] + (c - prev)[..., None] * w).reshape(B, C, T * N.HOP)


def _mod(u: torch.Tensor, pg: torch.Tensor, dg: torch.Tensor, pb: torch.Tensor, db: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """u [B,C,T·HOP] · (1 + γ) + β → LeakyReLU。γ, β はフレーム値(前フレーム pg と差分 dg)からその場で直線補間(音声レートの γ, β を作らない)。"""
    B, C, n = u.shape
    u4 = u.view(B, C, pg.shape[-1], -1)
    g = pg[..., None] + dg[..., None] * w
    b = pb[..., None] + db[..., None] * w
    return F.leaky_relu(u4 * (1 + g) + b, 0.1).view(B, C, n)


_mod_fused = torch.compile(_mod, dynamic=True)


class A2Layer(nn.Module):
    def __init__(self, ch: int, d: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, dilation=d)
        self.pw = nn.Conv1d(ch, ch, 1)
        self.d = d

    def forward(self, x: torch.Tensor, pg: torch.Tensor, dg: torch.Tensor, pb: torch.Tensor, db: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        u = self.conv(F.pad(x, (2 * self.d, 0)))
        return x + self.pw((_mod_fused if u.is_cuda else _mod)(u, pg, dg, pb, db, w))


class A2Circuit(nn.Module):
    def __init__(self, ch: int = 24, d_ctrl: int = N.N_MEL, d_spk: int = 256, dils: tuple = DILS):
        super().__init__()
        self.ch, self.dils = ch, dils
        self.inp = nn.Conv1d(3, ch, 1)
        self.ctrl = nn.Sequential(nn.Conv1d(d_ctrl + d_spk, 256, 1), nn.GELU(), nn.Conv1d(256, 2 * ch * len(dils), 1))
        nn.init.zeros_(self.ctrl[-1].weight)
        nn.init.zeros_(self.ctrl[-1].bias)
        self.layers = nn.ModuleList(A2Layer(ch, d) for d in dils)
        self.out = nn.Conv1d(ch, 1, 1)

    def forward(self, carriers: torch.Tensor, ctrl: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """carriers [B,3,T·HOP]・ctrl [B,d_ctrl,T]・spk [B,d_spk] → y [B,T·HOP]。"""
        T = ctrl.shape[-1]
        c = self.ctrl(torch.cat([ctrl, spk[..., None].expand(-1, -1, T)], 1))
        prev = torch.cat([c[..., :1], c[..., :-1]], -1)
        d = c - prev
        w = (torch.arange(N.HOP, device=c.device, dtype=c.dtype) + 1) / N.HOP
        h = self.inp(carriers)
        ch = self.ch
        for i, l in enumerate(self.layers):
            sg, sb = slice(2 * i * ch, (2 * i + 1) * ch), slice((2 * i + 1) * ch, (2 * i + 2) * ch)
            h = l(h, prev[:, sg], d[:, sg], prev[:, sb], d[:, sb], w)
        return self.out(h).squeeze(1)

    def forward_ref(self, carriers: torch.Tensor, ctrl: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        """参照実装(音声レートの γ, β を frame_interp で作る)。forward と数値一致を selftest で確かめる。"""
        T = ctrl.shape[-1]
        c = frame_interp(self.ctrl(torch.cat([ctrl, spk[..., None].expand(-1, -1, T)], 1)))
        h = self.inp(carriers)
        for i, l in enumerate(self.layers):
            g = c[:, 2 * i * self.ch:(2 * i + 1) * self.ch]
            b = c[:, (2 * i + 1) * self.ch:(2 * i + 2) * self.ch]
            h = h + l.pw(F.leaky_relu(l.conv(F.pad(h, (2 * l.d, 0))) * (1 + g) + b, 0.1))
        return self.out(h).squeeze(1)


def macs_per_second(m: A2Circuit) -> float:
    per = 3 * m.ch + len(m.dils) * (3 * m.ch * m.ch + m.ch * m.ch + 2 * m.ch) + m.ch
    return per * N.SR


def _selftest() -> None:
    torch.manual_seed(0)
    m = A2Circuit().eval()
    rf = sum(2 * d for d in m.dils)
    print("GMAC/s", round(macs_per_second(m) / 1e9, 2), "| params (M)", round(sum(p.numel() for p in m.parameters()) / 1e6, 3),
          "| receptive field", rf, "samples =", round(rf / N.SR * 1000, 1), "ms")
    T = 40
    car = torch.randn(1, 3, T * N.HOP)
    ctrl = torch.randn(1, N.N_MEL, T)
    spk = torch.randn(1, 256)
    for p in m.ctrl[-1].parameters():
        nn.init.normal_(p, 0, 0.1)
    with torch.no_grad():
        y = m(car, ctrl, spk)
        dref = float((m.forward_ref(car, ctrl, spk) - y).abs().max())
        assert dref < 1e-5, dref
        print(f"forward == forward_ref (max diff {dref:.1e})")
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False
            mc, cc, kc, sc = m.cuda(), car.cuda(), ctrl.cuda(), spk.cuda()
            dc = float((mc(cc, kc, sc) - mc.forward_ref(cc, kc, sc)).abs().max())
            assert dc < 1e-5, dc
            print(f"cuda fused == cuda forward_ref (TF32 off・max diff {dc:.1e})")
            m = m.cpu()
        for cut in (13, 27):
            c2, k2 = car.clone(), ctrl.clone()
            c2[..., cut * N.HOP:] = torch.randn_like(c2[..., cut * N.HOP:])
            k2[..., cut:] = torch.randn_like(k2[..., cut:])
            d = (m(c2, k2, spk) - y).abs()[0]
            first = int((d > 1e-7).nonzero()[0])
            assert first >= cut * N.HOP, (cut, first)
            print(f"future invariance: edit at frame {cut} (sample {cut * N.HOP}) → first change at sample {first}")
    print("a2vc selftest OK")


if __name__ == "__main__":
    _selftest()
