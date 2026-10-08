"""C1: 因果な内容符号器(converter.md §3・C1)。入力の因果 log-mel → ContentVec の単位(K 個・k-means)の事後確率。

入力: 因果 log-mel 128(nvoc の規約・フレーム t の窓は (t+1)·240 で終わる・出力部と F1 と共有の前処理)を (mel − log 1e-5)/10。
学習時だけ入力側に声道長の伸縮(振幅スペクトルを周波数方向に α 倍: |X_α|(f) = |X|(f/α)・α ∈ [0.8, 1.25])。教師は元の音声。
ネット: 因果な拡張畳み込み(k 3・dil 1,2,4,8,16,32 × 2・256ch・残差・受容野 約 254 フレーム = 1.27s)→ K ロジット。
教師: ContentVec(非因果・学習時だけ)の特徴を単位のコードブックへ cos/τ の softmax。生徒のフレーム t の正解 = 時刻 t·240 − 720(48kHz・15ms 前)の
教師の事後(ContentVec のフレーム中心 320j + 200(16kHz)の間を線形補間)。15ms の遅れは表の引きでほぼ無害(conv_c0 TAB_CVL3)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvoc as N

DILS = (1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)
RF = 2 + sum(2 * d for d in DILS)
LAG48 = 720
CV_HOP16 = 320
CV_C16 = 200


class MelFront(nn.Module):
    """x48 [B, (WIN − HOP) + T·HOP](先頭は左文脈)→ [B, 128, T]。alpha [B] があれば声道長の伸縮(学習の摂動)。"""

    def __init__(self):
        super().__init__()
        self.mel = N.CausalMel()

    def forward(self, x: torch.Tensor, alpha: torch.Tensor | None = None) -> torch.Tensor:
        T = (x.shape[-1] - (N.WIN - N.HOP)) // N.HOP
        fr = x.unfold(-1, N.WIN, N.HOP)[..., :T, :] * self.mel.win
        mag = torch.fft.rfft(fr, n=N.NFFT).abs()
        if alpha is not None:
            nb = mag.shape[-1]
            src = torch.arange(nb, device=x.device, dtype=mag.dtype)[None] / alpha[:, None].to(mag.dtype)
            i0 = src.floor().long().clamp(0, nb - 2)
            w = (src - i0).clamp(0, 1)
            g0 = torch.gather(mag, -1, i0[:, None, :].expand(-1, mag.shape[1], -1))
            g1 = torch.gather(mag, -1, (i0 + 1)[:, None, :].expand(-1, mag.shape[1], -1))
            mag = torch.where((src <= nb - 1)[:, None, :], g0 * (1 - w[:, None]) + g1 * w[:, None], torch.zeros_like(g0))
        mel = torch.log(torch.clamp(mag @ self.mel.fb.T, min=1e-5)).transpose(-1, -2)
        return (mel - math.log(1e-5)) / 10


class Block(nn.Module):
    def __init__(self, ch: int, d: int):
        super().__init__()
        self.d = d
        self.c1 = nn.Conv1d(ch, ch, 3, dilation=d)
        self.c2 = nn.Conv1d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(F.gelu(self.c1(F.pad(F.gelu(x), (2 * self.d, 0)))))


class C1(nn.Module):
    def __init__(self, k: int, ch: int = 256, dils: tuple = DILS):
        super().__init__()
        self.cfg = {"k": k, "ch": ch, "dils": list(dils)}
        self.inp = nn.Conv1d(N.N_MEL, ch, 3)
        self.blocks = nn.ModuleList(Block(ch, d) for d in dils)
        self.out = nn.Conv1d(ch, k, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.inp(F.pad(mel, (2, 0)))
        for b in self.blocks:
            x = b(x)
        return self.out(F.gelu(x))


def teacher_targets(post: torch.Tensor, pre48: int, T: int) -> torch.Tensor:
    """post [B, Tc, K](ContentVec のフレームの事後・区間の先頭 = 48kHz の標本 0)→ [B, K, T]: 生徒のフレーム t(区間の標本 pre48 + t·240 が窓の終わり − 240)の正解 = 時刻 pre48 + t·240 − LAG48。"""
    t48 = pre48 + torch.arange(T, device=post.device, dtype=torch.float64) * N.HOP - LAG48
    jf = ((t48 / 3.0) - CV_C16) / CV_HOP16
    j0 = jf.floor().long().clamp(0, post.shape[1] - 2)
    w = (jf - j0).clamp(0, 1).float()
    p = post[:, j0] * (1 - w)[None, :, None] + post[:, j0 + 1] * w[None, :, None]
    return p.transpose(1, 2)


def _selftest() -> None:
    torch.manual_seed(0)
    m = C1(200).eval()
    print("params (M)", round(sum(p.numel() for p in m.parameters()) / 1e6, 3), "RF frames", RF, f"({RF * N.HOP / N.SR:.2f}s)")
    fe = MelFront()
    T = 500
    x = torch.randn(1, (N.WIN - N.HOP) + T * N.HOP) * 0.05
    with torch.no_grad():
        y = m(fe(x))
        for cut in (137, 301):
            x2 = x.clone()
            x2[:, (N.WIN - N.HOP) + cut * N.HOP:] = torch.randn_like(x2[:, (N.WIN - N.HOP) + cut * N.HOP:]) * 0.05
            d = (m(fe(x2)) - y).abs().amax(1)[0]
            first = int((d > 0).nonzero()[0])
            assert first >= cut, (cut, first)
            print(f"未来不変性: 入力のサンプル {cut * N.HOP} 以降の書き換え → 最初に変わるフレーム {first}")
        a = fe(x, torch.tensor([1.0]))
        assert float((a - fe(x)).abs().max()) < 1e-5, "α = 1 で恒等でない"
        print("声道長の伸縮: α = 1 で恒等")
    post = torch.zeros(1, 40, 3)
    post[0, :, 0] = 1
    post[0, 20:, :] = torch.tensor([0.0, 1.0, 0.0])
    ctxm = N.WIN - N.HOP
    tg = teacher_targets(post, pre48=ctxm, T=200)
    sw = int((tg[0, 1] > 0.5).nonzero()[0])
    j_half = 19.5
    t48 = (j_half * CV_HOP16 + CV_C16) * 3
    exp = math.floor((t48 + LAG48 - ctxm) / N.HOP) + 1
    print(f"教師の整列(pre48 = CTXM): 事後が半分に切り替わる ContentVec の位置 19.5 = 48kHz {t48:.0f} → 生徒のフレーム {sw}(計算値 {exp})")
    assert sw == exp, (sw, exp)
    print("c1_content selftest OK")


if __name__ == "__main__":
    _selftest()
