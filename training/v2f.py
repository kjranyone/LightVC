"""V2F — 周波数構造を保つ出荷可能な幹（`PROCEDURE.md` 2.1 の後継）。

**なぜ 1D をやめるか**: 20000 step の同一プロトコル対照で、周波数軸を保つ
`Wavehax2D` は prior を横切って **+0.0912** まで伸びたのに対し、
1D 幹の `V1D` は dim 128/384/640 のどれでも頭打ちで**一度も越えなかった**
（`RESEARCH.md` 2026-08-12 切り分け結果 3）。`cin = 851` を dim へ一度に潰すのが律速。

**なぜ `Wavehax2D` をそのまま使わないか**: `nn.GroupNorm(1, ch)` は `[B, C, F, T]` の
**T 込み**で正規化する＝**発話全体統計**。`CLAUDE.md` の出荷ゲート
（推論経路に発話全体の統計を置かない）に落ちる。ストリームでは値が変わる。

∴ 同じ骨格のまま、**時間軸を跨がない正規化**に差し替える:
`FreqNorm` は `[B, C, F, T]` を **(C, F) 方向だけ**で正規化し、T では一切混ぜない。
∴ フレーム t の出力はフレーム t の入力だけで決まり、未来にも過去にも依存しない。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqNorm(nn.Module):
    """時間軸を跨がない正規化。**基準の取り方を `mode` で切り替える。**

    ⚠ 20000 step の対照で、`GroupNorm`（発話全体統計）を素朴な per-frame 正規化
    （`mode="freq"`）に替えると **8000 step で反転して悪化**した
    （`RESEARCH.md` 2026-08-12 切り分け結果 5）。差は正規化 1 点だけ。
    ∴ 争点は「発話全体統計が要る」ではなく「**基準が時刻ごとに揺れないこと**が要る」
    可能性がある。出荷可能なまま基準を安定させる手が 2 つあるので、切り分ける。

    - `freq`  : 各時刻で (C, F) の平均・分散。基準が毎フレーム動く（現行）
    - `none`  : 正規化しない（affine だけ）。「正規化そのものが要るか」の対照
    - `ema`   : 走行平均。**因果的**（過去だけ）なので未来不変性を壊さない
    - `fixed` : 固定定数で割る。完全に静的で出荷ゲートに何も抵触しない
    - `cummean`: **減衰なしの累積平均**＝`GroupNorm` の因果版。t の基準は
      「t 以前の全フレーム」で、発話が進むほど `GroupNorm` に漸近する。
      未来を見ないので出荷ゲートは通る。`ema`（減衰あり）が最悪だったので、
      **減衰の有無を 1 点だけ動かして区別する**。
    """

    def __init__(self, ch: int, eps: float = 1e-5, mode: str = "freq",
                 decay: float = 0.99, scale: float = 1.0):
        super().__init__()
        self.eps, self.mode, self.decay, self.scale = eps, mode, decay, scale
        self.g = nn.Parameter(torch.ones(1, ch, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x * self.g + self.b
        if self.mode == "fixed":
            return x * (self.g / self.scale) + self.b
        m = x.mean(dim=(1, 2), keepdim=True)          # [B,1,1,T]
        v = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        if self.mode == "cummean":
            # ⚠ 減衰なしの累積平均。t の基準 = t 以前の全フレームの平均。
            #   `cumsum / arange` は走行アキュムレータなのでストリームでも同一。
            k = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
            m2 = torch.cumsum(m, -1) / k
            v = torch.cumsum(v + m * m, -1) / k - m2 * m2
            m = m2
            v = v.clamp(min=0.0)
        elif self.mode == "ema":
            # ⚠ 因果的な指数移動平均。t の基準は t 以前だけで決まる。
            #   `cummax`/`cumsum` と同じで、ストリームでも同一値になる。
            w = self.decay ** torch.arange(
                x.shape[-1] - 1, -1, -1, device=x.device, dtype=x.dtype)
            cw = torch.cumsum(w.flip(0), 0).flip(0)
            m = torch.cumsum(m.flip(-1) * 0 + m * w, -1) / cw.clamp(min=1e-8)
            v = torch.cumsum(v * w, -1) / cw.clamp(min=1e-8)
        return (x - m) * torch.rsqrt(v + self.eps) * self.g + self.b


class V2F(nn.Module):
    def __init__(self, cin: int = 4, ch: int = 48, layers: int = 8,
                 kf: int = 7, kt: int = 3, norm: str = "freq"):
        super().__init__()
        self.cin, self.ch, self.layers, self.kf, self.kt = cin, ch, layers, kf, kt
        self.norm = norm
        self.inp = nn.Conv2d(cin, ch, 1)
        # 周波数方向の dilation を 1 層おきに倍にして帯域全体へ届かせる。
        self.dil = [2 ** (i // 2) for i in range(layers)]
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(nn.ModuleList([
                nn.Conv2d(ch, ch, (kf, kt), dilation=(self.dil[i], 1)),
                FreqNorm(ch, mode=norm),
                nn.Conv2d(ch, 3 * ch, 1),
                nn.Conv2d(3 * ch, ch, 1),
            ]))
        self.out = nn.Conv2d(ch, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def ctx(self) -> int:
        """幹の左文脈（合成フレーム数）。時間方向は各層 kt-1。"""
        return self.layers * (self.kt - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, cin, F, T] -> [B, 2, F, T]。**時間方向は左パディングのみ＝先読み 0**。"""
        h = self.inp(x)
        for (c, n, u, d), df in zip(self.blocks, self.dil):
            pf = ((self.kf - 1) * df) // 2
            y = F.pad(h, (self.kt - 1, 0, pf, pf))
            y = d(F.gelu(u(n(c(y)))))
            h = h + y
        return self.out(h)

    def arch(self) -> dict:
        return {"arch": "v2f", "cin": self.cin, "ch": self.ch, "L": self.layers,
                "kf": self.kf, "kt": self.kt, "ctx": self.ctx, "norm": self.norm}


class V2FStream:
    """ブロック実行。**層ごとに時間方向の左文脈 `kt-1` を保持**する。

    ⚠ キャッシュ無しで span 全体を毎ブロック再計算すると 9 倍（span 18 / emit 2）の
    無駄になる。未最適化実装の RTF を方式の否定に使わない
    （`memory/no-half-baked-implementations`）。
    """

    def __init__(self, net: "V2F", nbin: int = 257):
        self.net = net
        self.nbin = nbin
        self.reset()

    def reset(self) -> None:
        n = self.net
        self.cache = [torch.zeros(1, n.ch, self.nbin, n.kt - 1)
                      for _ in range(n.layers)]

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """x: [1, cin, F, emit] -> [1, 2, F, emit]。"""
        n = self.net
        h = n.inp(x)
        for i, ((c, nm, u, d), df) in enumerate(zip(n.blocks, n.dil)):
            hh = torch.cat([self.cache[i], h], dim=-1)
            self.cache[i] = hh[..., -(n.kt - 1):] if n.kt > 1 else self.cache[i]
            pf = ((n.kf - 1) * df) // 2
            y = F.pad(hh, (0, 0, pf, pf))
            y = d(F.gelu(u(nm(c(y)))))
            h = h + y
        return n.out(h)


class V2P(nn.Module):
    """周波数プーリング型。**ch を上げる代わりに周波数点数を下げる。**

    切り分け 8（`RESEARCH.md` 2026-08-12）: 品質の支配変数は容量で、
    `ch96`（1.997 M）は prior を越えるが `ch16`（0.056 M）は越えない。
    一方 RTF 予算 0.25 は ch16 しか許さない。**この衝突を演算量の形で解く。**

    畳み込みのコストは `ch^2 * kf * kt * F` なので、F を 1/8 にすれば
    ch を 2.8 倍にしても同じコストになる。周波数方向の情報は
    stride 付き conv で落とし、最後に転置 conv で戻す。

    ⚠ **時間方向は一切触らない**（stride も pooling も入れない）。
    先読み 0 と発話長不変を壊さないため。周波数方向だけを畳む。
    """

    def __init__(self, cin: int = 4, ch: int = 64, layers: int = 8,
                 kf: int = 7, kt: int = 3, norm: str = "cummean",
                 down: int = 2, nbin: int = 257):
        super().__init__()
        self.cin, self.ch, self.layers = cin, ch, layers
        self.kf, self.kt, self.norm, self.down = kf, kt, norm, down
        self.nbin = nbin
        self.inp = nn.Conv2d(cin, ch, 1)
        # 周波数を 2 段階で落とす: 257 -> 129 -> 33（stride 2 と 4）
        self.d1 = nn.Conv2d(ch, ch, (4, 1), stride=(2, 1), padding=(1, 0))
        self.d2 = nn.Conv2d(ch, ch, (8, 1), stride=(4, 1), padding=(2, 0))
        self.dil = [2 ** (i // 2) for i in range(layers)]
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(ch, ch, (kf, kt), dilation=(self.dil[i], 1)),
                FreqNorm(ch, mode=norm),
                nn.Conv2d(ch, 3 * ch, 1),
                nn.Conv2d(3 * ch, ch, 1),
            ]) for i in range(layers)
        ])
        self.u2 = nn.ConvTranspose2d(ch, ch, (8, 1), stride=(4, 1), padding=(2, 0))
        self.u1 = nn.ConvTranspose2d(ch, ch, (4, 1), stride=(2, 1), padding=(1, 0))
        self.out = nn.Conv2d(ch, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @property
    def ctx(self) -> int:
        return self.layers * (self.kt - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        f0 = h.shape[-2]
        h = self.d1(h)
        f1 = h.shape[-2]
        h = self.d2(h)
        for (c, n, u, d), df in zip(self.blocks, self.dil):
            pf = ((self.kf - 1) * df) // 2
            y = F.pad(h, (self.kt - 1, 0, pf, pf))
            y = d(F.gelu(u(n(c(y)))))
            h = h + y
        # ⚠ 257 は奇数なので転置 conv は 256 しか返さない。**末尾を複製して埋めない**
        #   （Nyquist 付近を捏造することになる）。足りない分だけ 0 詰めしてから
        #   `out` に渡し、`out` が学習で埋める。
        h = self.u2(h)
        if h.shape[-2] < f1:
            h = F.pad(h, (0, 0, 0, f1 - h.shape[-2]))
        h = h[..., :f1, :]
        h = self.u1(h)
        if h.shape[-2] < f0:
            h = F.pad(h, (0, 0, 0, f0 - h.shape[-2]))
        return self.out(h[..., :f0, :])

    def arch(self) -> dict:
        return {"arch": "v2p", "cin": self.cin, "ch": self.ch, "L": self.layers,
                "kf": self.kf, "kt": self.kt, "ctx": self.ctx,
                "norm": self.norm, "down": self.down, "nbin": self.nbin}
