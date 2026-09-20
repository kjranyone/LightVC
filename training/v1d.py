"""V 本体（`PROCEDURE.md` 2.1 の作るもの #1）。

調波 prior への**加算複素残差**を出す 1D 因果ネット。自由位相にしない
（位相は prior が供給する。励起源を入れながら位相を自由にすると源と競合して
「かすれ」が出るという確定した帰属がある）。

`rddsp_hf.Wavehax2D` は使わない——`nn.GroupNorm(1, ch)` が `[B,C,F,T]` の T 込みで
正規化するので**発話全体統計**になり、`CLAUDE.md` の出荷ゲート（推論経路に発話全体の
統計を置かない）に落ちる。ここは時間軸をまたぐ正規化層を一切置かない。

    入力  cin = N_MEL + 3 * NBIN     （mel / prior real / prior imag / prior log|.|）
      -> Conv1d(cin -> dim, k_in, 左パディングのみ)
      -> ConvNeXtBlock1d(dim, k=k, causal=True) * L
      -> LayerNorm(dim) -> Linear(dim -> 2 * NBIN)
      -> S = complex(P.real + o[0], P.imag + o[1])
"""
from __future__ import annotations

import torch
import torch.nn as nn

import ship_front as SF
from kansei_vocoder import ConvNeXtBlock1d

EPS = 1e-5


def cin_of(nbin: int) -> int:
    return SF.N_MEL + 3 * nbin


def ctx_of(k_in: int, k: int, layers: int) -> int:
    """幹の左文脈（合成フレーム数）。ckpt の `args` に焼いて Rust 側と突き合わせる。"""
    return (k_in - 1) + layers * (k - 1)


class V1D(nn.Module):
    def __init__(self, cin: int | None = None, dim: int = 256, L: int = 6,
                 nbin: int = SF.NFFT_S // 2 + 1, k_in: int = 7, k: int = 3):
        """署名は `PROCEDURE.md` 2.1 の宣言どおり（Rust 側とキー名を一致させる）。

        `cin` は `N_MEL + 3*nbin` から導けるが、**明示させて食い違いを早く落とす**。
        """
        super().__init__()
        want = cin_of(nbin)
        if cin is not None and cin != want:
            raise ValueError(f"cin={cin} は nbin={nbin} と合わない（正 {want}）")
        layers = L
        self.nbin = nbin
        self.dim = dim
        self.layers = layers
        self.k_in = k_in
        self.k = k
        self.cin = cin_of(nbin)
        self.ctx = ctx_of(k_in, k, layers)
        self.inp = nn.Conv1d(self.cin, dim, k_in, padding=0)
        # ⚠ k を明示して渡す。既定 7 のままだと CTX が 42・net 0.26 になり、
        #   27 時間走ってから RTF で落ちる（2.4b-1）。
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock1d(dim, k=k, causal=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 2 * nbin)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def features(self, mel_syn: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """[N_MEL, T] と複素 prior [NBIN, T] を cin 系列に畳む。"""
        return torch.cat([mel_syn, P.real, P.imag,
                          (P.abs() + EPS).log()], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """**正典の署名**（`PROCEDURE.md` 2.3 (1)）: [B, cin, T] -> [B, 2*NBIN, T]。

        左パディングのみ＝先読み 0。prior への加算は呼び出し側が行う
        （`residual()` / `apply()` が定型を持つ）。
        """
        x = self.inp(nn.functional.pad(x, (self.k_in - 1, 0)))
        for b in self.blocks:
            x = b(x)
        return self.out(self.norm(x.transpose(1, 2))).transpose(1, 2)

    def apply(self, x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """学習経路。x: [B, cin, T], P: [B, NBIN, T] complex -> [B, NBIN, T]。

        **加算複素残差**（自由位相にしない。位相は prior が供給する）。
        """
        o = self(x)
        return torch.complex(P.real + o[:, :self.nbin], P.imag + o[:, self.nbin:])

    def residual(self, mel_syn: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """1 発話ぶんの便宜経路。mel_syn: [N_MEL, T], P: [NBIN, T] complex。"""
        return self.apply(self.features(mel_syn, P)[None], P[None])[0]

    def arch(self) -> dict:
        """ckpt の `args` に焼く再構築情報（2.1 の作るもの i）。"""
        return {"arch": "v1d", "dim": self.dim, "L": self.layers,
                "k_in": self.k_in, "k": self.k, "nbin": self.nbin,
                "cin": self.cin, "ctx": self.ctx,
                "nfft_s": SF.NFFT_S, "hop_s": SF.HOP_S,
                "nfft_a": SF.NFFT_A, "hop_a": SF.HOP_A, "n_mel": SF.N_MEL}


class V1DStream:
    """ブロック実行（2.1 の作るもの #2c）。**層ごとに左文脈 `k-1` を保持**する。

    キャッシュ無しで毎ブロック span 全再計算すると net RTF 0.3971（実測）で
    予算 0.207 に到底届かない。それは方式の否定ではなく**未完成実装の否定**なので、
    RTF の判定はこちらで測る（`memory/no-half-baked-implementations`）。

    1 呼び出しで `emit` 個の合成フレームを出し、内部状態だけを進める。
    先読みは 0（右パディングを一切しない）。
    """

    def __init__(self, net: "V1D"):
        self.net = net
        self.reset()

    def reset(self) -> None:
        n = self.net
        self.c_in = torch.zeros(1, n.cin, n.k_in - 1)
        self.c_blk = [torch.zeros(1, n.dim, n.k - 1) for _ in range(n.layers)]

    @torch.no_grad()
    def step(self, mel_syn: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """mel_syn: [N_MEL, emit], P: [NBIN, emit] complex -> [NBIN, emit]。"""
        n = self.net
        x = n.features(mel_syn, P).unsqueeze(0)
        x = torch.cat([self.c_in, x], dim=-1)
        self.c_in = x[..., -(n.k_in - 1):] if n.k_in > 1 else self.c_in
        x = n.inp(x)
        for i, b in enumerate(n.blocks):
            h = torch.cat([self.c_blk[i], x], dim=-1)
            self.c_blk[i] = h[..., -(n.k - 1):] if n.k > 1 else self.c_blk[i]
            r = h[..., n.k - 1:]
            y = b.dw(h).transpose(1, 2)
            y = b.norm(y)
            y = b.pw2(b.act(b.pw1(y))).transpose(1, 2)
            x = r + y
        o = n.out(n.norm(x.transpose(1, 2))).transpose(1, 2)[0]
        return torch.complex(P.real + o[:n.nbin], P.imag + o[n.nbin:])
