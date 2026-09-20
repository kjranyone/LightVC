"""G-voc: the deployable decoder, trained on the FULL corpus.

CLAUDE.md: 本番学習は必ずフルコーパスを使う。少数話者・部分集合での学習は誤り。
Every run in this session used 80 utterances -- 6.7 minutes, 0.087% of what is on
disk -- and the deployable decoder stalled at PESQ 2.45. The refiner did not care
about data volume because its prior already carried the information; a GENERATOR
asked to build a waveform from f0 and mel has to learn voice variety itself.

    corpus   91,570 utterances / 3,445 speakers / ~127 h
             female-dataset (2776 real speakers) + irodori TTS (669)

What this trains (the only configuration in this project with a product path):

    prior        f0 sinusoids with random per-harmonic phase, redrawn every step
                 -- f0 is available at conversion time; measured glottal phase is
                 not, and has no low-rate description (utterance-level pulse shape
                 retains 0% of it), so the rddsp core cannot be the VC decoder
    conditioning 80-band log-mel
    output       complex spectrogram, predicted directly, causal in time
    losses       multi-resolution STFT with the log floor at -60 dB of the target
                 peak (the absolute 1e-5 floor sends 86% of the gradient below
                 -100 dB), spectrogram consistency, WavLM-conv feature distance,
                 and a sub-band CQT discriminator after --dstart

Evaluation speakers are HELD OUT of training by speaker id -- the same twelve
every number in this session was measured on, so the full-corpus result is
directly comparable to the 80-utterance one.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import score_one
from rddsp_hf import Wavehax2D, NFFT, HOP, NBIN
import ship_front as SF
from causal_mel import causal_mel
from rddsp_gpu import (stft, istft, mrstft, WavLMConvLoss, CTX, safe_score,
                       mel_to_linear, CACHE_DIR)
import rddsp_gpu as G

ROOT = Path(__file__).resolve().parent.parent
OUT_MODE = "residual"
V1DCLS = None           # v1d.V1D（--arch v1d のときだけ設定）
PRIOR_FN = None
FEATS = None
MELFN = None            # set from --melframe; None = use the stored centered mel
SHIP = False            # --front ship: 出荷 front-end（先読み 0 / 発話統計 0）
SHARDS = Path("/home/kojirotanaka/kjranyone/LightVC/data/full_prep")

# The twelve speakers every measurement in this session used (pick(80,12)'s test
# split). Held out by id so the full-corpus number is comparable to the small one.
HELD_OUT = {
    "ed8f7b91101f5a49", "3a6b1f5bb2b12ea5", "57babe52c37f4b0b", "e53ec77eff9f568d",
    "5e518c920ad7df09", "db511593103038ef", "604ddfc36afa35ca", "2b3e92bdcbb84477",
    "01763f790c9e37ee", "e52c37a2dad95317", "e9a550065733f0bb", "f6fb3bf73dee261e",
}


def is_held(spk: str) -> bool:
    return spk.split("/")[-1] in HELD_OUT


class ShardStream:
    """Random-order shard reader. 47 GB does not fit in RAM, so hold a few shards
    at a time and draw utterances from the union of those."""

    def __init__(self, files, keep: int = 3, seed: int = 0):
        self.files = list(files)
        self.keep = keep
        self.rng = random.Random(seed)
        self.rng.shuffle(self.files)
        self.pos = 0
        self.buf: list[dict] = []
        while len(self.buf) < keep * 400 and self.pos < len(self.files):
            self._load()

    def _load(self):
        if self.pos >= len(self.files):
            self.rng.shuffle(self.files)
            self.pos = 0
        d = torch.load(self.files[self.pos], weights_only=False)
        self.pos += 1
        self.buf.extend(x for x in d if not is_held(x.get("spk", "")))
        if len(self.buf) > self.keep * 1000:
            self.buf = self.buf[-self.keep * 1000:]

    def draw(self):
        if self.rng.random() < 0.002 or not self.buf:
            self._load()
        return self.buf[self.rng.randrange(len(self.buf))]


_GMEL = None             # (enet, gnet) — V を G 出力 mel 条件で学習する経路


def ship_item(w, dev, W):
    """出荷 front-end で 1 発話ぶんの条件を作る。発話全体の統計ゼロ・先読みゼロ
    （`ship_check.py` で PASS 確認済み: 台帳 18.71 ms / 残予算 +11.29 ms）。

    _GMEL 設定時は条件 mel を **G の予測 mel** に置き換える（製品では V は G の
    ぼやけた mel を食うのに GT mel でしか学習していない＝入力分布の食い違い。
    E male OOD と同型・2026-08-21 の帰属）。励起 f0 と教師波形は真値のまま。"""
    n = w.shape[-1]
    T = SF.n_frames(n)
    f0, _ = SF.causal_f0(w)
    if _GMEL is None:
        m = SF.mel(w)
    else:
        enet, gnet = _GMEL
        from train_vc_g import resample_to, F0_FPS
        w32 = w * 32768.0                      # E/G は x32768 慣習
        mel32 = SF.mel(w32)
        t = mel32.shape[-1]
        with torch.no_grad():
            c = enet(mel32[None])[0]
            f0g, _v = SF.causal_f0(w32)
            lf0 = torch.log(f0g.clamp(min=50.0) / 200.0)[:t]
            hop = 512
            nf = n // hop
            rms = torch.sqrt((w[: nf * hop].reshape(nf, hop) ** 2).mean(-1) + 1e-12)
            en = torch.log(resample_to(rms[None].cpu(), t, F0_FPS)[0].clamp(min=1e-4)).to(w.device)
            feat = torch.cat([c[:, :t], lf0[None, :t].to(w.device),
                              en[None, :t]], 0)[None]
            m = gnet(feat)[0] - SF.V_MEL_ADAPT  # 80-mel 空間で V 慣習へ
    return dict(w=w, T=T, f0=f0, mlin=SF.to_frames(W @ m, T),
                mel80=SF.to_frames(m, T))


_F0AUG = None             # (r_lo, r_hi, nz_lo, nz_hi) — 励起系セルフペア augmentation


def ship_prior(it, gen, dev, _level=None):
    f0 = it["f0"]
    nm = 0.3
    if _F0AUG is not None:
        r_lo, r_hi, nz_lo, nz_hi = _F0AUG
        r = float(torch.empty(1).uniform_(r_lo, r_hi))
        nm = float(torch.empty(1).uniform_(nz_lo, nz_hi))
        f0 = torch.where(f0 > 0, f0 * r, f0)
    return SF.nhv_spec(it["mlin"], f0, it["w"].shape[-1], gen, noise_mix=nm,
                       T=it["T"])


def v1d_feats(it):
    """V1D の入力 [cin, T]。**`mlin`（線形軸 257）ではなく mel(80)** を使う——
    `cin = N_MEL + 3 * NBIN = 851`（`PROCEDURE.md` 2.1 の具体値表）。"""
    P = it["_P"]
    T = P.shape[-1]
    return torch.cat([it["mel80"][:, :T], P.real, P.imag,
                      torch.log(P.abs() + 1e-5)], 0), P


def ship_feats(it):
    P = it["_P"]
    T = P.shape[-1]
    return torch.cat([it["mlin"][None, :, :T], P.real[None], P.imag[None],
                      torch.log(P.abs()[None] + 1e-5)], 0), P


_VRESID = False          # True: S = P + R(V)（v_resid.md 案A）


def mel_now(w, stored, dev):
    """Path B: recompute the mel left-aligned so the product ledger can close.

    The stored mel is librosa-centered at n_fft 2048 -- 23.2 ms of lookahead,
    which the ROADMAP marks path A (diagnostic only). A left-aligned window has
    ZERO lookahead at ANY n_fft: the window is the transient-smear knob, not the
    delay knob (causal_mel §Z0-A). So path B costs smearing, not latency."""
    if MELFN is None:
        return stored.to(dev).float()
    return MELFN(w)[0]


def to_gpu(it, dev, W):
    w = it["w"].to(dev).float() / 32767.0
    if SHIP:
        return ship_item(w, dev, W)
    mel = mel_now(w, it["mel"], dev)
    f0 = it["f0"].to(dev).float()
    T = stft(w).shape[-1]
    idx = (torch.arange(T, device=dev) * (mel.shape[-1] - 1) / max(T - 1, 1)).long()
    m = (W @ mel[:, idx])[None]
    n = w.shape[-1]
    f0u = R.frame_upsample(G_fill(f0).double().cpu(), n).clamp(min=0.0).to(dev)
    phi = torch.remainder(2 * math.pi * torch.cumsum(f0u, 0) / R.SR,
                          2 * math.pi).float()
    fv = f0[f0 > 50]
    return dict(w=w, m=m, T=T, phi=phi, f0med=float(fv.median()) if fv.numel() else 200.0)


def G_fill(f0):
    return R.fill_f0(f0.cpu())


def to_gpu_pre(x, dev, W):
    """Same tensors as to_gpu() but from the small cached corpus (float wav)."""
    w = x["w"].to(dev).float()
    if SHIP:
        return ship_item(w, dev, W)
    mel = mel_now(w, x["mel"], dev)
    f0 = x["f0"]
    T = stft(w).shape[-1]
    idx = (torch.arange(T, device=dev) * (mel.shape[-1] - 1) / max(T - 1, 1)).long()
    m = (W @ mel[:, idx])[None]
    n = w.shape[-1]
    f0u = R.frame_upsample(G_fill(f0).double(), n).clamp(min=0.0).to(dev)
    phi = torch.remainder(2 * math.pi * torch.cumsum(f0u, 0) / R.SR,
                          2 * math.pi).float()
    fv = f0[f0 > 50]
    return dict(w=w, m=m, T=T, phi=phi,
                f0med=float(fv.median()) if fv.numel() else 200.0)


def f0_prior(it, gen, dev, level: float):
    """Wavehax eq. 16: f0 sinusoids, per-harmonic phase redrawn every call."""
    phi, n = it["phi"], it["w"].shape[-1]
    K = max(1, int((R.SR / 2) / it["f0med"]))
    off = torch.rand(K, generator=gen, device=dev) * 2 * math.pi
    amp = math.sqrt(0.02 / K)
    e = torch.zeros(n, device=dev)
    for s0 in range(0, K, 32):
        kk = torch.arange(s0 + 1, min(s0 + 33, K + 1), device=dev,
                          dtype=torch.float32)[:, None]
        e = e + (amp * torch.sin(kk * phi[None] + kk * off[s0: s0 + 32, None])).sum(0)
    e = e + 0.01 * torch.randn(n, generator=gen, device=dev)
    return e * (level / e.std().clamp(min=1e-8))


def nhv_prior(it, gen, dev, level: float, noise_mix: float = 0.3):
    """NHV-style source-filter prior, from f0 and mel only.

    Neural Homomorphic Vocoder (Liu et al., Interspeech 2020) synthesises by
    filtering an impulse train and noise with linear time-varying filters. That
    is the family Beatrice v2 is in, and it reaches product quality at roughly a
    million parameters -- so "not enough capacity" cannot be what caps us at 2.6,
    since this net is already 2.0M.

    current/vocoder_a.md sets a strategy of NOT imitating that stack (F0 非隷属,
    to stay robust to octave errors). This is a MEASUREMENT of what that strategy
    costs, not an adoption of the stack.

    Deployable by construction: both ingredients come from f0 and mel, which
    exist at conversion time. The measured glottal phase does not (12.5), and has
    no low-rate description, which is why the rddsp core cannot be used here.

      p[n] = sum_k cos(k phi[n])      impulse train (ZERO relative phase)
      z[n] = white noise
      both filtered by the mel-derived envelope H(f,t)
    """
    phi, n = it["phi"], it["w"].shape[-1]
    K = max(1, int((R.SR / 2) / it["f0med"]))
    imp = torch.zeros(n, device=dev)
    for s0 in range(0, K, 32):
        kk = torch.arange(s0 + 1, min(s0 + 33, K + 1), device=dev,
                          dtype=torch.float32)[:, None]
        imp = imp + torch.cos(kk * phi[None]).sum(0)
    imp = imp / math.sqrt(K)
    z = torch.randn(n, generator=gen, device=dev)
    e = (1 - noise_mix) * imp + noise_mix * z
    # time-varying filter = the mel envelope, applied in the STFT domain
    E = stft(e)
    H = it["m"][0][:, : E.shape[-1]].exp()          # m is log-mel on the linear axis
    H = H / H.amax().clamp(min=1e-8)
    y = istft(E[:, : H.shape[-1]] * H, n)
    return y * (level / y.std().clamp(min=1e-8))


def feats(it, pw):
    P = stft(pw)[:, : it["T"]]
    T = P.shape[-1]
    return torch.cat([it["m"][:, :, :T], P.real[None], P.imag[None],
                      torch.log(P.abs()[None] + 1e-5)], 0), P


@torch.no_grad()
def evaluate(net, items, dev, seed=999):
    net.eval()
    ge = torch.Generator(device=dev).manual_seed(seed)
    ss, pr = [], []
    for it in items:
        n = it["w"].shape[-1]
        pw = PRIOR_FN(it, ge, dev, float(it["w"].std()))
        if SHIP:
            it["_P"] = pw
        f, P = FEATS(it) if SHIP else feats(it, pw)
        # ⚠ `hasattr(net, "apply")` で分岐しない——`nn.Module` は**常に** `apply` を
        #   持つので、Wavehax2D でも v1d 経路に入って TypeError で落ちる（実際に落ちた）。
        #   建築の判定は専用属性で行う。
        if V1DCLS is not None and isinstance(net, V1DCLS):
            S = (net.apply(f[None], P[None])[0] if OUT_MODE == "residual"
                 else (lambda o: torch.complex(o[:net.nbin], o[net.nbin:]))(net(f[None])[0]))
        else:
            o = net(f[None])[0]
            S = torch.complex(o[0], o[1])
            if _VRESID:
                S = S + P[:, :S.shape[-1]]
        gt = it["w"].cpu()
        ss.append(safe_score(gt, (SF.cistft(S, n) if SHIP else istft(S, n)).cpu()))
        pr.append(safe_score(gt, (SF.cistft(pw, n) if SHIP
                                  else pw[:n]).cpu()))
    net.train()
    return float(np.mean(ss)), float(np.mean(pr)), np.array(ss) - np.array(pr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=96)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--lr", type=float, default=3e-4)
    # ⚠ 既定を 100 -> 1.0 に変更（2026-08-12）。`WavLMConvLoss` を相対距離に直して
    #   値域が O(1e-4) から O(1) になったため、100 のままだと mrstft(≈1.5) を 100 倍で潰す。
    #   実測の識別力: 同一 0.000 / −6 dB 0.0027 / 無音 1.07 / 白色雑音 1.71 / 別発話 1.90。
    #   ⚠ **この項は音量差にほぼ盲目**（GroupNorm がスケールを除去。−6 dB で 0.27%）。
    #   レベルの教師は mrstft 側の仕事。
    ap.add_argument("--lmos", type=float, default=1.0)
    ap.add_argument("--logfloor", type=float, default=-60.0)
    ap.add_argument("--consist", type=float, default=1.0)
    ap.add_argument("--gan", type=float, default=1.0)
    ap.add_argument("--gmel-e", type=str, default=None,
                    help="V を G 出力 mel 条件で学習: E ckpt")
    ap.add_argument("--gmel-g", type=str, default=None,
                    help="同: G ckpt（--gmel-e とセット）")
    ap.add_argument("--ft-lr", type=float, default=0.0,
                    help="fine-tune: resume 時に net/disc のみ読み、固定 LR で新規 opt")
    ap.add_argument("--fm", type=float, default=2.0)
    ap.add_argument("--dstart", type=int, default=20000)
    ap.add_argument("--every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior", type=str, default="f0",
                    choices=["f0", "nhv"])
    ap.add_argument("--front", type=str, default="legacy",
                    choices=["legacy", "ship"])
    ap.add_argument("--melframe", type=str, default="stored",
                    choices=["stored", "causal"])
    ap.add_argument("--melnfft", type=int, default=1024)
    ap.add_argument("--tag", type=str, default="gvoc_full")
    # ⚠ 既定は v1d。Wavehax2D は `nn.GroupNorm(1, ch)` が [B,C,F,T] の T 込みで
    #    正規化する＝**発話全体統計**で、出荷ゲート（CLAUDE.md）に落ちる。
    ap.add_argument("--arch", type=str, default="v1d",
                    choices=["v1d", "wavehax", "v2f", "v2p"])
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--kin", type=int, default=7)
    ap.add_argument("--kblk", type=int, default=3)
    ap.add_argument("--snap", type=int, default=0,
                    help="この step ごとに中間スナップショットを残す（規則 7）")
    ap.add_argument("--hfw", type=float, default=0.0, help="高域重み（3.2）")
    ap.add_argument("--gainaug", type=int, default=1,
                    help="ゲイン aug。レベル依存性を消すための手段（3.5）")
    ap.add_argument("--f0aug", type=float, nargs=4, default=None,
                    metavar=("R_LO", "R_HI", "NZ_LO", "NZ_HI"),
                    help="励起系セルフペイ aug: f0 ratio 掃引 [r_lo,r_hi] × "
                         "励起ノイズ比掃引 [nz_lo,nz_hi]。教師は元 GT 波形")
    ap.add_argument("--vresid", action="store_true",
                    help="R0: S = P + R(V)（prior のコームを構造的に保持）")
    ap.add_argument("--resume", type=str, default=None)
    # ⚠ 診断専用。製品は residual（加算複素残差・自由位相にしない）。
    #   direct は「1D 幹への潰し」と「残差で prior に係留すること」を切り分けるためだけ。
    ap.add_argument("--out-mode", type=str, default="residual",
                    choices=["residual", "direct"])
    # ⚠ 位相の教師。**mrstft は magnitude のみ**（`.abs()`）で、`consist` は
    #   自己無矛盾性しか見ない。∴ rev までの目的関数には**位相を見る項が 1 つも無い**。
    #   実測（36 巡目）: gt との位相コヒーレンスが prior・net とも ~0。
    ap.add_argument("--norm", type=str, default="freq",
                    choices=["freq", "none", "ema", "fixed", "cummean"],
                    help="v2f の正規化。時間軸を跨がない範囲で基準の取り方を変える")
    ap.add_argument("--cplx", type=float, default=0.0,
                    help="複素領域 L1（S vs STFT(gt)）の重み。0 で無効")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    global PRIOR_FN, MELFN, SHIP, FEATS, OUT_MODE, V1DCLS
    SHIP = a.front == "ship"
    OUT_MODE = a.out_mode
    PRIOR_FN = (ship_prior if SHIP else
                (nhv_prior if a.prior == "nhv" else f0_prior))
    # ⚠ v1d は cin = 80 + 3*NBIN。ship_feats（4 面 × NBIN の 2D 用）を渡すと
    #    形が合わず、しかも mel を線形軸へ写した mlin が入ってしまう。
    FEATS = (v1d_feats if a.arch == "v1d" else ship_feats) if SHIP else feats
    if SHIP:
        import ship_check
        if not ship_check.main_ok():
            raise SystemExit("ship_check FAIL: 台帳が閉じないので学習を起動しない")
    if a.melframe == "causal":
        MELFN = lambda y: causal_mel(y, n_fft=a.melnfft, hop=R.HOP,
                                     num_mels=G.N_MEL, sr=R.SR)
    torch.manual_seed(a.seed)
    G.LOG_FLOOR_DB = a.logfloor
    W = mel_to_linear(dev)

    files = sorted(SHARDS.glob("sh_*.pt"))
    if not files:
        sys.exit("no shards; run prep_full.py first")
    stream = ShardStream(files, seed=a.seed)

    # EVALUATION SET: the exact twelve utterances every number in this session was
    # measured on, taken from the small cached corpus -- not re-drawn from the
    # shards. Pulling "some utterance from each held-out speaker" would give a
    # different set and make the full-corpus result incomparable to the
    # 80-utterance one, which is the whole point of running this. 12.14 is a
    # section about exactly that mistake.
    from rddsp_gpu import build as build_small
    _, te_small = build_small(80, 12)
    items_ev = []
    for x in te_small:
        items_ev.append(dict(w=x["gt"].to(dev), mel=x["mel"].to(dev),
                             f0=x["f0"], _pre=True))
    items_ev = [to_gpu_pre(x, dev, W) for x in items_ev]
    print(f"  shards {len(files)}  eval = the session's fixed {len(items_ev)} "
          f"held-out utterances", flush=True)

    if a.arch == "v2p":
        import v2f as V2
        net = V2.V2P(cin=4, ch=a.ch, layers=a.layers, norm=a.norm).to(dev)
        net.out.reset_parameters()
        CTX_N = net.ctx
    elif a.arch == "v2f":
        import v2f as V2
        net = V2.V2F(cin=4, ch=a.ch, layers=a.layers, norm=a.norm).to(dev)
        # ⚠ v2f は Wavehax と同じ**直接予測**。`out` の零初期化のままだと S = 0（無音）に
        #   なり、`consist` の分母 `|S|.clamp(min=1e-8)` が 1e8 倍に化けて 1 step 目から NaN。
        #   兄弟の Wavehax2D が `out.reset_parameters()` を呼ぶのと同じ理由。
        net.out.reset_parameters()
        CTX_N = net.ctx
    elif a.arch == "v1d":
        import v1d as V
        V1DCLS = V.V1D
        net = V.V1D(dim=a.dim, L=a.layers, k_in=a.kin, k=a.kblk).to(dev)
        # ⚠ residual では `out` の零初期化のままでよい（S = P から始まる）。
        #   direct は違う——零初期化だと S = 0（無音）になり、consist の分母
        #   `|S|.clamp(min=1e-8)` で 1e8 倍になって **1 step 目から NaN**。
        #   兄弟の Wavehax2D も同じ理由で `out.reset_parameters()` を呼んでいる。
        if a.out_mode == "direct":
            net.out.reset_parameters()
        CTX_N = net.ctx
    else:
        net = Wavehax2D(cin=4, ch=a.ch, layers=a.layers).to(dev)
        net.out.reset_parameters()      # direct prediction: zero-init has zero grad
        CTX_N = CTX
    print(f"  ch {a.ch}  {sum(p.numel() for p in net.parameters())/1e6:.3f}M  "
          f"batch {a.batch} crop {a.crop} steps {a.steps} logfloor {a.logfloor} "
          f"consist {a.consist} gan {a.gan} from {a.dstart}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.8, 0.99),
                            weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    if a.gmel_e and a.gmel_g:
        from train_vc_e import E1
        from train_vc_g import G1
        ek_ = torch.load(a.gmel_e, map_location=dev)
        en_ = E1(dim=ek_["args"]["dim"], layers=ek_["args"]["L"],
                 look=ek_["args"].get("look", 0)).to(dev).eval()
        en_.load_state_dict(ek_["net"])
        gk_ = torch.load(a.gmel_g, map_location=dev)
        gn_ = G1(dim=gk_["args"]["dim"], layers=gk_["args"]["L"]).to(dev).eval()
        gn_.load_state_dict(gk_["net"])
        for q_ in list(en_.parameters()) + list(gn_.parameters()):
            q_.requires_grad_(False)
        globals()["_GMEL"] = (en_, gn_)
    if a.f0aug is not None:
        globals()["_F0AUG"] = tuple(a.f0aug)
        print(f"  励起系 aug: f0 ratio x[{a.f0aug[0]},{a.f0aug[1]}] "
              f"noise [{a.f0aug[2]},{a.f0aug[3]}]（教師=元 GT 波形）", flush=True)
    if a.vresid:
        globals()["_VRESID"] = True
        print("  R0 residual: S = P + R(V)（zero-init R から開始）", flush=True)
        print(f"  条件 mel: G 予測（E={a.gmel_e} / G={a.gmel_g}）", flush=True)

    lmos = WavLMConvLoss(dev) if a.lmos > 0 else None
    disc = dopt = None
    if a.gan > 0:
        from cqt_disc import MSSubBandCQTDisc
        disc = MSSubBandCQTDisc(sr=R.SR).to(dev)
        dopt = torch.optim.AdamW(disc.parameters(), lr=a.lr, betas=(0.8, 0.99),
                                 weight_decay=1e-4)
    g = torch.Generator(device=dev).manual_seed(a.seed + 1)

    arch_args = (net.arch() if hasattr(net, "arch")
                 else {"arch": "wavehax", "ch": a.ch, "L": a.layers})
    arch_args.update({"front": a.front, "prior": a.prior, "hfw": a.hfw,
                      "gainaug": a.gainaug, "melframe": a.melframe,
                      "melnfft": a.melnfft, "tag": a.tag})
    RUN_DIR = ROOT / "results" / a.tag
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # ⚠ TEST は `safe_score` = **PESQ**（`rddsp_loop.score_one`）。**高いほど良い**。
    #    step 0 で out が零初期化 ⇒ S = P ⇒ TEST == prior。以後の gain = TEST − prior。
    best = (-1e9, 0)
    step0 = 0
    if a.resume and a.ft_lr > 0:
        # fine-tune: net/disc だけ読み、opt は固定 LR で新規（OneCycle の再加熱も
        # 終端 LR≈0 の凍結も両方避ける。train_vc_g の --ft と同じ判断）
        rk = torch.load(a.resume, map_location=dev)
        net.load_state_dict(rk["net"])
        if a.vresid and hasattr(net, "out"):
            nn.init.zeros_(net.out.weight)
            nn.init.zeros_(net.out.bias)
            print("  R0: out 層を再ゼロ初期化（S=P から開始）", flush=True)
        if disc is not None and rk.get("disc"):
            disc.load_state_dict(rk["disc"])
        opt = torch.optim.AdamW(net.parameters(), lr=a.ft_lr, betas=(0.8, 0.99),
                                weight_decay=1e-2)
        if disc is not None:
            dopt = torch.optim.AdamW(disc.parameters(), lr=a.ft_lr,
                                     betas=(0.8, 0.99), weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda _s: 1.0)
        print(f"  fine-tune: 固定 LR {a.ft_lr}（{a.resume}）", flush=True)
    elif a.resume:
        rk = torch.load(a.resume, map_location=dev)
        net.load_state_dict(rk["net"])
        opt.load_state_dict(rk["opt"])
        sch.load_state_dict(rk["sch"])
        # ⚠ OneCycleLR は state に total_steps を含む。--steps を伸ばして resume すると
        #   ckpt の total_steps が復元され、その先で ValueError になる（実測）。
        #   伸ばした先の値で上書きする（lr スケジュールは残り区間で継続）。
        if getattr(sch, "total_steps", None) != a.steps:
            sch.total_steps = a.steps
        if disc is not None and rk.get("disc"):
            disc.load_state_dict(rk["disc"])
            dopt.load_state_dict(rk["dopt"])
        torch.set_rng_state(rk["rng"].cpu() if hasattr(rk["rng"], "cpu")
                            else rk["rng"])
        step0 = int(rk["step"])
        print(f"  resume: step {step0} から（{a.resume}）", flush=True)

    te, pr, _ = evaluate(net, items_ev, dev)
    print(f"  step {0:6d}  TEST {te:.4f}  prior {pr:.4f}", flush=True)
    nseg = (a.crop - 1) * HOP
    t0 = time.time()
    for step in range(step0 + 1, a.steps + 1):
        F, TG, PB = [], [], []
        for _ in range(a.batch):
            it = to_gpu(stream.draw(), dev, W)
            pw = PRIOR_FN(it, g, dev, float(it["w"].std()))
            if SHIP:
                it["_P"] = pw
            f, P = FEATS(it) if SHIP else feats(it, pw)
            T = P.shape[-1]
            if T <= a.crop + CTX_N + 8:
                continue
            s = int(torch.randint(CTX_N + 4, T - a.crop - 4, (1,)))
            if a.arch == "v1d":
                F.append(f[:, s - CTX_N: s + a.crop])
                PB.append(P[:, s - CTX_N: s + a.crop])
            else:
                F.append(f[:, :, s - CTX_N: s + a.crop])
                PB.append(P[:, s - CTX_N: s + a.crop])
            TG.append(it["w"][s * HOP: s * HOP + nseg])
        if len(F) < 2:
            continue
        if a.arch == "v1d":
            xb = torch.stack(F)                          # [B, cin, CTX+crop]
            Pb = torch.stack(PB)
            if a.arch in ("v2f", "v2p"):
                o = net(torch.stack(F))[:, :, :, CTX_N:]
                S = torch.complex(o[:, 0], o[:, 1])
            elif a.out_mode == "direct":
                o = net(xb)[:, :, CTX_N:]
                S = torch.complex(o[:, :net.nbin], o[:, net.nbin:])
            else:
                S = net.apply(xb, Pb)[:, :, CTX_N:]
        else:
            # ⚠ `CTX`（rddsp_gpu の 24 固定）ではなく `CTX_N`（arch ごとの実値）。
            #   混ぜると s-CTX が負になって空テンソルになる（v2f で実際に落ちた）。
            o = net(torch.stack(F))[:, :, :, CTX_N:]
            S = torch.complex(o[:, 0], o[:, 1])
            if _VRESID:
                S = S + torch.stack(PB)[:, :, CTX_N:]
        y = SF.cistft(S, nseg) if SHIP else istft(S, nseg)
        tgt = torch.stack(TG)
        ys, ts = y[:, HOP * 2: -HOP * 2], tgt[:, HOP * 2: -HOP * 2]
        loss = mrstft(ys, ts)
        if a.cplx > 0:
            # 合成格子で S を gt の STFT に直接合わせる＝**位相を教師つきにする**。
            with torch.no_grad():
                RT = torch.stack([SF.cstft(t_, SF.NFFT_S, SF.HOP_S) for t_ in tgt])
            mc = min(RT.shape[-1], S.shape[-1])
            num = (S[..., :mc] - RT[..., :mc]).abs().flatten(-2).sum(-1)
            den = RT[..., :mc].abs().flatten(-2).sum(-1).clamp(min=1e-8)
            loss = loss + a.cplx * (num / den).mean()
        if lmos is not None:
            loss = loss + a.lmos * lmos(ys, ts)
        if a.consist > 0:
            S2 = SF.cstft(y, SF.NFFT_S, SF.HOP_S) if SHIP else stft(y)
            m = min(S2.shape[-1], S.shape[-1])
            loss = loss + a.consist * (
                (S2[..., :m] - S[..., :m]).abs().pow(2).flatten(-2).sum(-1).sqrt()
                / S[..., :m].abs().pow(2).flatten(-2).sum(-1).sqrt().clamp(min=1e-8)
            ).mean()
        if disc is not None and step > a.dstart:
            dopt.zero_grad(set_to_none=True)
            yr, yg, _, _ = disc(ts[:, None], ys.detach()[:, None])
            dl = sum(((r - 1) ** 2).mean() + (gq ** 2).mean()
                     for r, gq in zip(yr, yg)) / len(yr)
            dl.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            dopt.step()
            _, yg2, fr, fg = disc(ts[:, None], ys[:, None])
            adv = sum(((gq - 1) ** 2).mean() for gq in yg2) / len(yg2)
            fmv = sum(torch.nn.functional.l1_loss(b, a_.detach())
                      for A, B in zip(fr, fg) for a_, b in zip(A, B))
            loss = loss + a.gan * adv + a.fm * fmv / max(sum(len(A) for A in fr), 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % a.every == 0 or step == a.steps:
            te, pr, d = evaluate(net, items_ev, dev)
            ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            print(f"  step {step:6d}  loss {float(loss):.4f}  TEST {te:.4f} "
                  f"({d.mean():+.4f} +-{ci:.4f})  ({time.time()-t0:.0f}s)", flush=True)
            ck = {"net": net.state_dict(), "args": arch_args, "step": step,
                  "test": te, "prior": pr, "cli": vars(a),
                  "opt": opt.state_dict(), "sch": sch.state_dict(),
                  "disc": disc.state_dict() if disc is not None else None,
                  "dopt": dopt.state_dict() if dopt is not None else None,
                  "rng": torch.get_rng_state()}
            torch.save(ck, CACHE_DIR / f"{a.tag}_ch{a.ch}_s{a.seed}.pt")
            # a0: 毎チェックポイントで last を上書き（4.6 の resume の元）
            torch.save(ck, RUN_DIR / f"{a.tag}_last.pt")
            if te > best[0]:
                best = (te, step)
                torch.save(ck, RUN_DIR / f"{a.tag}_best.pt")
            # 規則 7: 中間 step のスナップショットを残す（最終 1 点で判定しない）
            if a.snap and step % a.snap == 0:
                torch.save(ck, RUN_DIR / f"{a.tag}_s{step}.pt")
    te, pr, d = evaluate(net, items_ev, dev)
    ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    print(f"\n{a.tag}: TEST {te:.4f}  prior {pr:.4f}  gain {d.mean():+.4f} "
          f"+-{ci:.4f} (t-CI, n={len(d)})  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
