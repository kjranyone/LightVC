"""出荷 front-end: mel / f0 / NHV 事前分布を、発話全体の統計ゼロ・先読みゼロで作る。

`ship_check.py` が現行 front-end を FAIL にした（f0 解析が実音声プローブで 1882 ms ＝
発話中央値が全フレームに伝播）。ここはその PASS 版で、`rddsp.py` は触らない
（既存の測定値と比較可能なまま残す）。

直したもの:

    centered STFT (17.2 ms)      -> 左寄せ (0 ms)
    voi / voi.median()           -> 較正済み固定閾値 VOI_ABS
    K = 22050 / 発話中央値f0      -> 固定 KMAX + 毎サンプル masking (k*f0 < Nyquist)
    H / H.amax()                 -> 固定基準 MEL_REF（H が絶対レベルを運ぶ）
    level = 発話の std           -> 廃止（同上）

設計をひとつ変えた。事前分布は波形を経由せず**スペクトルのまま返す**:

    旧  e -> stft -> xH -> istft -> pw -> stft -> P     (OLA 2 段)
    新  e -> stft -> xH ------------------------> P     (OLA 1 段)

ネットが要るのは P だけなので往復は冗長だった。これで OLA が 1 段（8.7 ms）減り、
`H.amax()` と `w.std()` も同時に消える。

較正（`female-dataset` 60 話者 41,400 フレーム, 2026-08-08）:
    voi 生値の中央値 1.9174 -> 現行の判定 voi/median*0.5 > 0.3 は voi > 1.1505 と同値
    causal log-mel(n_fft1024) の max 1.490 / p99.9 0.578
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from causal_mel import causal_mel

NFFT_A, HOP_A = 1024, R.HOP          # 解析（mel / f0）: 左寄せ, 先読み 0
NFFT_S, HOP_S = 512, 128             # 合成グリッド（ボコーダ）
N_MEL = 80
F0_MIN, F0_MAX = 60.0, 600.0
KMAX = int((R.SR / 2) // F0_MIN)     # 367。実効倍音数は毎サンプル masking で決まる
VOI_ABS = 1.1505                     # 較正済み。発話中央値正規化と同じ動作点
PRE_VOICED_HZ = 50.0                 # 初の有声フレーム前の固定 f0（旧版の実効値）
MEL_REF = -1.0254                    # H = exp(mel_lin - MEL_REF)。**再較正 2026-08-12**
# ⚠ 旧値 1.49 は別の格子（nfft_S / mel bank）で較正されたまま残っていた。
#   実測: 52 発話で prior 振幅が gt に対し **21.85 dB 小さい**（alpha 中央値 12.372、
#   log(alpha) の sd 0.0842 ＝ 発話変動ではなく定数ずれ）。
#   PESQ は score_one が RMS 整合するので写らないが、**学習は壊れる**——
#   mrstft の log(Y+1e-5) 項が 12 倍小さい Y で床に張り付き、開始時の勾配を潰す。
#   ⚠ ここを動かすとネットの入力分布が変わる＝重みの流用不可（CLAUDE.md）。
#   ⚠ Rust 側 crates/lightvc-core/src/ship_front.rs の MEL_REF と**必ず同時に直す**。


def _win(n: int, dev) -> torch.Tensor:
    return torch.hann_window(n, device=dev)


def cstft(x: torch.Tensor, nfft: int = NFFT_A, hop: int = HOP_A):
    """左寄せ STFT。フレーム t の最右サンプル = t*hop + hop - 1、先読み 0。"""
    # ⚠ **零詰め**（2.1 の作るもの #9）。`reflect` は t=0 のフレームが x[1:nfft-hop+1]
    #   を読む＝**先読み 768**。製品のブロック実行系は起動時に実サンプルだけを
    #   解析する（2.4a-2 状態 7・8）ので、学習側も同じ頭にしないと分布が食い違う。
    #   実測: 差が出るのは解析フレーム 0〜2 のみ（crop は 11 以降からしか始まらない）。
    xp = torch.nn.functional.pad(x[..., None, :], (nfft - hop, 0))[..., 0, :]
    # 右側 0 詰め。未来のサンプルではなく「まだ出せない末尾」なので先読みにならない。
    # (nfft-hop) 余分に取るのは、重畳の立ち下がり（窓和が 0 に落ちて 0 除算で暴れる
    # 区間）を [:n] の外へ押し出すため。n が hop の倍数ちょうどのとき、これが無いと
    # 末尾 384 サンプルが立ち下がりに当たって往復誤差が -142dB から -44.8dB に落ちる。
    q = ((-x.shape[-1]) % hop) + (nfft - hop)
    xp = torch.nn.functional.pad(xp, (0, q))
    return torch.stft(xp, nfft, hop, nfft, _win(nfft, x.device), center=False,
                      return_complex=True)


def cistft(S: torch.Tensor, n: int, nfft: int = NFFT_S, hop: int = HOP_S):
    """左寄せ iSTFT。固有遅延 = nfft - hop（重畳が閉じるまで）で、先読みではない。

    `torch.istft` は center=False だと端で窓和が 0 に落ちるのを拒否する。ここで捨てる
    立ち上がり区間がまさにそれなので、重畳加算を自前で書く（fold は微分可能）。"""
    dev = S.device
    w = _win(nfft, dev)
    x = torch.fft.irfft(S, nfft, dim=-2) * w[:, None]
    B, T = x.shape[:-2], x.shape[-1]
    L = (T - 1) * hop + nfft
    f = torch.nn.functional.fold
    y = f(x.reshape(-1, nfft, T), (1, L), (1, nfft), stride=(1, hop))[:, 0, 0]
    ws = f((w * w)[None, :, None].expand(1, nfft, T), (1, L), (1, nfft),
           stride=(1, hop))[0, 0, 0]
    y = (y / ws.clamp(min=1e-8))[..., nfft - hop:]
    if y.shape[-1] < n:
        y = torch.nn.functional.pad(y, (0, n - y.shape[-1]))
    return y[..., :n].reshape(*B, n)


def n_frames(n: int, hop: int = HOP_S, nfft: int = NFFT_S) -> int:
    """cstft が実際に出すフレーム数。"""
    return (n + ((-n) % hop) + (nfft - hop)) // hop


def causal_f0(x: torch.Tensor):
    """harmonic_sum_f0 と同じ二葉スコアを、左寄せ解析と固定閾値で。

    負の葉（k+0.5 の位置を減点）は残す。これが弱基音でのオクターブ誤りを止めている
    部分で、[[f0-octave-weak-fundamental]] の対処そのもの。"""
    mag = cstft(x, NFFT_A, HOP_A).abs()
    nb, T = mag.shape
    binhz = R.SR / NFFT_A
    ncand = int(math.log2(F0_MAX / F0_MIN) * 1200 / 10) + 1
    cand = F0_MIN * 2 ** (torch.arange(ncand, device=x.device,
                                       dtype=torch.float32) * 10 / 1200)

    def take(f):
        ok = (f < R.SR / 2 - binhz) & (f > 0)
        b = (f / binhz).clamp(0, nb - 2)
        lo = b.long()
        fr = (b - lo)[:, None]
        return (mag[lo] * (1 - fr) + mag[lo + 1] * fr) * ok[:, None].float()

    score = torch.zeros(ncand, T, device=x.device)
    for k in range(1, 21):
        w = 1.0 / math.sqrt(k)
        score = score + w * take(cand * k) - 0.5 * w * take(cand * (k + 0.5))
    # near-tie ヒステリシス（走行状態のみ・発話統計なし）: 素の argmax は
    # オクターブ級の飛びが毎分 216 回（クリーン音声でも同率・2026-08-21 実測）で
    # 励起の瞬間移動＝クリック連発の主犯。飛びの機構は「オクターブ候補が僅差で
    # フリップする」ことなので、**上位 δ=8% 以内の僅差のときだけ**前値に最も近い
    # 候補を選ぶ。明確な証拠（>8% 差）は素通し＝誤レジスタへのロックを防ぐ
    # （距離減点方式 λ=0.4/oct は harvest 一致 0.68→0.09 に崩壊した負の結果）。
    log2c = torch.log2(cand)
    idx = torch.empty(T, dtype=torch.long, device=x.device)
    prev: float | None = None
    smax = score.amax(0).clamp(min=1e-8)
    vden = mag.sum(0) / math.sqrt(nb) + R.EPS    # voi の分母（フレーム内演算）
    for ti in range(T):
        sc = score[:, ti]
        bi = int(sc.argmax())
        if prev is not None:
            tie = sc >= smax[ti] * 0.85
            if int(tie.sum()) > 1:
                cands = torch.nonzero(tie, as_tuple=False)[:, 0]
                bi = int(cands[(log2c[cands] - prev).abs().argmin()])
        idx[ti] = bi
        # prev は有声判定フレームでのみ更新（無声雑音に引きずられない）
        if float(sc[bi] / vden[ti]) > VOI_ABS:
            prev = float(log2c[bi])
    f0 = cand[idx]
    voi = score.gather(0, idx[None])[0] / (mag.sum(0) / math.sqrt(nb) + R.EPS)
    f0 = _causal_median(f0, 5)
    return torch.where(voi > VOI_ABS, f0, torch.zeros_like(f0)), voi


def _causal_median(v: torch.Tensor, k: int) -> torch.Tensor:
    p = torch.nn.functional.pad(v[None, None], (k - 1, 0), mode="replicate")[0, 0]
    return p.unfold(0, k, 1).median(-1).values


def mel(x: torch.Tensor) -> torch.Tensor:
    return causal_mel(x, n_fft=NFFT_A, hop=HOP_A, num_mels=N_MEL, sr=R.SR)[0]


def to_frames(m: torch.Tensor, T: int) -> torch.Tensor:
    """解析レート [.., Tm] -> 合成レート [.., T]。局所写像でなければならない。

    `arange(T) * (Tm-1) / (T-1)` という書き方は発話長で伸縮する全体スケーリングで、
    発話が終わるまで添字が決まらない。実測 6.30 ms の先読みはこれが原因だった。
    さらに t*HOP_S // HOP_A でも足りない。解析フレーム j が使えるのは j*HOP_A+HOP_A-1
    を出し終えた後で、合成フレーム t が閉じるのは t*HOP_S+HOP_S-1。よって
    j = (t*HOP_S + HOP_S - HOP_A) // HOP_A が使ってよい最新（実測 2.70 ms がこの差）。"""
    i = ((torch.arange(T, device=m.device) * HOP_S + HOP_S - HOP_A)
         // HOP_A).clamp(min=0)
    return m[..., i.clamp(max=m.shape[-1] - 1)]


def frame_upsample_causal(v: torch.Tensor, n: int, hop: int) -> torch.Tensor:
    """フレーム -> サンプル。過去 2 フレームだけで補間する。

    `R.frame_upsample` はサンプル n をフレーム i と **i+1** で補間しており、i+1 は
    まだ窓が閉じていない未来のフレーム。実測 6.30 ms の先読みはここだった。
    フレーム j が使えるのは j*hop + hop - 1 を出し終えた後なので、サンプル n で
    使ってよい最新は j = floor((n+1)/hop) - 1。その j と j-1 で補間する。
    結果として f0 が約 1 フレーム遅れるが、これは遅延であって先読みではない。"""
    t = (torch.arange(n, device=v.device, dtype=torch.float64) + 1.0) / hop - 2.0
    i = t.floor().clamp(0, v.shape[-1] - 2)
    fr = (t - i).clamp(0.0, 1.0).to(v.dtype)
    j = i.long()
    return v[..., j] * (1 - fr) + v[..., j + 1] * fr


def phase_of(f0: torch.Tensor, n: int) -> torch.Tensor:
    """f0[T] -> 位相[n]。cumsum は走行アキュムレータなのでストリームでも同じ値。"""
    f0u = frame_upsample_causal(_fill(f0).double(), n, HOP_A).clamp(min=0.0)
    return torch.remainder(2 * math.pi * torch.cumsum(f0u, 0) / R.SR,
                           2 * math.pi).float(), f0u.float()


def _fill(f0: torch.Tensor) -> torch.Tensor:
    """直近の有声 f0 で埋める。**発話全体の統計を使わない**（2.1 の作るもの #7）。

    旧版は `ok.any()`（発話全体）で「1 つも有声が無い」場合を判定していた。
    ストリームでは発話が終わるまで判定できないので規則 2 違反。
    ∴ **走行状態だけで決める**: 初の有声フレームより前は固定定数 `PRE_VOICED_HZ`。

    ⚠ `PRE_VOICED_HZ = 50.0` は設計値ではなく**旧版の実効値**（旧版は index 0 の
    unvoiced 0 Hz を拾って `clamp(min=50)` していた）。ここを変えると位相
    アキュムレータがずれてネットの入力分布が変わる＝学習し直しになるので、
    **意図的に旧値を保存している**。実測: 有声が 1 つでもある発話では旧版とビット一致。
    """
    v = f0.clone()
    ok = v > 50
    i = torch.arange(len(v), device=v.device, dtype=v.dtype)
    iv = torch.where(ok, i, torch.full_like(i, -1.0)).cummax(0).values
    out = v[iv.clamp(min=0).long()].clamp(min=50.0)
    return torch.where(iv >= 0, out, torch.full_like(v, PRE_VOICED_HZ))


def nhv_spec(mel_lin: torch.Tensor, f0: torch.Tensor, n: int,
             gen=None, noise_mix: float = 0.3, T: int = None):
    """NHV 事前分布を**スペクトルのまま**返す。波形を作らないので OLA を通らない。

    励起 = インパルス列（相対位相ゼロ）+ 白色雑音。倍音は k*f0[n] < Nyquist で
    毎サンプル masking するので、発話中央値から K を決める必要がない。実効倍音数で
    正規化して、f0 が動いてもレベルが動かないようにする。

    mel_lin: [NBIN, T] 線形周波数軸に写した log-mel。H = exp(mel_lin - MEL_REF) が
    そのまま絶対レベルを運ぶので、包絡の amax 正規化も発話 RMS も要らない。"""
    dev = mel_lin.device
    phi, f0u = phase_of(f0, n)
    imp = torch.zeros(n, device=dev)
    cnt = torch.zeros(n, device=dev)
    nyq = R.SR / 2
    for s0 in range(0, KMAX, 32):
        kk = torch.arange(s0 + 1, min(s0 + 33, KMAX + 1), device=dev,
                          dtype=torch.float32)[:, None]
        m = (kk * f0u[None] < nyq).float()
        imp = imp + (torch.cos(kk * phi[None]) * m).sum(0)
        cnt = cnt + m.sum(0)
    imp = imp / cnt.clamp(min=1.0).sqrt()
    z = torch.randn(n, generator=gen, device=dev)
    E = cstft((1 - noise_mix) * imp + noise_mix * z, NFFT_S, HOP_S)
    Tn = T or E.shape[-1]
    H = (mel_lin[:, :Tn] - MEL_REF).exp()
    m = min(E.shape[-1], H.shape[-1], Tn)
    return E[:, :m] * H[:, :m]

V_MEL_ADAPT = math.log(32768.0)
