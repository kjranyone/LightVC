"""Artic-A2 Step 0 の DSP 部品(自前実装・numpy/numba)。設計: current/artic_a2.md §3-§4。

因果時変線形予測:
  プリエンファシス(μ=0.97 固定) → 時刻 kH で過去 W サンプルだけの Hann 窓自己相関(lag 窓・白色補正)
  → Levinson → 反射係数 k → LAR=log((1-k)/(1+k))。
  サンプル n∈[kH,(k+1)H) の係数は LAR_{k-1}→LAR_k をサブブロック B 毎に線形補間(因果・先読み0)。
  解析(FIR)と合成(全極 IIR)は同じ係数列の直接形 I なので厳密に可逆。LAR の実数値は常に |k|<1(安定)に写る。
残差の TD-PSOLA(ピッチだけを変え時間は変えない): 声門閉鎖点=残差のピーク列を f0 で追跡し、
  半窓 L=min(入力周期, 出力周期) の Hann で切り出して出力周期間隔に重ねる。無声区間は素通し。
"""
from __future__ import annotations

import numpy as np
from numba import njit

SR = 48000
H = 120
W = 960
B = 24
MU = 0.97


def preemph(x: np.ndarray, mu: float = MU) -> np.ndarray:
    y = x.copy()
    y[1:] = x[1:] - mu * x[:-1]
    return y


@njit(cache=True)
def deemph(y: np.ndarray, mu: float) -> np.ndarray:
    out = np.empty_like(y)
    prev = y.dtype.type(0.0)
    m = y.dtype.type(mu)
    for n in range(y.shape[0]):
        prev = y[n] + m * prev
        out[n] = prev
    return out


def levinson(r: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """r [K,order+1] → (a [K,order+1] a0=1, k [K,order])。A(z)=1+Σa_i z^-i。無音フレームは k=0。"""
    K = r.shape[0]
    a = np.zeros((K, order + 1))
    a[:, 0] = 1.0
    ks = np.zeros((K, order))
    err = r[:, 0].copy()
    live = err > 1e-12
    for i in range(1, order + 1):
        acc = r[:, i] + (a[:, 1:i] * r[:, i - 1:0:-1]).sum(1) if i > 1 else r[:, i].copy()
        k = np.where(live, -acc / np.where(live, err, 1.0), 0.0)
        k = np.clip(k, -0.99999, 0.99999)
        prev = a[:, 1:i].copy()
        a[:, 1:i] = prev + k[:, None] * prev[:, ::-1]
        a[:, i] = k
        ks[:, i - 1] = k
        err = err * (1.0 - k * k)
        live = live & (err > 1e-12)
    return a, ks


def k_to_lar(k: np.ndarray) -> np.ndarray:
    return np.log((1.0 - k) / (1.0 + k))


def lar_to_k(lar: np.ndarray) -> np.ndarray:
    return -np.tanh(lar / 2.0)


def k_to_a(k: np.ndarray) -> np.ndarray:
    """k [...,p] → a [...,p](a0=1 を除く)。step-up 再帰。"""
    p = k.shape[-1]
    a = np.zeros(k.shape[:-1] + (p,))
    for i in range(p):
        ki = k[..., i]
        if i > 0:
            prev = a[..., :i].copy()
            a[..., :i] = prev + ki[..., None] * prev[..., ::-1]
        a[..., i] = ki
    return a


def lar_frames(x: np.ndarray, order: int, lag_bw: float = 60.0, wnc: float = 1e-4, la: int = 0) -> np.ndarray:
    """x(48k・プリエンファシス前) → LAR [K,order]。フレーム k は xp[kH-W+la : kH+la](la=先読み・H の倍数)。K=ceil(N/H)。
    la=0 が因果。la=W/2 は窓の中心がフレーム時刻に来る(先読み W/2)。末尾の足りない分は 0 詰め。"""
    assert la % H == 0
    xp = preemph(x.astype(np.float64))
    N = len(xp)
    K = (N + H - 1) // H
    pad = np.concatenate([np.zeros(W - la), xp, np.zeros(H + la)])
    frames = np.lib.stride_tricks.sliding_window_view(pad, W)[0:K * H:H]
    win = np.hanning(W)
    F = np.fft.rfft(frames * win, n=2048)
    r = np.fft.irfft(np.abs(F) ** 2, n=2048)[:, :order + 1]
    i = np.arange(order + 1)
    r = r * np.exp(-0.5 * (2 * np.pi * lag_bw * i / SR) ** 2)
    r[:, 0] *= 1.0 + wnc
    _, ks = levinson(r, order)
    return k_to_lar(ks)


def coef_schedule(lar: np.ndarray, n_samples: int) -> np.ndarray:
    """LAR [K,p] → サブブロック毎の a [n_sub,p]。hop k の中は LAR_{k-1}→LAR_k を線形補間(因果)。"""
    K, p = lar.shape
    per = H // B
    prev = np.concatenate([lar[:1], lar[:-1]], 0)
    w = (np.arange(per) + 1.0) / per
    sub = prev[:, None, :] + (lar - prev)[:, None, :] * w[None, :, None]
    sub = sub.reshape(K * per, p)[: (n_samples + B - 1) // B]
    return k_to_a(lar_to_k(sub))


@njit(cache=True)
def analysis_fir(xp: np.ndarray, a_sub: np.ndarray, blk: int) -> np.ndarray:
    N = xp.shape[0]
    p = a_sub.shape[1]
    e = np.empty_like(xp)
    for n in range(N):
        sb = n // blk
        acc = xp[n]
        for i in range(1, p + 1):
            if n - i >= 0:
                acc += a_sub[sb, i - 1] * xp[n - i]
        e[n] = acc
    return e


@njit(cache=True)
def synthesis_iir(e: np.ndarray, a_sub: np.ndarray, blk: int) -> np.ndarray:
    N = e.shape[0]
    p = a_sub.shape[1]
    y = np.empty_like(e)
    for n in range(N):
        sb = n // blk
        acc = e[n]
        for i in range(1, p + 1):
            if n - i >= 0:
                acc -= a_sub[sb, i - 1] * y[n - i]
        y[n] = acc
    return y


def analyze(x: np.ndarray, order: int, dtype=np.float64, la: int = 0):
    """x → (lar [K,p], a_sub, e 残差)。e は dtype で計算。la は解析窓の先読み(lar_frames 参照)。"""
    lar = lar_frames(x, order, la=la)
    a_sub = coef_schedule(lar, len(x)).astype(dtype)
    e = analysis_fir(preemph(x.astype(np.float64)).astype(dtype), a_sub, B)
    return lar, a_sub, e


def synthesize(e: np.ndarray, a_sub: np.ndarray) -> np.ndarray:
    return deemph(synthesis_iir(e, a_sub.astype(e.dtype), B), MU)


def lar_lowpass(lar: np.ndarray, fc: float, causal: bool = True, order: int = 2) -> np.ndarray:
    from scipy.signal import butter, lfilter, filtfilt
    fs = SR / H
    b, a = butter(order, fc / (fs / 2))
    if causal:
        zi = None
        from scipy.signal import lfilter_zi
        zi = lfilter_zi(b, a)[:, None] * lar[:1]
        return lfilter(b, a, lar, axis=0, zi=zi)[0]
    return filtfilt(b, a, lar, axis=0)


def pitch_marks(e: np.ndarray, f0_t: np.ndarray, f0_v: np.ndarray, search: float = 0.2) -> list[np.ndarray]:
    """残差 e と f0 列(時刻 f0_t[s], 値 f0_v[Hz], 0=無声) → 有声区間ごとの声門閉鎖点(サンプル位置)列。"""
    N = len(e)
    env = np.convolve(np.abs(e), np.ones(3) / 3, mode="same")
    f0s = np.interp(np.arange(N) / SR, f0_t, f0_v)
    voiced = np.interp(np.arange(N) / SR, f0_t, (f0_v > 0).astype(float)) > 0.5
    segs = []
    n = 0
    while n < N:
        if not voiced[n]:
            n += 1
            continue
        st = n
        while n < N and voiced[n]:
            n += 1
        en = n
        marks = []
        P = int(SR / max(f0s[st], 50.0))
        lo, hi = st, min(en, st + P)
        if hi - lo < 4:
            continue
        m = lo + int(np.argmax(env[lo:hi]))
        while m < en:
            marks.append(m)
            P = int(SR / max(f0s[m], 50.0))
            c = m + P
            lo, hi = int(c - search * P), int(c + search * P)
            if hi >= en or lo <= m:
                break
            m = lo + int(np.argmax(env[lo:hi]))
        if len(marks) >= 2:
            segs.append(np.array(marks))
    return segs


def psola_shift(e: np.ndarray, segs: list[np.ndarray], ratio: float, xfade: int = 96) -> tuple[np.ndarray, list[np.ndarray]]:
    """残差のピッチだけ ratio 倍(時間は不変)。返り値=(新しい残差, 新しいマーク列)。"""
    N = len(e)
    out = e.copy()
    new_segs = []
    for marks in segs:
        st = max(0, marks[0] - (marks[1] - marks[0]))
        en = min(N, marks[-1] + (marks[-1] - marks[-2]))
        periods = np.diff(marks).astype(float)
        seg_out = np.zeros(en - st)
        s = float(marks[0])
        outm = []
        while s < marks[-1]:
            i = int(np.argmin(np.abs(marks - s)))
            Pin = periods[min(i, len(periods) - 1)]
            Pout = Pin / ratio
            L = int(max(4, min(Pin, Pout)))
            t = marks[i]
            a0, a1 = t - L, t + L
            if a0 < 0 or a1 > N:
                s += Pout
                continue
            w = np.hanning(2 * L + 1)[:-1]
            piece = e[a0:a1] * w
            c = int(round(s)) - st
            b0, b1 = c - L, c + L
            p0, p1 = max(0, -b0), 2 * L - max(0, b1 - (en - st))
            if p1 > p0:
                seg_out[max(0, b0):max(0, b0) + (p1 - p0)] += piece[p0:p1]
            outm.append(int(round(s)))
            s += Pout
        ramp = np.ones(en - st)
        k = min(xfade, (en - st) // 4)
        if k > 0:
            ramp[:k] = np.linspace(0, 1, k)
            ramp[-k:] = np.linspace(1, 0, k)
        out[st:en] = ramp * seg_out + (1 - ramp) * e[st:en]
        new_segs.append(np.array(outm))
    return out, new_segs


F0_NFFT = 1024
F0_HOP = 240
F0_MIN, F0_MAX = 60.0, 1000.0


def causal_f0(x: np.ndarray, voi_abs: float = 1.15) -> tuple[np.ndarray, np.ndarray]:
    """ship_front.causal_f0 の 48k 版(左寄せ STFT の調波和・負の葉・near-tie ヒステリシス・因果メディアン5)。
    フレーム j は x[:j*F0_HOP+F0_HOP] だけで決まる(先読み0)。返り値 (f0 [T](0=無声), voi [T])。"""
    N = len(x)
    T = (N + F0_HOP - 1) // F0_HOP
    pad = np.concatenate([np.zeros(F0_NFFT - F0_HOP), x.astype(np.float64), np.zeros(F0_HOP)])
    fr = np.lib.stride_tricks.sliding_window_view(pad, F0_NFFT)[0:T * F0_HOP:F0_HOP]
    mag = np.abs(np.fft.rfft(fr * np.hanning(F0_NFFT), axis=1)).T
    nb = mag.shape[0]
    binhz = SR / F0_NFFT
    ncand = int(np.log2(F0_MAX / F0_MIN) * 120) + 1
    cand = F0_MIN * 2 ** (np.arange(ncand) / 120.0)
    score = np.zeros((ncand, T))
    for k in range(1, 21):
        w = 1.0 / np.sqrt(k)
        for mult, sgn in ((k, 1.0), (k + 0.5, -0.5)):
            f = cand * mult
            ok = (f < SR / 2 - binhz)
            b = np.clip(f / binhz, 0, nb - 2)
            lo = b.astype(int)
            fr_ = (b - lo)[:, None]
            score += sgn * w * (mag[lo] * (1 - fr_) + mag[lo + 1] * fr_) * ok[:, None]
    log2c = np.log2(cand)
    vden = mag.sum(0) / np.sqrt(nb) + 1e-8
    smax = np.maximum(score.max(0), 1e-8)
    idx = np.empty(T, dtype=int)
    prev = None
    for t in range(T):
        sc = score[:, t]
        bi = int(sc.argmax())
        if prev is not None:
            tie = np.nonzero(sc >= smax[t] * 0.85)[0]
            if len(tie) > 1:
                bi = int(tie[np.abs(log2c[tie] - prev).argmin()])
        idx[t] = bi
        if sc[bi] / vden[t] > voi_abs:
            prev = log2c[bi]
    f0 = cand[idx]
    voi = score[idx, np.arange(T)] / vden
    f0p = np.concatenate([np.full(4, f0[0]), f0])
    f0m = np.median(np.lib.stride_tricks.sliding_window_view(f0p, 5), axis=1)
    return np.where(voi > voi_abs, f0m, 0.0), voi


def f0_known(n: int, f0: np.ndarray) -> float:
    """時刻 n で既知の最新 f0 フレーム(フレーム j は j*F0_HOP+F0_HOP-1 で確定)。"""
    j = (n - F0_HOP + 1) // F0_HOP
    return float(f0[min(j, len(f0) - 1)]) if j >= 0 else 0.0


def causal_marks(e: np.ndarray, f0: np.ndarray, s: float = 0.25,
                 f0_cont: np.ndarray | None = None, back_periods: float = 3.0) -> list[list[tuple[int, int]]]:
    """残差 e と因果 f0 → 有声区間ごとの [(マーク位置, 確定時刻)]。
    env[n] は |e[n-2..n]| の平均(因果)。予測位置 c=m+P の窓 [c-sP, c+sP] が揃った時刻に確定。
    区間の開始は f0(厳しい有声判定)、継続は f0_cont(緩い有声判定)で判断する(ヒステリシス・走行状態のみ)。
    有声が既知になった時刻 n で、直前 1 周期 [n-P, n] の最大をアンカーにし、過去へ back_periods 周期ぶんマークを遡る(確定時刻は n)。"""
    if f0_cont is None:
        f0_cont = f0
    N = len(e)
    a = np.abs(e)
    env = a.copy()
    env[1:] += a[:-1]
    env[2:] += a[:-2]
    runs = []
    n = 0
    last = -10 ** 9
    while n < N:
        fz = f0_known(n, f0)
        if fz <= 0:
            j = (n - F0_HOP + 1) // F0_HOP
            n = max(n + 1, (j + 1) * F0_HOP + F0_HOP - 1)
            continue
        P = SR / fz
        floor_ = max(0, last + int(0.5 * P))
        w0 = max(floor_, n - int(P) + 1)
        if w0 > n:
            n = w0
            continue
        m0 = w0 + int(np.argmax(env[w0:n + 1]))
        run = [(m0, n)]
        while back_periods > 0:
            c = run[0][0] - P
            lo, hi = int(c - s * P), int(c + s * P) + 1
            if lo < max(floor_, n - back_periods * P):
                break
            run.insert(0, (lo + int(np.argmax(env[lo:hi])), n))
        while True:
            fz = f0_known(run[-1][1], f0_cont)
            if fz <= 0:
                break
            P = SR / fz
            c = run[-1][0] + P
            lo, hi = int(c - s * P), int(c + s * P) + 1
            if hi > N or lo <= run[-1][0]:
                break
            if f0_known(hi - 1, f0_cont) <= 0:
                break
            run.append((lo + int(np.argmax(env[lo:hi])), hi - 1))
        if len(run) >= 2:
            runs.append(run)
        last = run[-1][0]
        n = max(run[-1][1] + 1, last + int(0.5 * P))
    return runs


def causal_psola(e: np.ndarray, runs: list, ratio: float, D: int, s: float = 0.25,
                 f0: np.ndarray | None = None, wait: int = 24) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """上げ方向(ratio≥1)の残差 PSOLA。出力サンプル n は壁時計 n+D で書く前提で、出力マーク s_j に置く粒の入力マーク t は
    確定時刻 ≤ s_j−L+D(粒を書き始める前に既知)かつ t ≤ s_j+D、かつ s_j からの距離が許容内(後方 P+max(0,L+2sP−D)・前方 P/2)。
    候補が無ければ wait サンプル待つ(区間は捨てない)。素通し残差との重み w は置いた粒の Hann 窓の和(上限 1)。"""
    assert ratio >= 1.0
    N = len(e)
    acc = np.zeros(N)
    wsum = np.zeros(N)
    out_segs = []
    lag = []
    for run in runs:
        tm = np.array([m for m, _ in run])
        cf = np.array([c for _, c in run])
        pin = np.empty(len(tm))
        pin[1:] = np.diff(tm)
        pin[0] = SR / max(f0_known(int(cf[0]), f0), F0_MIN) if f0 is not None else pin[1]
        Lg = np.maximum(4, np.floor(pin / ratio)).astype(int)
        back = pin + np.maximum(0.0, Lg + 2 * s * pin - D)
        sj = float(max(tm[0], cf[0] + Lg[0] - D))
        placed = []
        while sj <= tm[-1] + pin[-1]:
            ok = ((cf <= np.floor(sj) - Lg + D) & (tm <= sj + D)
                  & (tm - sj >= -back) & (tm - sj <= pin / 2))
            if not ok.any():
                sj += wait
                continue
            ii = np.nonzero(ok)[0]
            i = int(ii[np.argmin(np.abs(tm[ii] - sj))])
            L = int(Lg[i])
            c = int(np.floor(sj))
            t = int(tm[i])
            if c - L < 0 or c + L > N or t - L < 0 or t + L > N:
                break
            win = np.hanning(2 * L + 1)[:-1]
            acc[c - L:c + L] += e[t - L:t + L] * win
            wsum[c - L:c + L] += win
            placed.append(c)
            lag.append(c - t)
            sj += pin[i] / ratio
        if placed:
            out_segs.append(np.array(placed))
    w = np.minimum(wsum, 1.0)
    out = acc + (1.0 - w) * e
    info = {"grain_lag_ms_median": float(np.median(lag)) / SR * 1000 if lag else None,
            "grain_lag_ms_p95": float(np.percentile(lag, 95)) / SR * 1000 if lag else None}
    return out, out_segs, info


YIN_W = 1536


def causal_yin(x: np.ndarray, thr: float = 0.15, voi_max: float = 0.35, floor_db: float = -60.0,
               fmin: float = F0_MIN, fmax: float = F0_MAX) -> tuple[np.ndarray, np.ndarray]:
    """左寄せ YIN(累積平均正規化差分)。フレーム j は x[j*F0_HOP+F0_HOP-YIN_W : j*F0_HOP+F0_HOP](先読み0)。
    有声 = 最小 d' < voi_max かつ フレーム RMS > floor_db(固定・発話統計なし)。返り値 (f0 [T](0=無声), d'最小 [T])。"""
    N = len(x)
    T = (N + F0_HOP - 1) // F0_HOP
    W = YIN_W
    pad = np.concatenate([np.zeros(W - F0_HOP), x.astype(np.float64), np.zeros(F0_HOP)])
    fr = np.lib.stride_tricks.sliding_window_view(pad, W)[0:T * F0_HOP:F0_HOP]
    tmin, tmax = int(SR / fmax), int(SR / fmin)
    nfft = 4096
    F = np.fft.rfft(fr, n=nfft, axis=1)
    r = np.fft.irfft(np.abs(F) ** 2, n=nfft, axis=1)[:, :tmax + 2]
    cs = np.concatenate([np.zeros((T, 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
    tau = np.arange(tmax + 2)
    e0 = cs[:, W - tau]
    et = cs[:, W:W + 1] - cs[:, tau]
    d = np.maximum(e0 + et - 2 * r, 0.0)
    d[:, 0] = 0.0
    cm = np.cumsum(d[:, 1:], axis=1) / np.arange(1, tmax + 2)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] / np.maximum(cm, 1e-12)
    f0 = np.zeros(T)
    dmin = np.ones(T)
    rms_db = 10 * np.log10(np.maximum(cs[:, -1] / W, 1e-20))
    seg = dn[:, tmin:tmax + 1]
    for j in range(T):
        s_ = seg[j]
        below = np.nonzero(s_ < thr)[0]
        if not len(below):
            below = np.nonzero(s_ < voi_max)[0]
        if len(below):
            k = below[0]
            while k + 1 < len(s_) and s_[k + 1] < s_[k]:
                k += 1
        else:
            k = int(np.argmin(s_))
        dmin[j] = s_[k] if k < len(s_) - 2 else 1.0
        tk = k + tmin
        if 1 <= k < len(s_) - 1:
            a, b, c = s_[k - 1], s_[k], s_[k + 1]
            den = a - 2 * b + c
            tk = tk + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
        if dmin[j] < voi_max and rms_db[j] > floor_db:
            f0[j] = SR / tk
    return f0, dmin


def warp_lar(lar: np.ndarray, alpha, order: int | None = None, nfft: int = 2048,
             lag_bw: float = 60.0, wnc: float = 1e-4) -> np.ndarray:
    """包絡の周波数伸縮(声道長の変換)。フレームごと(因果): LAR → 包絡(プリエンファシスを外す) → 目標周波数 f に
    元の f/α(f) の包絡を置く → プリエンファシスを戻す → 自己相関 → Levinson → LAR。alpha は定数か f[Hz] の関数。"""
    K, p = lar.shape
    order = order or p
    a = k_to_a(lar_to_k(lar))
    A = np.fft.rfft(np.concatenate([np.ones((K, 1)), a], 1), n=nfft, axis=1)
    f = np.fft.rfftfreq(nfft, 1 / SR)
    pre = np.abs(1 - MU * np.exp(-2j * np.pi * f / SR)) ** 2
    Pd = 1.0 / np.maximum(np.abs(A) ** 2, 1e-12) / np.maximum(pre, 1e-12)
    al = alpha(f) if callable(alpha) else np.full_like(f, float(alpha))
    src = np.clip(f / al, 0, f[-1])
    idx = src / (f[1] - f[0])
    i0 = np.clip(np.floor(idx).astype(int), 0, len(f) - 2)
    fr = idx - i0
    Pw = np.exp(np.log(Pd[:, i0]) * (1 - fr) + np.log(Pd[:, i0 + 1]) * fr) * pre
    r = np.fft.irfft(Pw, n=nfft, axis=1)[:, :order + 1]
    i = np.arange(order + 1)
    r = r * np.exp(-0.5 * (2 * np.pi * lag_bw * i / SR) ** 2)
    r[:, 0] *= 1.0 + wnc
    _, ks = levinson(r, order)
    return k_to_lar(ks)


def dsp_convert(x: np.ndarray, st: float, alpha, D: int, order: int = 24, method: str = "rrps",
                voi: float = 0.25, voi_cont: float = 0.45) -> np.ndarray:
    """学習なしの男→女変換(物理骨格): 因果 LPC 分析 → 残差を st 半音上げ(RRPS か因果 PSOLA・出力遅延 D)
    → 包絡を α で伸縮 → 全極合成。"""
    lar, a_sub, e = analyze(x, order)
    if st > 0:
        f0p, _ = causal_yin(x, voi_max=voi_cont)
        if method == "rrps":
            e, _ = rrps(e, f0p, 2 ** (st / 12), D, voiced=voiced_known(len(x), f0p, D, F0_HOP))
        else:
            f0c, _ = causal_yin(x, voi_max=voi)
            runs = causal_marks(e, f0c, f0_cont=f0p)
            e, _, _ = causal_psola(e, runs, 2 ** (st / 12), D, f0=f0c)
    if callable(alpha) or float(alpha) != 1.0:
        a_sub = coef_schedule(warp_lar(lar, alpha), len(x))
    return synthesize(e, a_sub)


@njit(cache=True)
def _sinc_read(e: np.ndarray, pos: float, half: int, fc: float) -> float:
    i0 = int(np.floor(pos))
    acc = 0.0
    for k in range(i0 - half + 1, i0 + half + 1):
        if k < 0 or k >= e.shape[0]:
            continue
        t = pos - k
        w = 0.5 + 0.5 * np.cos(np.pi * t / half) if abs(t) < half else 0.0
        x = np.pi * fc * t
        s = fc if abs(x) < 1e-9 else fc * np.sin(x) / x
        acc += e[k] * s * w
    return acc


@njit(cache=True)
def _align(e: np.ndarray, p: float, q0: float, P: float, srch: float, avail: int) -> float:
    """跳び先 q0 の近傍 ±srch·P で、直前 1 周期の波形が読み位置 p の直前 1 周期と最も似る位置(正規化相互相関)。
    比較に使う窓の右端は avail(= 壁時計で届いている最後のサンプル+1)を超えない(因果)。"""
    W = int(P)
    ip = int(p)
    if ip - W < 0 or ip > avail or W < 8:
        return q0
    a = e[ip - W:ip]
    na = np.sqrt((a * a).sum()) + 1e-12
    best, arg = -2.0, 0
    r = int(srch * P)
    found = False
    for d in range(-r, r + 1):
        iq = int(q0) + d
        if iq - W < 0 or iq > avail:
            continue
        b = e[iq - W:iq]
        c = (a * b).sum() / (na * (np.sqrt((b * b).sum()) + 1e-12))
        if c > best:
            best, arg = c, d
            found = True
    return q0 + arg if found else q0


@njit(cache=True)
def _rrps(e: np.ndarray, Pn: np.ndarray, rate: np.ndarray, D: int, half: int, xf_max: int, srch: float) -> tuple:
    """読み位置 p を許容窓 [n−Lb, n+La] に保つ。La = min(P/2, D−half−(r−1)X−srch·P)(供給限界から決まる最大の先行)、
    Lb = P − La(窓幅は常に 1 周期)。窓を外れたら周期の整数倍だけ跳び、跳び先は届いているサンプルだけで相関整列。
    D が大きければ |p−n| ≤ P/2(残差と全極フィルタのずれ最小)、D が小さければ遅れ側に寄る(PSOLA の D 依存と同じ)。"""
    N = e.shape[0]
    out = np.zeros(N)
    p = 0.0
    q = 0.0
    xf_left = 0
    xf_len = 1
    lag_sum = 0.0
    jumps = 0
    for n in range(N):
        avail = min(N, n + D + 1)
        P = Pn[n]
        r = rate[n]
        fc = min(1.0, 1.0 / r)
        if xf_left == 0:
            X = min(xf_max, int(P))
            La = min(0.5 * P, D - half - (r - 1.0) * X - srch * P)
            Lb = P - La
            d = p - n
            if d > La:
                k = int(np.ceil((d - La) / P))
                q = _align(e, p, p - k * P, P, srch, avail - half)
                if q - n > La:
                    q = n + La
                xf_left = X
                xf_len = X
                jumps += 1
            elif d < -Lb:
                k = int(np.floor((La - d) / P))
                if k >= 1:
                    q = _align(e, p, p + k * P, P, srch, avail - half)
                    if q - n > La:
                        q = n + La
                    xf_left = X
                    xf_len = X
                    jumps += 1
        a = _sinc_read(e, p, half, fc) if p >= 0 else 0.0
        if xf_left > 0:
            b = _sinc_read(e, q, half, fc) if q >= 0 else 0.0
            w = 0.5 - 0.5 * np.cos(np.pi * (xf_len - xf_left + 1) / (xf_len + 1))
            out[n] = (1 - w) * a + w * b
            q += r
            xf_left -= 1
            if xf_left == 0:
                p = q
            else:
                p += r
        else:
            out[n] = a
            p += r
        lag_sum += n - p
    return out, lag_sum / N, jumps


def voiced_known(n_samples: int, f0: np.ndarray, D: int, hop: int) -> np.ndarray:
    """出力サンプル n(壁時計 n+D)で既知の最新フレームの有声判定。"""
    n = np.arange(n_samples) + D
    j = np.clip((n - hop + 1) // hop, -1, len(f0) - 1)
    return np.where(j >= 0, f0[np.maximum(j, 0)] > 0, False)


def rrps(e: np.ndarray, f0: np.ndarray, ratio, D: int, voiced: np.ndarray | None = None,
         half: int = 8, xf_ms: float = 4.0, p0_ms: float = 5.0, hold_ms: float = 50.0, srch: float = 0.2,
         ramp_ms: float = 5.0) -> tuple[np.ndarray, dict]:
    """残差の再標本化ピッチシフト(因果・出力遅延 D)。読み位置は有声で毎サンプル ratio、無声で 1 進む(窓付き sinc 補間・
    帯域制限 1/rate)。有声(voiced: 出力サンプル毎・壁時計 n+D で既知の判定)への出入りは ramp_ms で rate を滑らかに切替。
    読み位置 p は許容窓 [n−Lb, n+La](_rrps 参照)に保ち、外れたら局所周期 P(因果 f0・無声は hold_ms まで保持・以後 p0_ms)の
    整数倍だけ跳ぶ。跳び先は ±srch·P で直前 1 周期の相互相関が最大の位置(WSOLA 型・届いているサンプルのみ)。
    無声では周期的な跳びが起きない(等速で素通し)=雑音に周期変調を作らない。"""
    N = len(e)
    Pf = np.zeros(len(f0))
    last, since = SR * p0_ms / 1000, 10 ** 9
    for j in range(len(f0)):
        if f0[j] > 0:
            last, since = SR / f0[j], 0
        else:
            since += F0_HOP
        Pf[j] = last if since <= SR * hold_ms / 1000 else SR * p0_ms / 1000
    n = np.arange(N)
    jj = np.clip((n + D - F0_HOP + 1) // F0_HOP, -1, len(f0) - 1)
    Pn = np.where(jj >= 0, Pf[np.maximum(jj, 0)], SR * p0_ms / 1000)
    v = np.ones(N, bool) if voiced is None else voiced.astype(bool)
    k = max(1, int(SR * ramp_ms / 1000))
    g = np.zeros(N)
    acc = 0.0
    for i in range(N):
        acc = min(1.0, acc + 1.0 / k) if v[i] else max(0.0, acc - 1.0 / k)
        g[i] = acc
    rate = 1.0 + (np.asarray(ratio, dtype=np.float64) - 1.0) * g
    out, lag, jumps = _rrps(e.astype(np.float64), Pn.astype(np.float64), rate.astype(np.float64), int(D), int(half),
                            int(SR * xf_ms / 1000), float(srch))
    return out, {"mean_read_lag_ms": lag / SR * 1000, "jumps": int(jumps), "frac_shifted": float(g.mean())}


def register_ratio(f0: np.ndarray, n_samples: int, D: int, mu_s: float, sd_s: float, mu_t: float, sd_t: float,
                   tau_ms: float = 30.0) -> np.ndarray:
    """f0 レジスタの写像 log f0' = μ_T + (σ_T/σ_S)(log f0 − μ_S) を RRPS の時変比 r(n) にする(出力サンプル毎・壁時計 n+D で既知の f0)。
    比は有声フレームで更新し一次の因果平滑(時定数 tau_ms)、無声では直前値を保持。μ/σ は登録時の定数(発話統計ではない)。"""
    a = np.exp(-F0_HOP / (SR * tau_ms / 1000))
    rf = np.empty(len(f0))
    cur = np.exp(mu_t - mu_s)
    for j in range(len(f0)):
        if f0[j] > 0:
            tgt = np.exp(mu_t + (sd_t / max(sd_s, 1e-3)) * (np.log(f0[j]) - mu_s)) / f0[j]
            cur = a * cur + (1 - a) * tgt
        rf[j] = cur
    n = np.arange(n_samples) + D
    j = np.clip((n - F0_HOP + 1) // F0_HOP, 0, len(f0) - 1)
    return rf[j]
