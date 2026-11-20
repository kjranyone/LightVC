"""P0a(synth_nam_vc §8・学習なし): P 線の前段 = 自前の信号処理だけ・48 kHz・200 fps(H = 240)・フレーム t の窓は (t+1)H で終わる(先読み 0)。
高さと有声 = 因果の低域通過(Butterworth 4 次)→ 左寄せの CMNDF(基準窓 = 最新の 1200・遅れ 20〜960 = 2400〜50 Hz)→ 谷の候補 K_CAND 個(放物線の補間・最も高い候補から 1 オクターブ低いごとに BETA を足す = 同じ深さなら短い周期)+ 無声の状態
  + 倍音の検査(左寄せの Hann 4096 のパワーで、候補 f の奇数倍が偶数倍より OE_DOWN 未満なら f は低すぎ・f/2 の奇数倍が OE_UP を超えれば f は高すぎ → ETA の費用)
  + 保持の状態(直前が有声で 1 kHz 未満のパワーの比 ≥ LB_MIN の間、直前の高さを保って有声を続ける・費用 HOLD_BASE + HOLD_STEP × 続いたフレーム数)
  → 前向きだけの動的計画(局所の費用 = 谷の値・無声 = U_COST・高さの跳び LAMBDA × |Δ log2 f0|・有声と無声の切り替え GAMMA)で、各時刻にそこで終わる最良の経路の状態を出す
  (過去の判定を書き換えない = 先読み 0・発話の統計なし)。
包絡 = CheapTrick を因果の時刻 τ_t = min(tH, (t+1)H − 1.5 T0) で(窓の右端 ≤ (t+1)H)→ 128 帯 mel → log → DCT c0..c24。
周期性 = 左寄せの Hann 2048 で 4 帯の倍音の山と谷のパワー比(pae.periodicity と同じ式・窓だけ左寄せ)。
cond = 描画器 rvoc の界面 [31, T](c0n・c1..c24 / 10・log(f0 / 200)・有声・周期性 × 有声)。
    cd training && uv run --no-sync python snv_p0_front.py test
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import nvoc as N
import pae as PA

SR = N.SR
H = N.HOP
W_YIN = 1200
LP_HZ = 1500.0
LP_ORDER = 4
K_CAND = 6
C_MAX = 0.7
BETA = 0.03
LAMBDA = 1.0
GAMMA = 0.3
U_COST = 0.65
HOLD_BASE = 0.5
HOLD_STEP = 0.02
LB_HZ = 1000.0
LB_MIN = 0.7
TMIN = 20
TMAX = 960
E_FLOOR = 1e-7
NFFT_P = 2048
NFFT_X = 4096
NFFT_S = 4096
HARM_HZ = 3000.0
OE_DOWN = -10.0
OE_UP = -6.0
OE_W = 2.0
ETA = 0.0
PAD_L = 4096
ROOT = Path(__file__).resolve().parent.parent


def n_frames(n: int) -> int:
    return n // H


def _segments(x: torch.Tensor, length: int, T: int) -> torch.Tensor:
    xl = torch.nn.functional.pad(x, (length - H, 0))
    return xl.unfold(-1, length, H)[..., :T, :]


def lowpass(x: torch.Tensor) -> torch.Tensor:
    from scipy.signal import butter, lfilter
    b, a = butter(LP_ORDER, LP_HZ / (SR / 2))
    return torch.from_numpy(lfilter(b, a, x.double().cpu().numpy(), axis=-1)).to(x.device)


def cmndf(x: torch.Tensor, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.double()
    T = n_frames(x.shape[-1])
    L = w + TMAX
    s = _segments(x, L, T)
    r = s[..., TMAX:]
    X = torch.fft.irfft(torch.fft.rfft(s, NFFT_X) * torch.fft.rfft(r, NFFT_X).conj(), NFFT_X)[..., : TMAX + 1]
    C = X.flip(-1)
    cs = torch.nn.functional.pad((s * s).cumsum(-1), (1, 0))
    taus = torch.arange(TMAX + 1, device=x.device)
    e_lag = cs[..., TMAX - taus + w] - cs[..., TMAX - taus]
    e_ref = cs[..., L] - cs[..., TMAX]
    d = (e_ref[..., None] + e_lag - 2 * C).clamp_min(0)
    cum = d[..., 1:].cumsum(-1)
    dn = torch.ones_like(d)
    dn[..., 1:] = d[..., 1:] * taus[1:] / cum.clamp_min(1e-20)
    return dn, e_ref / w


def candidates(dn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seg = dn[:, TMIN - 1: TMAX + 1]
    mid = seg[:, 1:-1]
    is_min = (mid < seg[:, :-2]) & (mid <= seg[:, 2:]) & (mid < C_MAX)
    lag = np.arange(TMIN, TMAX)[None, :]
    shortest = np.where(is_min, lag, TMAX).min(1, keepdims=True)
    score = np.where(is_min, mid + BETA * np.log2(lag / shortest), np.inf)
    idx = np.argsort(score, axis=1)[:, :K_CAND]
    cost = np.take_along_axis(score, idx, 1)
    tau = idx + TMIN
    dm = np.take_along_axis(dn, tau - 1, 1)
    d0 = np.take_along_axis(dn, tau, 1)
    dp = np.take_along_axis(dn, np.minimum(tau + 1, TMAX), 1)
    den = dm - 2 * d0 + dp
    off = np.clip(np.where(np.abs(den) > 1e-12, 0.5 * (dm - dp) / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0), -1, 1)
    f = np.where(np.isfinite(cost), SR / (tau + off), 0.0)
    return f, cost


def power_spec(x: torch.Tensor) -> np.ndarray:
    T = n_frames(x.shape[-1])
    fr = _segments(x.double(), NFFT_S, T)
    win = torch.hann_window(NFFT_S, periodic=False, dtype=torch.float64, device=x.device)
    return torch.fft.rfft(fr * win, n=NFFT_S).abs().pow(2).cpu().numpy()


def odd_even(S: np.ndarray, f: np.ndarray) -> np.ndarray:
    df = SR / NFFT_S
    nb = S.shape[-1]

    def pw(q: np.ndarray) -> np.ndarray:
        i = np.clip(np.round(q / df).astype(np.int64), 2, nb - 3)
        win = np.stack([np.take_along_axis(S, i + o, 1) for o in range(-2, 3)], -1)
        return np.where(q < HARM_HZ, win.max(-1), 0.0)
    odd = sum(pw(k * f) for k in (1, 3, 5, 7))
    even = sum(pw(k * f) for k in (2, 4, 6, 8))
    ok = (f > 0) & (2 * f < HARM_HZ)
    return np.where(ok, 10 * np.log10((odd + 1e-20) / (even + 1e-20)), np.nan)


def harmonic_cost(S: np.ndarray, f: np.ndarray) -> np.ndarray:
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    s_f = odd_even(S, f)
    s_h = odd_even(S, f / 2)
    down = np.where(np.isnan(s_f), 0.0, sig((OE_DOWN - s_f) / OE_W))
    up = np.where(np.isnan(s_h), 0.0, sig((s_h - OE_UP) / OE_W))
    return ETA * (down + up)


def lowband(S: np.ndarray) -> np.ndarray:
    k = int(LB_HZ / (SR / NFFT_S))
    return S[:, 1:k].sum(1) / (S[:, 1:].sum(1) + 1e-20)


def online_dp(f: np.ndarray, c: np.ndarray, ms: np.ndarray, lb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, K = f.shape
    lf = np.log2(np.maximum(f, 1.0))
    out = np.zeros(T, np.float32)
    best = np.zeros(T, np.float32)
    prev = np.full(K + 2, np.inf)
    prev_lf = np.zeros(K + 2)
    hold_n = 0
    for t in range(T):
        live = ms[t] > E_FLOOR
        loc = np.full(K + 2, np.inf)
        loc[:K] = np.where(np.isfinite(c[t]) & live, c[t], np.inf)
        loc[K] = U_COST if live else 0.0
        if live and lb[t] >= LB_MIN:
            loc[K + 1] = HOLD_BASE + HOLD_STEP * hold_n
        cur_lf = np.append(lf[t], [0.0, 0.0])
        if not np.isfinite(prev).any():
            acc = loc.copy()
            acc[K + 1] = np.inf
        else:
            vprev = np.r_[np.arange(K), K + 1]
            jump = np.abs(lf[t][:, None] - prev_lf[None, vprev])
            vv = prev[None, vprev] + LAMBDA * jump
            bi = vv.argmin(1)
            from_v = np.minimum(vv[np.arange(K), bi], prev[K] + GAMMA)
            hp = int(vprev[int(prev[vprev].argmin())])
            from_u = min(prev[hp] + GAMMA, prev[K])
            acc = loc + np.r_[from_v, from_u, prev[hp]]
            cur_lf[K + 1] = prev_lf[hp]
        fin = np.isfinite(acc)
        acc = acc - acc[fin].min()
        j = int(np.argmin(acc))
        hold_n = hold_n + 1 if j == K + 1 else 0
        out[t] = 2.0 ** cur_lf[j] if j != K and cur_lf[j] > 0 else 0.0
        best[t] = c[t, j] if j < K else 1.0
        prev, prev_lf = acc, cur_lf
    return out, best


def f0_track(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dn, ms = cmndf(lowpass(x), W_YIN)
    f, c = candidates(dn.cpu().numpy())
    S = power_spec(x)
    c = c + harmonic_cost(S, f)
    f0, best = online_dp(f, c, ms.cpu().numpy(), lowband(S))
    return torch.from_numpy(f0), torch.from_numpy(best)


def voice_share(P: torch.Tensor, m: torch.Tensor, f: np.ndarray, n_h: int = 8) -> np.ndarray:
    df = SR / NFFT_P
    nb = P.shape[-1]
    Pn, Mn = P.cpu().numpy(), m.double().pow(2).cpu().numpy()
    T = min(len(f), Pn.shape[0])
    num = np.zeros(f.shape)
    den = np.zeros(f.shape)
    for k in range(1, n_h + 1):
        i = np.clip(np.round(k * f[:T] / df).astype(np.int64), 0, nb - 1)
        pk = np.take_along_axis(Pn[:T], i, 1)
        num[:T] += pk * np.take_along_axis(Mn[:T], i, 1)
        den[:T] += pk
    return np.where(den > 0, num / np.maximum(den, 1e-30), 1.0)


def f0_track_masked(x: torch.Tensor, m: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    dn, ms = cmndf(lowpass(x), W_YIN)
    f, c = candidates(dn.cpu().numpy())
    S = power_spec(x)
    c = c + harmonic_cost(S, f)
    c = c + alpha * (1.0 - voice_share(power_p(x, f.shape[0]), m, f))
    f0, best = online_dp(f, c, ms.cpu().numpy(), lowband(S))
    return torch.from_numpy(f0), torch.from_numpy(best)


def mask_to_grid(m: torch.Tensor, nb: int) -> torch.Tensor:
    m = m.double()
    pos = torch.arange(nb, dtype=torch.float64, device=m.device) * (m.shape[1] - 1) / (nb - 1)
    lo = pos.floor().long().clamp(max=m.shape[1] - 1)
    hi = (lo + 1).clamp(max=m.shape[1] - 1)
    fr = pos - lo
    return m[:, lo] * (1 - fr) + m[:, hi] * fr


def cmndf_masked(x: torch.Tensor, w: int, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.double()
    T = n_frames(x.shape[-1])
    L = w + TMAX
    s0 = _segments(x, L, T)
    mm = m.double().to(x.device)[:T]
    keep = (mm >= 1.0).all(1)
    sm = torch.fft.irfft(torch.fft.rfft(s0, NFFT_X) * mask_to_grid(mm, NFFT_X // 2 + 1), NFFT_X)[..., :L]
    s = torch.where(keep[:, None], s0, sm)
    r = s[..., TMAX:]
    X = torch.fft.irfft(torch.fft.rfft(s, NFFT_X) * torch.fft.rfft(r, NFFT_X).conj(), NFFT_X)[..., : TMAX + 1]
    C = X.flip(-1)
    cs = torch.nn.functional.pad((s * s).cumsum(-1), (1, 0))
    taus = torch.arange(TMAX + 1, device=x.device)
    e_lag = cs[..., TMAX - taus + w] - cs[..., TMAX - taus]
    e_ref = cs[..., L] - cs[..., TMAX]
    d = (e_ref[..., None] + e_lag - 2 * C).clamp_min(0)
    cum = d[..., 1:].cumsum(-1)
    dn = torch.ones_like(d)
    dn[..., 1:] = d[..., 1:] * taus[1:] / cum.clamp_min(1e-20)
    return dn, e_ref / w


def f0_track_myin(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dn, ms = cmndf_masked(lowpass(x), W_YIN, m)
    f, c = candidates(dn.cpu().numpy())
    S = power_spec(x)
    T = min(S.shape[0], m.shape[0])
    mk = mask_to_grid(m.double()[:T], S.shape[1]).pow(2).cpu().numpy()
    keep = (m.double()[:T] >= 1.0).all(1).cpu().numpy()
    S = S.copy()
    S[:T] = np.where(keep[:, None], S[:T], S[:T] * mk)
    c = c + harmonic_cost(S, f)
    f0, best = online_dp(f, c, ms.cpu().numpy(), lowband(S))
    return torch.from_numpy(f0), torch.from_numpy(best)


def envelope_amp(x: np.ndarray, f0: np.ndarray) -> np.ndarray:
    import pyworld
    fb, _ = PA._np_consts()
    T = len(f0)
    xa = np.concatenate([np.zeros(PAD_L), x.astype(np.float64), np.zeros(PA.A)])
    T0 = 1.0 / np.where(f0 > 0, f0, 200.0)
    t = np.arange(T)
    tau = np.minimum(t * H, (t + 1) * H - 2 - np.ceil(1.5 * T0 * SR))
    sp = pyworld.cheaptrick(xa, np.where(f0 > 0, f0, 0.0).astype(np.float64), (PAD_L + tau) / SR, SR, fft_size=PA.NFFT)
    return np.sqrt(sp) @ fb.T


def envelope_from_amp(a: np.ndarray) -> np.ndarray:
    _, dct = PA._np_consts()
    return (dct[:25] @ np.log(np.maximum(a, 1e-5)).T).astype(np.float32)


def envelope(x: np.ndarray, f0: np.ndarray) -> np.ndarray:
    return envelope_from_amp(envelope_amp(x, f0))


def power_p(x: torch.Tensor, T: int) -> torch.Tensor:
    fr = _segments(x.double(), NFFT_P, T)
    win = torch.hann_window(NFFT_P, periodic=False, device=x.device, dtype=torch.float64)
    return torch.fft.rfft(fr * win, n=NFFT_P).abs().pow(2)


def periodicity(x: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    return periodicity_from_power(power_p(x, f0.shape[-1]), f0)


def periodicity_from_power(P: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    df = SR / NFFT_P
    f0d = f0.double()
    fv = torch.where(f0d > 0, f0d, torch.full_like(f0d, 500.0))
    kmax = int(PA.HB[-1][1] / 60.0) + 1
    k = torch.arange(1, kmax + 1, device=P.device, dtype=torch.float64)
    fk = fv[..., None] * k
    pk = torch.stack([PA._interp_bins(P, (fk + o * fv[..., None]) / df) for o in (-0.2, -0.1, 0.0, 0.1, 0.2)], -1).amax(-1)
    vl = torch.stack([PA._interp_bins(P, (fk + o * fv[..., None]) / df) for o in (0.4, 0.5, 0.6)], -1).amin(-1)
    out = []
    for lo, hi in PA.HB:
        m = ((fk >= lo) & (fk < hi) & (fk < SR / 2 - 200)).double()
        num, den = (pk * m).sum(-1), (vl * m).sum(-1)
        hnr = 10 * torch.log10((num + 1e-20) / (den + 1e-20))
        ok = (m.sum(-1) > 0) & (f0d > 0) & (num > 1e-12)
        out.append(torch.where(ok, (hnr / 30).clamp(0, 1.5), torch.zeros_like(hnr)))
    return torch.stack(out, -2).float()


def fixed_env(x: np.ndarray, n: int, win_ms: float = 40.0) -> np.ndarray:
    """段 P91: 固定の窓(ハン・左寄せ = フレーム t は (t + 1) H までの標本だけを見る)の STFT → mel の帯 → 対数 → DCT 25 の包絡。"""
    fb, dct = PA._np_consts()
    W = int(win_ms * 1e-3 * SR)
    xp = np.concatenate([np.zeros(W), np.asarray(x, np.float64), np.zeros(max(0, (n + 1) * H - len(x)))])
    w = np.hanning(W)
    out = np.empty((25, n), np.float32)
    for a in range(0, n, 256):
        t = np.arange(a, min(a + 256, n))
        idx = ((t + 1) * H)[:, None] + np.arange(W)[None]
        amp = np.abs(np.fft.rfft(xp[idx] * w, PA.NFFT, axis=1))
        out[:, t] = dct[:25] @ np.log(np.maximum(amp @ fb.T, 1e-5)).T
    return out


def causal_smooth(e: np.ndarray, w: float) -> np.ndarray:
    q = np.pad(e, ((0, 0), (2, 0)), mode="edge")
    return (w * q[:, :-2] + (1 - 2 * w) * q[:, 1:-1] + w * q[:, 2:]).astype(np.float32)


def analyze(x: np.ndarray, dev: str = "cpu") -> dict:
    xt = torch.from_numpy(x.astype(np.float32)).to(dev)
    f0, ap = f0_track(xt)
    f0n = f0.cpu().numpy().astype(np.float64)
    env = envelope(x, f0n)
    per = periodicity(xt, f0).cpu().numpy()
    return {"f0": f0n.astype(np.float32), "ap": ap.cpu().numpy(), "env": env, "per": per}


def cond(f: dict, f0: np.ndarray | None = None, smooth: float = 0.0) -> np.ndarray:
    f0 = f["f0"] if f0 is None else f0
    env = causal_smooth(f["env"], smooth) if smooth > 0 else f["env"]
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
    v = (f0 > 0).astype(np.float32)[None]
    per = np.clip(f["per"], 0, 1.5) * v
    return np.concatenate([((env[0] - PA.C0_SIL) / 10)[None], env[1:25] / 10, lf, v, per], 0).astype(np.float32)


def _selftest() -> None:
    t = np.arange(SR) / SR
    for hz in (55.0, 110.0, 220.0, 440.0, 1100.0, 1800.0):
        x = 0.3 * np.sign(np.sin(2 * np.pi * hz * t)) * (1 + 0.0 * t)
        f0, _ = f0_track(torch.from_numpy(x.astype(np.float32)))
        v = f0[20:].numpy()
        assert (v > 0).mean() > 0.95, (hz, (v > 0).mean())
        err = np.abs(1200 * np.log2(v[v > 0] / hz))
        assert np.median(err) < 5, (hz, np.median(err))
    x = np.random.default_rng(0).normal(0, 0.1, SR).astype(np.float32)
    f0, _ = f0_track(torch.from_numpy(x))
    assert (f0.numpy() > 0).mean() < 0.05, (f0.numpy() > 0).mean()
    f0, _ = f0_track(torch.zeros(SR))
    assert float(f0.abs().max()) == 0.0
    print("selftest ok")


def _future() -> None:
    from ship_check import future_invariance, probes
    pl = [p.float() for p in probes(2 * SR, sr=SR, male=2)]
    la_f0 = future_invariance(lambda x: f0_track(x)[0], H, sr=SR, probe_list=pl, n_edit=8)
    la_ap = future_invariance(lambda x: f0_track(x)[1], H, sr=SR, probe_list=pl, n_edit=8)

    def env_fn(x: torch.Tensor) -> torch.Tensor:
        xn = x.numpy().astype(np.float64)
        return torch.from_numpy(envelope(xn, f0_track(x)[0].numpy().astype(np.float64)))

    def per_fn(x: torch.Tensor) -> torch.Tensor:
        return periodicity(x, f0_track(x)[0])

    la_env = future_invariance(env_fn, H, sr=SR, probe_list=pl[:4], n_edit=4)
    la_per = future_invariance(per_fn, H, sr=SR, probe_list=pl, n_edit=8)
    for k, la in (("f0", la_f0), ("cmndf_min", la_ap), ("envelope", la_env), ("periodicity", la_per)):
        print(f"future_invariance {k}: {la}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _selftest()
        _future()
