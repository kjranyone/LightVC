"""高さの追跡 v4(paper_snv 段 P108・段 P124): 網側の高さ・有声 f0_net(8 次 LPC 残差 dn8)と包絡・写し側の高さ・有声 f0_voc(4 次 LPC 混成 + 実候補優先 HOLD + 跳び上限)を先読み 0・発話統計なしで出す。"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

import pae as PA
import snv_p0_front as P

DEC = int(os.environ.get("SNV_V4_DEC", "8"))
ORD = int(os.environ.get("SNV_V4_ORD", "8"))
MIX = float(os.environ.get("SNV_V4_MIX", "1.0"))
U_COST = float(os.environ.get("SNV_V4_UCOST", str(P.U_COST)))
C_MAX = float(os.environ.get("SNV_V4_CMAX", str(P.C_MAX)))
BETA = float(os.environ.get("SNV_V4_BETA", str(P.BETA)))
PRE = DEC * ORD
SPLIT_HZ = float(os.environ.get("SNV_V4_SPLIT", "0"))
C_HI = float(os.environ.get("SNV_V4_CHI", "0.15"))
TH_W = float(os.environ.get("SNV_V4_THW", "0.3"))
MODE = os.environ.get("SNV_V4_MODE", "veto")
U_LO = float(os.environ.get("SNV_V4_ULO", "9"))
EXCL = os.environ.get("SNV_V4_EXCL", "0") == "1"
LV_HI = float(os.environ.get("SNV_V4_LVHI", "-15"))
LV_LO = float(os.environ.get("SNV_V4_LVLO", "-30"))
LV_DECAY = float(os.environ.get("SNV_V4_DECAY", "0.9995"))
U_EXT = float(os.environ.get("SNV_V4_UEXT", "0"))
U_ON = float(os.environ.get("SNV_V4_UON", "0"))
EXT_OCT = float(os.environ.get("SNV_V4_EXTOCT", "0.15"))

W_YIN_V4 = 600
TS_HI = 120
TS_ULTRA = 64
TMAX_NET = 880
TMAX_VOC = 800


def whiten_ord(seg: torch.Tensor, order: int, dec: int = 8) -> torch.Tensor:
    s = seg.double()
    pre = dec * order
    d = s[:, pre::dec]
    d = d * torch.hann_window(d.shape[1], periodic=False, dtype=torch.float64, device=s.device)
    n = d.shape[1]
    r = torch.stack([(d[:, : n - k] * d[:, k:]).sum(1) for k in range(order + 1)], 1)
    r = torch.cat([r[:, :1] * 1.001 + 1e-12, r[:, 1:]], 1)
    idx = (torch.arange(order, device=s.device)[:, None] - torch.arange(order, device=s.device)[None, :]).abs()
    a = torch.linalg.solve(r[:, idx], -r[:, 1:, None])[..., 0]
    out = s[:, pre:].clone()
    for k in range(order):
        out = out + a[:, k : k + 1] * s[:, pre - dec * (k + 1) : s.shape[1] - dec * (k + 1)]
    return out


def whiten(seg: torch.Tensor) -> torch.Tensor:
    out = whiten_ord(seg, ORD, DEC)
    return MIX * out + (1.0 - MIX) * seg.double()[:, PRE:]


def cmndf_seg_ms(s: torch.Tensor, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    L, TM = w + P.TMAX, P.TMAX
    r = s[..., TM:]
    X = torch.fft.irfft(torch.fft.rfft(s, P.NFFT_X) * torch.fft.rfft(r, P.NFFT_X).conj(), P.NFFT_X)[..., : TM + 1]
    C = X.flip(-1)
    cs = torch.nn.functional.pad((s * s).cumsum(-1), (1, 0))
    taus = torch.arange(TM + 1, device=s.device)
    e_lag = cs[..., TM - taus + w] - cs[..., TM - taus]
    e_ref = cs[..., L] - cs[..., TM]
    esum = e_ref[..., None] + e_lag
    d = (esum - 2 * C).clamp_min(0)
    d = torch.where(d < 1e-12 * esum, torch.zeros_like(d), d)
    cum = d[..., 1:].cumsum(-1)
    dn = torch.ones_like(d)
    dn[..., 1:] = torch.where(cum > 1e-20, d[..., 1:] * taus[1:] / cum.clamp_min(1e-20), torch.ones_like(cum))
    return dn, e_ref / w


def cmndf_seg(s: torch.Tensor, w: int) -> torch.Tensor:
    return cmndf_seg_ms(s, w)[0]


def candidates_with(dn: np.ndarray, c_max: float = 0.70, beta: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    seg = dn[:, P.TMIN - 1 : P.TMAX + 1]
    mid = seg[:, 1:-1]
    is_min = (mid < seg[:, :-2]) & (mid <= seg[:, 2:]) & (mid < c_max)
    lag = np.arange(P.TMIN, P.TMAX)[None, :]
    shortest = np.where(is_min, lag, P.TMAX).min(1, keepdims=True)
    score = np.where(is_min, mid + beta * np.log2(lag / shortest), np.inf)
    idx = np.argsort(score, axis=1)[:, : P.K_CAND]
    cost = np.take_along_axis(score, idx, 1)
    tau = idx + P.TMIN
    dm = np.take_along_axis(dn, tau - 1, 1)
    d0 = np.take_along_axis(dn, tau, 1)
    dp = np.take_along_axis(dn, np.minimum(tau + 1, P.TMAX), 1)
    den = dm - 2 * d0 + dp
    off = np.clip(np.where(np.abs(den) > 1e-12, 0.5 * (dm - dp) / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0), -1, 1)
    f = np.where(np.isfinite(cost), P.SR / (tau + off), 0.0)
    return f, cost


def online_dp_v4(
    f: np.ndarray,
    c: np.ndarray,
    ms: np.ndarray,
    lb: np.ndarray,
    efl: float,
    ucost: float,
    hold_base: float,
    hold_min_c: float,
    jump_cap: float,
    sub_exit_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    Tn, K = f.shape
    lf = np.log2(np.maximum(f, 1.0))
    out = np.zeros(Tn, np.float32)
    best = np.zeros(Tn, np.float32)
    prev = np.full(K + 2, np.inf)
    prev_lf = np.zeros(K + 2)
    hold_n = 0
    for t in range(Tn):
        live = ms[t] > efl
        loc = np.full(K + 2, np.inf)
        loc[:K] = np.where(np.isfinite(c[t]) & live, c[t], np.inf)
        loc[K] = ucost if live else 0.0
        cmin = float(np.min(loc[:K]))
        if live and lb[t] >= P.LB_MIN and cmin >= hold_min_c:
            loc[K + 1] = hold_base + P.HOLD_STEP * hold_n
        cur_lf = np.append(lf[t], [0.0, 0.0])
        if not np.isfinite(prev).any():
            acc = loc.copy()
            acc[K + 1] = np.inf
        else:
            vprev = np.r_[np.arange(K), K + 1]
            dlf = lf[t][:, None] - prev_lf[None, vprev]
            jcost = np.minimum(P.LAMBDA * np.abs(dlf), jump_cap)
            if sub_exit_c > 0.0:
                is_mult = (loc[:K, None] < sub_exit_c) & (
                    (np.abs(dlf - 1.0) < 0.12) | (np.abs(dlf - 1.585) < 0.12) | (np.abs(dlf - 2.0) < 0.12)
                )
                jcost = np.where(is_mult, np.minimum(jcost, 0.12), jcost)
            vv = prev[None, vprev] + jcost
            bi = vv.argmin(1)
            from_v = np.minimum(vv[np.arange(K), bi], prev[K] + P.GAMMA)
            hp = int(vprev[int(prev[vprev].argmin())])
            from_u = min(prev[hp] + P.GAMMA, prev[K])
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


def _multi_hi(raw: torch.Tensor) -> np.ndarray:
    tau = torch.arange(TS_HI, device=raw.device)
    m2 = torch.stack([raw[:, (2 * tau + o).clamp(0, P.TMAX)] for o in (-1, 0, 1)], 0).min(0).values
    m3 = torch.stack([raw[:, (3 * tau + o).clamp(0, P.TMAX)] for o in (-2, -1, 0, 1, 2)], 0).min(0).values
    return torch.maximum(raw[:, :TS_HI], torch.maximum(m2, m3)).cpu().numpy()


def _extend_net(f0: np.ndarray, best: np.ndarray, raw: np.ndarray, dn8: np.ndarray, ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = f0.copy()
    bo = best.copy()
    prev = 0.0
    ext_n = 0
    for t in range(len(out)):
        if out[t] > 0.0:
            prev = float(out[t])
            ext_n = 0
            continue
        if ms[t] <= 1e-7:
            prev = 0.0
            ext_n = 0
            continue
        r = raw[t]
        dw = dn8[t]
        if prev > 0.0 and ext_n < 8:
            cc = P.SR / prev
            lo = max(P.TMIN, int(cc * 2.0 ** -0.18))
            hi_i = min(P.TMAX, int(np.ceil(cc * 2.0 ** 0.18)))
            j = lo + int(np.argmin(r[lo : hi_i + 1]))
            if r[j] < 0.35 and dw[j] < 0.72 and lo < j < hi_i:
                out[t] = P.SR / j
                bo[t] = float(r[j])
                prev = float(out[t])
                ext_n += 1
                continue
        elif prev <= 0.0:
            seg = r[TS_HI : TMAX_NET + 1]
            mn = seg[1:-1]
            ok = (mn < seg[:-2]) & (mn <= seg[2:]) & (mn < 0.22) & (dw[TS_HI + 1 : TMAX_NET] < 0.60)
            if ok.any():
                idx = np.where(ok & (mn <= mn[ok].min() + 0.10))[0]
                j = TS_HI + 1 + int(idx.max())
                out[t] = P.SR / j
                bo[t] = float(r[j])
                prev = float(out[t])
                continue
        prev = 0.0
        ext_n = 0
    return out, bo


def f0_dual_track(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xc = x.detach().cpu()
    Tn = P.n_frames(xc.shape[-1])
    xl = P.lowpass(xc).double()
    L = W_YIN_V4 + P.TMAX
    seg8 = P._segments(xl, L + 64, Tn)
    raw, ms = cmndf_seg_ms(seg8[:, 64:], W_YIN_V4)
    _, ms_fb = P.cmndf(xc, W_YIN_V4)
    dn8 = cmndf_seg(whiten_ord(seg8, 8, 8), W_YIN_V4)
    dn4 = cmndf_seg(whiten_ord(seg8[:, 32:], 4, 8), W_YIN_V4)
    mu = _multi_hi(raw)
    raw_np = raw.numpy()
    dn8_np = dn8.numpy()
    dn4_np = dn4.numpy()
    S = P.power_spec(xc)
    lb = P.lowband(S)
    raw_lo_min = raw_np[:, TS_HI:400].min(1)

    hi_net_b = np.maximum(raw_np[:, :TS_HI], dn8_np[:, :TS_HI]).copy()
    hi_net_b[:, :TS_ULTRA] = 1.0
    okm_n = mu < 0.15
    hi_n = np.where(okm_n, mu, np.where(okm_n.any(1, keepdims=True), 1.0, hi_net_b))
    dn_n = np.concatenate([hi_n, dn8_np[:, TS_HI:TMAX_NET], np.ones((Tn, P.TMAX + 1 - TMAX_NET))], 1)
    dn_n[(~okm_n.any(1)) & (lb < 0.32) & (raw_lo_min > 0.35)] = 1.0
    fn_c, cn_c = candidates_with(dn_n, c_max=0.70, beta=0.05)
    ms_net = np.maximum(ms.numpy(), 0.20 * ms_fb.numpy())
    f_net, b_net = online_dp_v4(fn_c, cn_c, ms_net, lb, efl=1e-7, ucost=0.50, hold_base=0.50, hold_min_c=0.40, jump_cap=0.45, sub_exit_c=0.22)
    f_net, b_net = _extend_net(f_net, b_net, raw_np, dn8_np, ms_net)

    hi_voc_b = np.maximum(raw_np[:, :TS_HI], dn4_np[:, :TS_HI]).copy()
    hi_voc_b[:, :TS_ULTRA] = 1.0
    okm_v = mu < 0.18
    hi_v = np.where(okm_v, mu, np.where(okm_v.any(1, keepdims=True), 1.0, hi_voc_b))
    lo_v = 0.88 * raw_np[:, TS_HI:TMAX_VOC] + 0.12 * dn4_np[:, TS_HI:TMAX_VOC]
    dn_v = np.concatenate([hi_v, lo_v, np.ones((Tn, P.TMAX + 1 - TMAX_VOC))], 1)
    dn_v[(~okm_v.any(1)) & (lb < 0.30) & (raw_lo_min > 0.35)] = 1.0
    fv_c, cv_c = candidates_with(dn_v, c_max=0.70, beta=0.05)
    ms_voc = np.maximum(ms.numpy(), 0.25 * ms_fb.numpy())
    f_voc, _ = online_dp_v4(fv_c, cv_c, ms_voc, lb, efl=1e-9, ucost=0.65, hold_base=0.50, hold_min_c=0.45, jump_cap=0.38, sub_exit_c=0.25)

    return torch.from_numpy(f_net), torch.from_numpy(f_voc), torch.from_numpy(b_net)


def f0_net_track(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fn, _, bn = f0_dual_track(x)
    return fn, bn


def f0_voc_track(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, fv, bn = f0_dual_track(x)
    return fv, bn


def short_c0(x: np.ndarray, n: int, win_ms: float = 20.0) -> np.ndarray:
    return ((P.fixed_env(x, n, win_ms=win_ms)[0] - PA.C0_SIL) / 10.0).astype(np.float32)


def _core(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xl = P.lowpass(x).double()
    Tn = P.n_frames(x.shape[-1])
    L = P.W_YIN + P.TMAX
    seg = P._segments(xl, L + PRE, Tn)
    _, ms = P.cmndf(xl, P.W_YIN)
    dn = cmndf_seg(whiten(seg), P.W_YIN)
    raw_all = cmndf_seg(seg[:, PRE:], P.W_YIN)
    if SPLIT_HZ > 0:
        ts = int(P.SR / SPLIT_HZ)
        raw = raw_all
        if MODE in ("max", "multi"):
            hi = torch.maximum(raw[:, :ts], dn[:, :ts])
            if MODE == "multi":
                tau = torch.arange(ts, device=raw.device)
                m2 = torch.stack([raw[:, (2 * tau + o).clamp(0, P.TMAX)] for o in (-1, 0, 1)], 0).min(0).values
                m3 = torch.stack([raw[:, (3 * tau + o).clamp(0, P.TMAX)] for o in (-2, -1, 0, 1, 2)], 0).min(0).values
                mu = torch.maximum(raw[:, :ts], torch.maximum(m2, m3))
                okm = mu < C_HI
                hi = torch.where(okm, mu, torch.where(okm.any(1, keepdim=True) & EXCL, torch.ones_like(hi), hi))
            dn = torch.cat([hi, dn[:, ts:]], 1)
        else:
            free = dn[:, ts:].min(1).values > TH_W
            hi = torch.where(raw[:, :ts] < C_HI, raw[:, :ts], torch.ones_like(raw[:, :ts]))
            dn = torch.cat([torch.where(free[:, None], hi, dn[:, :ts]), dn[:, ts:]], 1)
    keep = (P.C_MAX, P.BETA, P.U_COST)
    P.C_MAX, P.BETA, P.U_COST = C_MAX, BETA, U_COST
    try:
        f, c = P.candidates(dn.cpu().numpy())
        S = P.power_spec(x)
        c = c + P.harmonic_cost(S, f)
        msn = ms.cpu().numpy()
        if U_LO < U_COST:
            run, rel = 0.0, np.zeros(len(msn))
            for i, e in enumerate(msn):
                run = max(float(e), run * LV_DECAY)
                rel[i] = 10.0 * np.log10((float(e) + 1e-20) / (run + 1e-20))
            u = U_LO + (U_COST - U_LO) * np.clip((rel - LV_LO) / (LV_HI - LV_LO), 0.0, 1.0)
            c = c + (U_COST - u)[:, None]
        f0, best = P.online_dp(f, c, msn, P.lowband(S))
    finally:
        P.C_MAX, P.BETA, P.U_COST = keep
    return f0, best, raw_all.cpu().numpy(), msn


def extend(f0: np.ndarray, best: np.ndarray, raw: np.ndarray, ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out, bo = f0.copy(), best.copy()
    ts = int(P.SR / SPLIT_HZ) if SPLIT_HZ > 0 else P.TMIN
    prev = 0.0
    for t in range(len(f0)):
        if out[t] > 0 or ms[t] <= P.E_FLOOR:
            prev = out[t]
            continue
        r = raw[t]
        if prev > 0 and U_EXT > 0:
            c = P.SR / prev
            lo, hi = max(P.TMIN, int(c * 2.0 ** -EXT_OCT)), min(P.TMAX, int(np.ceil(c * 2.0 ** EXT_OCT)))
            j = lo + int(np.argmin(r[lo : hi + 1]))
            if r[j] < U_EXT and lo < j < hi:
                out[t], bo[t] = P.SR / j, r[j]
        elif prev <= 0 and U_ON > 0:
            seg = r[ts : P.TMAX + 1]
            mn = seg[1:-1]
            ok = (mn < seg[:-2]) & (mn <= seg[2:]) & (mn < U_ON)
            if ok.any():
                idx = np.where(ok & (mn <= mn[ok].min() + 0.1))[0]
                j = ts + 1 + int(idx.max())
                out[t], bo[t] = P.SR / j, r[j]
        prev = out[t]
    return out, bo


def f0_track(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if os.environ.get("SNV_V4_FINAL") == "voc":
        return f0_voc_track(x)
    if os.environ.get("SNV_V4_FINAL") == "net":
        return f0_net_track(x)
    f0, best, raw, ms = _core(x)
    if U_EXT > 0 or U_ON > 0:
        f0, best = extend(f0, best, raw, ms)
    return torch.from_numpy(f0.astype(np.float32)), torch.from_numpy(best.astype(np.float32))


def _future() -> None:
    torch.set_num_threads(4)
    from pathlib import Path
    import librosa
    from ship_check import future_invariance
    root = Path(__file__).resolve().parent.parent
    srcs = [
        root / "data/snv_p_feat/real/uirou_0.wav",
        root / "data/snv_p_feat/real/namikawa_0.wav",
        root / "data/snv_e1/ef100/t05.wav",
        root / "data/snv_e1/ef032/t05.wav",
    ]
    n = 2 * P.SR
    pl = [torch.from_numpy(librosa.load(str(p), sr=P.SR, mono=True, duration=3.0)[0][:n].astype(np.float32)) for p in srcs if p.exists()]
    assert len(pl) == 4
    la_net = future_invariance(lambda x: f0_net_track(x)[0], P.H, sr=P.SR, probe_list=pl, n_edit=8, quantity=False)
    la_voc = future_invariance(lambda x: f0_voc_track(x)[0], P.H, sr=P.SR, probe_list=pl, n_edit=8, quantity=False)
    la_c0 = future_invariance(lambda x: torch.from_numpy(short_c0(x.numpy(), P.n_frames(x.shape[-1]), 20.0)), P.H, sr=P.SR, probe_list=pl, n_edit=8, quantity=False)
    for name, la in (("f0_net", la_net), ("f0_voc", la_voc), ("short_c0_20ms", la_c0)):
        print(f"future_invariance {name}: {la}", flush=True)
        assert not la.unbounded and not la.inconclusive and la.samples == 0, (name, la)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _future()
