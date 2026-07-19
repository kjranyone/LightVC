"""E0 oracle analysis-synthesis gate for NSF-LTV (current/vocoder.md §6).

Zero-training kill switch: real audio -> True Envelope / CheapTrick + D4C ->
LtvRenderer resynthesis. Measures the physical ceiling of the LTV design
family before any SGD: K_v x Nb sweep (F1 muffle onset), +/- pitch-sync noise
modulation (F2 breath fusion), min-phase vs linear-phase, TE vs CheapTrick.
WORLD synthesize is the anchor arm (contrast ratio ~1.02 reference).

Gate (objective half): contrast_ratio >= 0.95 * WORLD arm on real voices.
Final verdict is the human ear (listen_gui.py on the exported wavs).

Usage:
  cd training
  uv run python e0_oracle_ltv.py                    # full sweep on golden picks
  uv run python e0_oracle_ltv.py --quick            # reduced sweep
  uv run python e0_oracle_ltv.py --files a.wav ...  # explicit inputs
Outputs:
  results/e0_oracle_ltv/{utt}_{arm}.wav (+ _gt) for listen_gui.py
  results/e0_oracle_ltv.json scoreboard (GT alongside, persisted)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
import torch

from ltv_render import HOP, NBINS, NFFT, SR, LtvRenderer

FRAME_MS = 1000.0 * HOP / SR
ROOT = Path(__file__).resolve().parent.parent
EPS = 1e-8

DEFAULT_CATS = ["target_female:small_voice", "target_female:emotional",
                "target_female:sibilant", "target_female:breathy",
                "source_male:deep"]


def load_wav(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    peak = np.abs(y).max() + EPS
    if peak > 0.95:
        y = y * (0.95 / peak)
    return y.astype(np.float64)


def world_analysis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f0, t = pw.harvest(x, SR, frame_period=FRAME_MS)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR, fft_size=NFFT)
    ap = pw.d4c(x, f0, t, SR, fft_size=NFFT)
    return f0, t, sp, ap


def stft_logmag(x: np.ndarray, win: int = NFFT) -> np.ndarray:
    s = librosa.stft(x.astype(np.float32), n_fft=NFFT, hop_length=HOP,
                     win_length=win, center=True)
    return np.log(np.abs(s) + EPS).T


def _cep_smooth(a: np.ndarray, order: int) -> np.ndarray:
    c = np.fft.irfft(a, NFFT)
    c[order + 1:NFFT - order] = 0.0
    return np.fft.rfft(c, NFFT).real


def true_envelope(logmag: np.ndarray, f0: np.ndarray, max_iter: int = 80,
                  tol: float = 0.023, uv_f0: float = 250.0) -> np.ndarray:
    env = np.empty_like(logmag)
    for i in range(logmag.shape[0]):
        f = f0[i] if f0[i] > 1.0 else uv_f0
        order = int(round(SR / (2.0 * max(f, 70.0))))
        a = logmag[i]
        v = _cep_smooth(a, order)
        for _ in range(max_iter):
            v = _cep_smooth(np.maximum(a, v), order)
            if (a - v).max() < tol:
                break
        env[i] = v
    return env


def split_envelopes(log_env: np.ndarray, ap: np.ndarray,
                    a0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    apc = np.clip(ap, 1e-5, 1.0 - 1e-6)
    apv = np.clip((apc - a0) / (1.0 - a0), 1e-5, 1.0 - 1e-6) if a0 > 0.0 else apc
    h_v = log_env + 0.5 * np.log(np.clip(1.0 - apv ** 2, 1e-12, None))
    h_n = log_env + np.log(apc)
    return h_v, h_n


def tsmooth_env(h: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return h
    pad = w // 2
    hp = np.pad(h, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([hp[i:i + w].mean(0) for i in range(h.shape[0])])


def oracle_subframe_gain(gt: np.ndarray, T: int, lo_hz: float = 2000.0,
                         j: int = 4) -> np.ndarray:
    n = T * HOP
    y = np.fft.rfft(gt[:n].astype(np.float64))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    y[f < lo_hz] = 0.0
    hf = np.fft.irfft(y, n)
    sub = np.sqrt((hf.reshape(T * j, HOP // j) ** 2).mean(-1) + 1e-10)
    a = sub.reshape(T, j)
    return a / (a.mean(-1, keepdims=True) + 1e-8)


def oracle_d(ap: np.ndarray, f0: np.ndarray, lo_hz: float = 2000.0) -> np.ndarray:
    lo = int(lo_hz / (SR / 2) * (ap.shape[1] - 1))
    d = np.clip(ap[:, lo:].mean(1), 0.0, 1.0)
    return d * (f0[:len(d)] > 1.0)


def analysis_cached(uid: str, x: np.ndarray, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(x[:4096].tobytes() + str(len(x)).encode()).hexdigest()[:10]
    p = cache_dir / f"{uid}_{key}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    f0, t, sp, ap = world_analysis(x.astype(np.float64))
    T = min(len(f0), int(len(x) // HOP))
    log_ct = 0.5 * np.log(sp[:T] + EPS)
    log_te = true_envelope(stft_logmag(x)[:T], f0[:T])
    out = {"f0": f0[:T], "sp": sp[:T], "ap": ap[:T],
           "log_ct": log_ct, "log_te": log_te}
    np.savez_compressed(p, **out)
    return out


def hpv_envelopes(logmag: np.ndarray, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, NB = logmag.shape
    freqs = np.linspace(0.0, SR / 2, NB)
    binhz = (SR / 2) / (NB - 1)
    hv = np.empty_like(logmag)
    hn = np.empty_like(logmag)
    for i in range(T):
        A = logmag[i]
        if f0[i] > 1.0:
            K = max(2, int((SR / 2) / f0[i]))
            fk = np.arange(1, K + 1) * f0[i]
            pb = np.clip((fk / binhz).round().astype(int), 0, NB - 1)
            pk = np.array([A[max(0, b - 3):b + 4].max() for b in pb])
            hv[i] = np.interp(freqs, fk, pk)
            fm = (np.arange(1, K) + 0.5) * f0[i]
            mb = np.clip((fm / binhz).round().astype(int), 0, NB - 1)
            fl = np.array([np.percentile(A[max(0, b - 2):b + 3], 25) for b in mb])
            hn[i] = np.interp(freqs, fm, fl)
        else:
            k = 15
            sm = np.convolve(np.pad(A, k, mode="edge"),
                             np.ones(2 * k + 1) / (2 * k + 1), "valid")
            hn[i] = sm
            hv[i] = sm - 14.0
    return hv, hn


def oracle_mvf(logmag: np.ndarray, f0: np.ndarray, tau: float = 0.3) -> np.ndarray:
    T, NB = logmag.shape
    binhz = (SR / 2) / (NB - 1)
    fm = np.full(T, np.nan)
    for i in range(T):
        f = f0[i]
        if f <= 1.0:
            continue
        K = int((SR / 2 - 600.0) / f)
        if K < 4:
            fm[i] = SR / 2
            continue
        ks = np.arange(1, K)
        pb = np.clip((ks * f / binhz).round().astype(int), 0, NB - 1)
        vb = np.clip(((ks + 0.5) * f / binhz).round().astype(int), 0, NB - 1)
        sh = logmag[i][pb] - logmag[i][vb]
        sh = np.convolve(sh, np.ones(3) / 3.0, mode="same")
        below = np.where(sh < tau)[0]
        k_star = int(ks[below[0]]) if len(below) else K
        fm[i] = float(np.clip(k_star * f, 1200.0, SR / 2))
    idx = np.arange(T)
    good = np.isfinite(fm)
    fm = np.interp(idx, idx[good], fm[good]) if good.any() else np.full(T, 4000.0)
    return np.array([np.median(fm[max(0, i - 2):i + 3]) for i in range(T)])


def hpv_paw_envelopes(x: np.ndarray, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = len(f0)
    hv_p = np.empty((T, NBINS))
    hn_p = np.empty((T, NBINS))
    lms = {w: stft_logmag(x, win=w)[:T] for w in (2048, 1024, 512)}
    t0 = 3.0 * SR / np.maximum(f0, 1.0)
    for w in (2048, 1024, 512):
        hvw, hnw = hpv_envelopes(lms[w], f0)
        hvw = hvw + np.log(NFFT / w)
        hnw = hnw + 0.5 * np.log(NFFT / w)
        if w == 2048:
            sel = (f0 <= 1.0) | (t0 > 1024)
        elif w == 1024:
            sel = (f0 > 1.0) & (t0 <= 1024) & (t0 > 512)
        else:
            sel = (f0 > 1.0) & (t0 <= 512)
        hv_p[sel], hn_p[sel] = hvw[sel], hnw[sel]
    return hv_p, hn_p


_HPV_CAL: dict = {}


def hpv_calibration() -> tuple[float, float]:
    if "c" not in _HPV_CAL:
        T = 90
        r = LtvRenderer(k_v=1024, k_n=256, nb_in=NBINS)
        f0t = torch.full((1, T), 200.0)
        flat = torch.zeros(1, T, NBINS)
        dead = torch.full((1, T, NBINS), -30.0)
        torch.manual_seed(0)
        nz = torch.randn(1, T * HOP)
        with torch.no_grad():
            yh = r(f0t, flat, dead, noise=nz)["y"][0].numpy()
            yn = r(f0t, dead, flat, noise=nz)["y"][0].numpy()
        lm_h = stft_logmag(yh)[10:T - 10]
        lm_n = stft_logmag(yn)[10:T - 10]
        binhz = (SR / 2) / (NBINS - 1)
        pb = np.clip(((np.arange(1, 100) * 200.0) / binhz).round().astype(int), 0, NBINS - 1)
        c_h = float(np.median([row[pb].mean() for row in lm_h]))
        c_n = float(np.median(lm_n[:, 50:NBINS - 50]))
        _HPV_CAL["c"] = (c_h, c_n)
    return _HPV_CAL["c"]


def subsample_env(h: np.ndarray, nb: int) -> np.ndarray:
    if nb == NBINS:
        return h
    src = np.linspace(0.0, 1.0, NBINS)
    dst = np.linspace(0.0, 1.0, nb)
    return np.stack([np.interp(dst, src, row) for row in h])


def frame_rms(y: np.ndarray, n: int) -> np.ndarray:
    return np.sqrt((y[:n * HOP].reshape(n, HOP) ** 2).mean(-1) + EPS)


def gain_match(y: np.ndarray, gt: np.ndarray, smooth: int = 9) -> tuple[np.ndarray, float]:
    n = min(len(y), len(gt)) // HOP
    ry, rg = frame_rms(y, n), frame_rms(gt, n)
    if smooth > 1:
        pad = smooth // 2
        k = np.hanning(smooth)
        k = k / k.sum()
        ry = np.convolve(np.pad(ry, pad, mode="edge"), k, mode="valid")[:n]
        rg = np.convolve(np.pad(rg, pad, mode="edge"), k, mode="valid")[:n]
    g = np.clip(rg / np.maximum(ry, 0.05 * np.median(ry) + 1e-8), 0.0, 10.0)
    gs = torch.tensor(g, dtype=torch.float32).view(1, 1, -1)
    gu = torch.nn.functional.interpolate(gs, scale_factor=HOP, mode="linear",
                                         align_corners=False).numpy().ravel()
    out = y[:n * HOP] * gu
    peak = np.abs(out).max()
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out, float(np.median(g))


def sharp_contrast(y: np.ndarray, active: np.ndarray) -> float:
    s = np.abs(librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=HOP))
    f = np.fft.rfftfreq(2048, 1.0 / SR)
    band = np.log(s[(f >= 1000) & (f <= 5000)] + EPS)
    nf = min(band.shape[1], len(active))
    b = band[:, :nf][:, active[:nf]]
    if b.shape[1] < 3:
        return float("nan")
    return float((b.max(0) - np.median(b, 0)).mean())


def contrast_split(y: np.ndarray, gt: np.ndarray, f0: np.ndarray,
                   act: np.ndarray) -> dict:
    n = min(len(act), len(f0))
    v = act[:n] & (f0[:n] > 1.0)
    uv = act[:n] & (f0[:n] <= 1.0)
    out = {}
    for tag, sel in (("v", v), ("uv", uv)):
        cy, cg = sharp_contrast(y, sel), sharp_contrast(gt, sel)
        out[f"contrast_{tag}"] = round(cy, 3)
        out[f"contrast_{tag}_ratio"] = round(cy / (cg + EPS), 3) if np.isfinite(cy) and np.isfinite(cg) else float("nan")
    return out


def active_frames(gt: np.ndarray) -> np.ndarray:
    n = len(gt) // HOP
    r = frame_rms(gt, n)
    return r > 0.05 * np.percentile(r, 99)


def band_metrics(y: np.ndarray) -> dict:
    s = np.abs(librosa.stft(y.astype(np.float32), n_fft=2048, hop_length=HOP)) ** 2
    f = np.fft.rfftfreq(2048, 1.0 / SR)
    tot = s.sum() + EPS
    bands = {"low_0_1k": (0, 1000), "mid_1_4k": (1000, 4000),
             "hi_4_8k": (4000, 8000), "air_8_16k": (8000, 16000)}
    out = {k: float(s[(f >= lo) & (f < hi)].sum() / tot * 100.0) for k, (lo, hi) in bands.items()}
    out["centroid_hz"] = float(librosa.feature.spectral_centroid(S=np.sqrt(s), sr=SR).mean())
    return out


def formant_bw(y: np.ndarray, f0: np.ndarray) -> float:
    fs = 11025
    yd = librosa.resample(y.astype(np.float32), orig_sr=SR, target_sr=fs)
    yd = np.append(yd[0], yd[1:] - 0.97 * yd[:-1])
    hop_d = int(fs * FRAME_MS / 1000.0)
    win = int(0.025 * fs)
    bws = []
    for i, f in enumerate(f0):
        if f <= 1.0:
            continue
        s = i * hop_d
        fr = yd[s:s + win]
        if len(fr) < win or np.abs(fr).max() < 1e-4:
            continue
        try:
            a = librosa.lpc(fr * np.hanning(win), order=12)
        except Exception:
            continue
        r = np.roots(a)
        r = r[np.imag(r) > 0.01]
        fr_hz = np.angle(r) * fs / (2 * np.pi)
        bw = -np.log(np.abs(r)) * fs / np.pi
        sel = (fr_hz > 200) & (fr_hz < 4000) & (bw < 1000)
        if sel.sum() == 0:
            continue
        order_idx = np.argsort(fr_hz[sel])[:3]
        bws.append(float(bw[sel][order_idx].mean()))
    return float(np.mean(bws)) if bws else float("nan")


def mel_l1(y: np.ndarray, gt: np.ndarray) -> float:
    def m(x):
        s = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=SR, n_fft=2048,
                                           hop_length=HOP, n_mels=128)
        return np.log(s + 1e-5)
    n = min(len(y), len(gt))
    a, b = m(y[:n]), m(gt[:n])
    return float(np.abs(a - b).mean())


def sanitize(m: dict) -> dict:
    return {k: (v if not isinstance(v, float) or np.isfinite(v) else None)
            for k, v in m.items()}


def measure(y: np.ndarray, gt: np.ndarray, f0: np.ndarray, act: np.ndarray) -> dict:
    from e0_discriminator_hunt import (m_band_traj_dist, m_line_sharp,
                                       m_lsd_bands, m_mod_spec_dist)
    n = min(len(y), len(gt))
    y, gt = y[:n], gt[:n]
    ctx = {"f0": f0, "act": act, "n": min(len(act), len(f0))}
    tex = {**m_mod_spec_dist(y, gt, ctx), **m_band_traj_dist(y, gt, ctx),
           **m_lsd_bands(y, gt, ctx), **m_line_sharp(y, gt, ctx)}
    c_y, c_g = sharp_contrast(y, act), sharp_contrast(gt, act)
    bm_y, bm_g = band_metrics(y), band_metrics(gt)
    return {
        **{k: round(v, 3) for k, v in tex.items()},
        "contrast": round(c_y, 3), "contrast_gt": round(c_g, 3),
        "contrast_ratio": round(c_y / (c_g + EPS), 3),
        **contrast_split(y, gt, f0, act),
        "centroid_delta_hz": round(bm_y["centroid_hz"] - bm_g["centroid_hz"], 0),
        "low_excess_pt": round(bm_y["low_0_1k"] - bm_g["low_0_1k"], 1),
        "air_share": round(bm_y["air_8_16k"], 1), "air_share_gt": round(bm_g["air_8_16k"], 1),
        "formant_bw_hz": round(formant_bw(y, f0), 0),
        "formant_bw_gt_hz": round(formant_bw(gt, f0), 0),
        "mel_l1": round(mel_l1(y, gt), 3),
    }


def upsample_frames(x: np.ndarray, sub: int) -> np.ndarray:
    T = x.shape[0]
    src = np.arange(T, dtype=np.float64)
    dst = np.linspace(0.0, T - 1.0, T * sub)
    if x.ndim == 1:
        return np.interp(dst, src, x)
    return np.stack([np.interp(dst, src, x[:, j]) for j in range(x.shape[1])], axis=1)


def render_arm(f0: np.ndarray, h_v: np.ndarray, h_n: np.ndarray, gt: np.ndarray,
               k_v: int, nb: int, noise: torch.Tensor, d_mod: float = 0.0,
               phase_mode: str = "min", v_off: float = 0.0,
               n_off: float = 0.0, sub: int = 1, mod_p: int = 1,
               d_arr: np.ndarray | None = None,
               a_arr: np.ndarray | None = None,
               exc: dict | None = None,
               mvf_arr: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    exc = exc or {}
    T = h_v.shape[0]
    f0f = f0[:T].copy()
    if sub > 1:
        vm = upsample_frames((f0f > 1.0).astype(np.float64), sub) > 0.5
        f0h = f0f.copy()
        f0h[f0h <= 1.0] = np.interp(np.where(f0f <= 1.0)[0],
                                    np.where(f0f > 1.0)[0], f0f[f0f > 1.0]) \
            if (f0f > 1.0).any() else 220.0
        f0f = upsample_frames(f0h, sub)
        f0f[~vm] = 0.0
        h_v, h_n = upsample_frames(h_v, sub), upsample_frames(h_n, sub)
        if d_arr is not None:
            d_arr = upsample_frames(d_arr[:T], sub)
        if a_arr is not None:
            a_arr = upsample_frames(a_arr[:T], sub)
        T = T * sub
    r = LtvRenderer(hop=HOP // sub, k_v=k_v, k_n=min(256, k_v), nb_in=nb,
                    phase_mode=phase_mode, mod_p=mod_p,
                    jitter=exc.get("jitter", 0.0), shimmer=exc.get("shimmer", 0.0),
                    disp=exc.get("disp", "none"), disp_c=exc.get("disp_c", 0.0))
    hv = torch.tensor(subsample_env(h_v, nb) + v_off, dtype=torch.float32).unsqueeze(0)
    hn = torch.tensor(subsample_env(h_n, nb) + n_off, dtype=torch.float32).unsqueeze(0)
    f0t = torch.tensor(f0f, dtype=torch.float32).unsqueeze(0)
    d = None
    if d_arr is not None:
        d = torch.tensor(d_arr[:T], dtype=torch.float32).unsqueeze(0)
    elif d_mod > 0.0:
        d = torch.tensor((f0f > 1.0) * d_mod, dtype=torch.float32).unsqueeze(0)
    a = None
    if a_arr is not None:
        a = torch.tensor(a_arr[:T], dtype=torch.float32).unsqueeze(0)
    mv = None
    if mvf_arr is not None:
        if sub > 1:
            mvf_arr = upsample_frames(mvf_arr, sub)
        mv = torch.tensor(mvf_arr[:T], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y = r(f0t, hv, hn, d=d, a=a, noise=noise, mvf=mv)["y"][0].numpy()
    return gain_match(y, gt)


def build_arms(quick: bool, diag: bool = False, modhunt: bool = False) -> list[dict]:
    if modhunt:
        base = {"k": 1024, "nb": 1025}
        b = {"k": 1024, "nb": 2049, "env": "hpv", "n_off": -0.6}
        return [
            {"name": "mh_j03", **b, "exc": {"jitter": 0.003}},
            {"name": "mh_paw", **b, "paw": True},
            {"name": "mh_j03_paw", **b, "paw": True, "exc": {"jitter": 0.003}},
            {"name": "mh_j03_paw_dt", **b, "paw": True, "exc": {"jitter": 0.003},
             "od": True, "p": 4},
        ]
    if diag:
        return [
            {"name": "te_k1024_nb1025", "k": 1024, "nb": 1025},
            {"name": "diag_harm_only", "k": 1024, "nb": 1025, "n_off": -20.0},
            {"name": "diag_noise_only", "k": 1024, "nb": 1025, "v_off": -20.0},
            {"name": "diag_noise_m6db", "k": 1024, "nb": 1025, "n_off": -0.69},
            {"name": "diag_mod_d09", "k": 1024, "nb": 1025, "d": 0.9},
            {"name": "diag_hybrid_env", "k": 1024, "nb": 1025, "env": "hybrid"},
            {"name": "diag_hybrid_mod", "k": 1024, "nb": 1025, "env": "hybrid", "d": 0.65},
            {"name": "diag_tsmooth3", "k": 1024, "nb": 1025, "env": "hybrid", "tsm": 3},
            {"name": "diag_tsmooth5", "k": 1024, "nb": 1025, "env": "hybrid", "tsm": 5},
            {"name": "diag_wexc_ltv", "k": 1024, "nb": 2049, "wexc": True},
            {"name": "diag_hop256", "k": 1024, "nb": 1025, "env": "hybrid", "sub": 2},
            {"name": "diag_hop128", "k": 1024, "nb": 1025, "env": "hybrid", "sub": 4},
        ]
    arms = []
    ks = [256, 1024, 2048] if quick else [256, 512, 1024, 2048]
    nbs = [1025] if quick else [257, 513, 1025, 2049]
    for k in ks:
        for nb in nbs:
            arms.append({"name": f"te_k{k}_nb{nb}", "k": k, "nb": nb})
    if quick:
        arms += [{"name": "te_k1024_nb257", "k": 1024, "nb": 257},
                 {"name": "te_k1024_nb2049", "k": 1024, "nb": 2049}]
    arms += [
        {"name": "te_k1024_nb1025_mod", "k": 1024, "nb": 1025, "d": 0.65},
        {"name": "te_k1024_nb1025_lin", "k": 1024, "nb": 1025, "phase": "linear"},
        {"name": "ct_k1024_nb1025", "k": 1024, "nb": 1025, "env": "cheaptrick"},
        {"name": "hy_k1024_nb1025", "k": 1024, "nb": 1025, "env": "hybrid"},
        {"name": "hy_k1024_nb1025_mod", "k": 1024, "nb": 1025, "env": "hybrid", "d": 0.65},
        {"name": "ctq30_k1024_nb1025", "k": 1024, "nb": 1025, "env": "ct_q1", "q1": -0.30},
        {"name": "ctq30_k1024_nb1025_od_qp4", "k": 1024, "nb": 1025, "env": "ct_q1",
         "q1": -0.30, "od": True, "p": 4},
        {"name": "hpvpaw_k1024", "k": 1024, "nb": 2049, "env": "hpv", "paw": True,
         "n_off": -0.6},
        {"name": "hpvpaw_k1024_j03_od_qp4", "k": 1024, "nb": 2049, "env": "hpv",
         "paw": True, "n_off": -0.6, "od": True, "p": 4, "exc": {"jitter": 0.003}},
    ]
    return arms


def pick_golden(tsv: Path, cats: list[str]) -> list[tuple[str, Path]]:
    rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
    picks = []
    for cat in cats:
        for r in rows:
            if r["category"] == cat:
                p = (Path(__file__).resolve().parent / r["path"]).resolve()
                if p.exists():
                    picks.append((r["golden_id"], p))
                    break
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(ROOT / "data/kansei_vc/golden_mini.tsv"))
    ap.add_argument("--cats", default=",".join(DEFAULT_CATS))
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--modhunt", action="store_true")
    ap.add_argument("--max-sec", type=float, default=8.0)
    args = ap.parse_args()
    stem = "e0_modhunt" if args.modhunt else "e0_diag" if args.diag else "e0_oracle_ltv"
    if args.out is None:
        args.out = str(ROOT / f"results/{stem}")
    if args.json is None:
        args.json = str(ROOT / f"results/{stem}.json")

    if args.files:
        utts, seen = [], {}
        for f in args.files:
            s = Path(f).stem
            seen[s] = seen.get(s, 0) + 1
            utts.append((s if seen[s] == 1 else f"{s}_{seen[s]}", Path(f)))
    else:
        utts = pick_golden(Path(args.golden), args.cats.split(","))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    arms = build_arms(args.quick, args.diag, args.modhunt)
    report = {"date": time.strftime("%Y-%m-%d %H:%M"), "quick": args.quick,
              "diag": args.diag,
              "gate": "contrast_ratio >= 0.95 * world arm (objective half; final=ear)",
              "utts": {}}

    for uid, path in utts:
        x = load_wav(path)
        if len(x) > args.max_sec * SR:
            x = x[:int(args.max_sec * SR)]
        an = analysis_cached(uid, x, ROOT / "results/e0_cache")
        f0, sp, ap_w = an["f0"], an["sp"], an["ap"]
        log_ct, log_te = an["log_ct"], an["log_te"]
        T = len(f0)
        gt = x[:T * HOP].astype(np.float32)
        ap_c = ap_w
        act = active_frames(gt)
        torch.manual_seed(int(hashlib.md5(uid.encode()).hexdigest()[:8], 16))
        noise = torch.randn(1, T * HOP)

        y_world = pw.synthesize(f0.astype(np.float64), sp[:T], ap_w[:T], SR, FRAME_MS)
        y_world = np.pad(y_world, (0, max(0, T * HOP - len(y_world))))[:T * HOP]
        y_world, _ = gain_match(y_world, gt)
        rows = {"world": sanitize(measure(y_world, gt, f0, act))}
        sf.write(outdir / f"{uid}_gt.wav", gt, SR)
        sf.write(outdir / f"{uid}_world.wav", y_world.astype(np.float32), SR)

        hv_te, hn_te = split_envelopes(log_te, ap_c)
        hv_ct, hn_ct = split_envelopes(log_ct, ap_c)
        ctq_cache: dict = {}
        for arm in arms:
            if arm.get("wexc"):
                from ltv_render import MinPhaseFIR, ltv_ola
                e_w = pw.synthesize(f0.astype(np.float64), np.ones_like(sp[:T]),
                                    ap_w[:T], SR, FRAME_MS)
                e_w = np.pad(e_w, (0, max(0, T * HOP - len(e_w))))[:T * HOP]
                fir = MinPhaseFIR(nb_in=NBINS, k=arm["k"])
                with torch.no_grad():
                    b = fir(torch.tensor(log_ct, dtype=torch.float32).unsqueeze(0))
                    y = ltv_ola(torch.tensor(e_w, dtype=torch.float32).unsqueeze(0), b)[0].numpy()
                y, g_med = gain_match(y, gt)
                m = measure(y, gt, f0, act)
                m["gain_med"] = round(g_med, 3)
                rows[arm["name"]] = sanitize(m)
                sf.write(outdir / f"{uid}_{arm['name']}.wav", y.astype(np.float32), SR)
                continue
            if arm.get("env") == "hpv":
                if arm.get("paw"):
                    hkey = "hpv_paw"
                    if hkey not in ctq_cache:
                        ctq_cache[hkey] = hpv_paw_envelopes(x, f0)
                else:
                    win = arm.get("win", NFFT)
                    key = f"spec_logmag_{win}"
                    if key not in ctq_cache:
                        ctq_cache[key] = stft_logmag(x, win=win)[:T]
                    hkey = f"hpv_{win}"
                    if hkey not in ctq_cache:
                        ctq_cache[hkey] = hpv_envelopes(ctq_cache[key], f0)
                hv_g, hn_g = ctq_cache[hkey]
                if arm.get("hn_tsm", 0) > 1:
                    hn_g = tsmooth_env(hn_g, arm["hn_tsm"])
                c_h, c_n = hpv_calibration()
                dh = c_h + 0.5 * np.log(np.maximum(f0, 1.0) / 200.0)
                hv = hv_g - dh[:, None]
                hn = hn_g - c_n
                if arm.get("dsm"):
                    key4 = f"spec_logmag_{NFFT}"
                    if key4 not in ctq_cache:
                        ctq_cache[key4] = stft_logmag(x)[:T]
                    if "mvf" not in ctq_cache:
                        ctq_cache["mvf"] = oracle_mvf(ctq_cache[key4], f0)
                    k15 = 15
                    sm = np.stack([np.convolve(np.pad(row, k15, mode="edge"),
                                               np.ones(2 * k15 + 1) / (2 * k15 + 1),
                                               "valid") for row in ctq_cache[key4]])
                    fgrid = np.linspace(0.0, SR / 2, NBINS)
                    wq = np.clip((fgrid[None, :] - (ctq_cache["mvf"][:, None] - 250.0))
                                 / 500.0, 0.0, 1.0)
                    wlo = 0.5 * (1.0 + np.cos(np.pi * wq))
                    hn = wlo * hn + (1.0 - wlo) * (sm - c_n)
                if arm.get("hnmin"):
                    q1 = -0.30
                    if q1 not in ctq_cache:
                        tpos = np.arange(T, dtype=np.float64) * FRAME_MS / 1000.0
                        spq = pw.cheaptrick(x.astype(np.float64), f0.astype(np.float64),
                                            tpos, SR, q1=q1, fft_size=NFFT)
                        ctq_cache[q1] = 0.5 * np.log(spq[:T] + EPS)
                    key4 = f"spec_logmag_{NFFT}"
                    if key4 not in ctq_cache:
                        ctq_cache[key4] = stft_logmag(x)[:T]
                    du = float(np.median(ctq_cache[key4] - ctq_cache[q1]))
                    hn_ct = (ctq_cache[q1] + du) + np.log(np.clip(ap_c, 1e-5, 1 - 1e-6)) - c_n
                    hn = np.minimum(hn, hn_ct)
            elif arm.get("env") == "mix":
                key = f"spec_logmag_{NFFT}"
                if key not in ctq_cache:
                    ctq_cache[key] = stft_logmag(x)[:T]
                if "hpv_%d" % NFFT not in ctq_cache:
                    ctq_cache["hpv_%d" % NFFT] = hpv_envelopes(ctq_cache[key], f0)
                q1 = arm.get("q1", -0.30)
                if q1 not in ctq_cache:
                    tpos = np.arange(T, dtype=np.float64) * FRAME_MS / 1000.0
                    spq = pw.cheaptrick(x.astype(np.float64), f0.astype(np.float64),
                                        tpos, SR, q1=q1, fft_size=NFFT)
                    ctq_cache[q1] = 0.5 * np.log(spq[:T] + EPS)
                c_h, c_n = hpv_calibration()
                dh = c_h + 0.5 * np.log(np.maximum(f0, 1.0) / 200.0)
                hv = ctq_cache["hpv_%d" % NFFT][0] - dh[:, None]
                du = float(np.median(ctq_cache[key] - ctq_cache[q1]))
                hn = (ctq_cache[q1] + du) + np.log(np.clip(ap_c, 1e-5, 1 - 1e-6)) - c_n
            elif arm.get("env") == "spec":
                win = arm.get("win", NFFT)
                key = f"spec_logmag_{win}"
                if key not in ctq_cache:
                    ctq_cache[key] = stft_logmag(x, win=win)[:T]
                hv, hn = split_envelopes(ctq_cache[key], ap_c,
                                         a0=arm.get("a0", 0.0))
            elif arm.get("env") == "ct_q1":
                q1 = arm["q1"]
                if q1 not in ctq_cache:
                    tpos = np.arange(T, dtype=np.float64) * FRAME_MS / 1000.0
                    spq = pw.cheaptrick(x.astype(np.float64),
                                        f0.astype(np.float64), tpos, SR,
                                        q1=q1, fft_size=NFFT)
                    ctq_cache[q1] = 0.5 * np.log(spq[:T] + EPS)
                hv, hn = split_envelopes(ctq_cache[q1], ap_c, a0=arm.get("a0", 0.0))
            elif arm.get("env") == "cheaptrick":
                hv, hn = hv_ct, hn_ct
            elif arm.get("env") == "hybrid":
                hv, hn = hv_te, hn_ct
            elif arm.get("env") == "hybrid_l":
                delta = (log_te - log_ct).mean(1, keepdims=True)
                hv, hn = hv_te, split_envelopes(log_ct + delta, ap_c)[1]
            elif arm.get("env") == "apw":
                apc2 = np.clip(ap_c, 0.0, 1.0)
                env_apw = (1.0 - apc2) * log_te + apc2 * log_ct
                hv, hn = split_envelopes(env_apw, ap_c)
            elif arm.get("env") == "blend":
                fgrid = np.linspace(0.0, SR / 2, NBINS)
                lo, hi = arm.get("blo", 4000.0), arm.get("bhi", 7000.0)
                w = np.clip((hi - fgrid) / (hi - lo), 0.0, 1.0)[None, :]
                env_bl = log_ct + w * (log_te - log_ct)
                hv = split_envelopes(env_bl, ap_c)[0]
                hn = hn_ct
            else:
                hv, hn = hv_te, hn_te
            if arm.get("tsm", 0) > 1:
                hv, hn = tsmooth_env(hv, arm["tsm"]), tsmooth_env(hn, arm["tsm"])
            if arm.get("mvf") and "mvf" not in ctq_cache:
                key4 = f"spec_logmag_{NFFT}"
                if key4 not in ctq_cache:
                    ctq_cache[key4] = stft_logmag(x)[:T]
                ctq_cache["mvf"] = oracle_mvf(ctq_cache[key4], f0)
            y, g_med = render_arm(f0, hv, hn, gt, arm["k"], arm["nb"], noise,
                                  d_mod=arm.get("d", 0.0),
                                  phase_mode=arm.get("phase", "min"),
                                  v_off=arm.get("v_off", 0.0),
                                  n_off=arm.get("n_off", 0.0),
                                  sub=arm.get("sub", 1),
                                  mod_p=arm.get("p", 1),
                                  d_arr=oracle_d(ap_c, f0) if arm.get("od") else None,
                                  a_arr=oracle_subframe_gain(gt, T) if arm.get("oa") else None,
                                  exc=arm.get("exc"),
                                  mvf_arr=ctq_cache.get("mvf") if arm.get("mvf") else None)
            m = measure(y, gt, f0, act)
            m["gain_med"] = round(g_med, 3)
            rows[arm["name"]] = sanitize(m)
            sf.write(outdir / f"{uid}_{arm['name']}.wav", y.astype(np.float32), SR)
        report["utts"][uid] = {"path": str(path), "n_frames": T,
                               "voiced_pct": round(float((f0 > 1.0).mean() * 100), 1),
                               "arms": rows}
        w = rows["world"]["contrast_ratio"]
        b = rows.get("te_k1024_nb1025", {}).get("contrast_ratio")
        ok = "n/a" if (b is None or w is None) else ("PASS" if b >= 0.95 * w else "FAIL")
        print(f"[{uid}] T={T} world_contrast_ratio={w} te_k1024_nb1025={b} ({ok} objective)")

    ref_arm = "te_k1024_nb1025"
    summary = {}
    tex_summary = {}
    for arm in ["world"] + [a["name"] for a in arms]:
        vals = [u["arms"][arm]["contrast_ratio"] for u in report["utts"].values()
                if arm in u["arms"] and u["arms"][arm]["contrast_ratio"] is not None]
        if vals:
            summary[arm] = round(float(np.mean(vals)), 3)
        tex = {}
        for k in ["mod_8_16k_dist", "mod_2_8k_dist", "trajdist_b4_8k",
                  "lsd_b1_4k", "lsd_b4_8k", "lsharp_dev"]:
            tv = [u["arms"][arm][k] for u in report["utts"].values()
                  if arm in u["arms"] and u["arms"][arm].get(k) is not None]
            if tv:
                tex[k] = round(float(np.mean(tv)), 3)
        if tex:
            tex_summary[arm] = tex
    report["summary_texture"] = tex_summary
    if tex_summary:
        print(f"{'arm':22s} {'mod8_16k':>9} {'mod2_8k':>9} {'traj4_8k':>9} "
              f"{'lsd1_4k':>9} {'lsd4_8k':>9} {'lsharp':>8} {'contrast':>9}")
        for arm, tex in tex_summary.items():
            print(f"{arm:22s} {tex.get('mod_8_16k_dist', float('nan')):>9} "
                  f"{tex.get('mod_2_8k_dist', float('nan')):>9} "
                  f"{tex.get('trajdist_b4_8k', float('nan')):>9} "
                  f"{tex.get('lsd_b1_4k', float('nan')):>9} "
                  f"{tex.get('lsd_b4_8k', float('nan')):>9} "
                  f"{tex.get('lsharp_dev', float('nan')):>8} {summary.get(arm, ''):>9}")
    wm = summary.get("world", float("nan"))
    report["summary_contrast_ratio"] = summary
    pass_per_utt = {}
    for uid, u in report["utts"].items():
        if ref_arm in u["arms"] and "world" in u["arms"]:
            rv = u["arms"][ref_arm]["contrast_ratio"]
            wv = u["arms"]["world"]["contrast_ratio"]
            pass_per_utt[uid] = bool(rv is not None and wv is not None
                                     and rv >= 0.95 * wv)
    report["gate_objective"] = {
        "world_anchor": wm,
        "ref_arm": ref_arm,
        "ref_value": summary.get(ref_arm),
        "pass_per_utt": pass_per_utt,
        "pass_all": bool(pass_per_utt and all(pass_per_utt.values())),
        "pass_mean": bool(summary.get(ref_arm, 0.0) >= 0.95 * wm),
        "pass": bool(pass_per_utt and all(pass_per_utt.values())),
        "note": "objective half only; ear gate via listen_gui.py on " + str(outdir),
    }
    Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"summary_contrast_ratio": summary,
                      "gate_objective": report["gate_objective"]}, indent=2))
    print(f"-> {args.json}\n-> listen: uv run python listen_gui.py --dir {outdir}")


if __name__ == "__main__":
    main()
