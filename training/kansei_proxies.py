from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np

ANALYSIS_SR = 44100
N_FFT = 2048
HOP = 512
WIN = 2048
EPS = 1e-8

BANDS = {
    "low": (0, 1000),
    "mid": (1000, 4000),
    "presence": (4000, 5000),
    "sibilance": (5000, 9000),
    "brilliance": (9000, 16000),
    "air": (16000, 22050),
}

F0_MIN = 70.0
F0_MAX = 500.0


def load_wav(path, sr: int = ANALYSIS_SR) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _stft_mag(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=WIN))
    freqs = librosa.fft_frequencies(sr=ANALYSIS_SR, n_fft=N_FFT)
    return S, freqs


def _active_frames(S: np.ndarray, rel_db: float = -40.0) -> np.ndarray:
    e = (S ** 2).sum(axis=0) + EPS
    edb = 10 * np.log10(e)
    return edb > (edb.max() + rel_db)


def spectral_metrics(y: np.ndarray, sr: int = ANALYSIS_SR) -> dict:
    import librosa
    S, freqs = _stft_mag(y)
    act = _active_frames(S)
    if act.sum() == 0:
        act = np.ones(S.shape[1], dtype=bool)
    Sa = S[:, act]
    power = Sa ** 2
    total = power.sum(axis=0) + EPS

    out = {}
    for name, (lo, hi) in BANDS.items():
        if lo >= sr / 2:
            out[f"band_{name}"] = 0.0
            continue
        m = (freqs >= lo) & (freqs < min(hi, sr / 2))
        frac = power[m, :].sum(axis=0) / total
        out[f"band_{name}"] = float(np.mean(frac))

    out["hf_ratio"] = float(np.mean(power[freqs >= 8000, :].sum(axis=0) / total))
    out["sib_ratio"] = out["band_sibilance"]

    centroid = librosa.feature.spectral_centroid(S=Sa, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=Sa, sr=sr, roll_percent=0.85)[0]
    out["centroid_hz"] = float(np.mean(centroid))
    out["rolloff85_hz"] = float(np.mean(rolloff))

    flat = librosa.feature.spectral_flatness(S=Sa)[0]
    out["flatness"] = float(np.mean(flat))
    hf_mask = freqs >= 8000
    if hf_mask.sum() > 4:
        hf = Sa[hf_mask, :]
        gm = np.exp(np.mean(np.log(hf + EPS), axis=0))
        am = np.mean(hf, axis=0) + EPS
        out["hf_flatness"] = float(np.mean(gm / am))
    else:
        out["hf_flatness"] = 0.0

    tilt_mask = (freqs > 100) & (freqs < 8000)
    logmag = 20 * np.log10(np.mean(Sa[tilt_mask, :], axis=1) + EPS)
    fk = freqs[tilt_mask] / 1000.0
    if len(fk) > 2:
        A = np.vstack([fk, np.ones_like(fk)]).T
        slope, _ = np.linalg.lstsq(A, logmag, rcond=None)[0]
        out["tilt_db_per_khz"] = float(slope)
    else:
        out["tilt_db_per_khz"] = 0.0

    e48 = power[(freqs >= 4000) & (freqs < 8000), :].sum() + EPS
    e812 = power[(freqs >= 8000) & (freqs < 12000), :].sum()
    out["eight_k_cliff"] = float(e812 / e48)
    return out


def cpp_mean(y: np.ndarray, sr: int = ANALYSIS_SR) -> float:
    S, _ = _stft_mag(y)
    act = _active_frames(S)
    if act.sum() == 0:
        return 0.0
    log_mag_db = 20 * np.log10(S[:, act] + EPS)
    cep = np.fft.irfft(log_mag_db, n=N_FFT, axis=0)
    q = np.arange(N_FFT) / sr
    qmin, qmax = 1.0 / F0_MAX, 1.0 / F0_MIN
    win = (q >= qmin) & (q <= qmax)
    if win.sum() < 4:
        return 0.0
    qw = q[win]
    cepw = cep[win, :]
    A = np.vstack([qw, np.ones_like(qw)]).T
    cpps = []
    for n in range(cepw.shape[1]):
        col = cepw[:, n]
        coef, _, _, _ = np.linalg.lstsq(A, col, rcond=None)
        baseline = A @ coef
        prom = col - baseline
        cpps.append(prom.max())
    return float(np.mean(cpps))


def hnr_mean(y: np.ndarray, sr: int = ANALYSIS_SR) -> float:
    frame = int(0.04 * sr)
    hop = int(0.01 * sr)
    lag_min = int(sr / F0_MAX)
    lag_max = int(sr / F0_MIN)
    vals = []
    for start in range(0, max(1, len(y) - frame), hop):
        seg = y[start:start + frame]
        if len(seg) < frame:
            break
        seg = seg - seg.mean()
        e = np.sum(seg ** 2)
        if e < EPS:
            continue
        seg = seg * np.hanning(len(seg))
        r = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        r0 = r[0] + EPS
        rn = r / r0
        if lag_max >= len(rn):
            continue
        peak = np.max(rn[lag_min:lag_max])
        peak = min(max(peak, 1e-6), 1 - 1e-6)
        if peak > 0.3:
            vals.append(10 * np.log10(peak / (1 - peak)))
    return float(np.mean(vals)) if vals else 0.0


def h1h2_mean(y: np.ndarray, sr: int = ANALYSIS_SR) -> float:
    import librosa
    try:
        f0, vflag, _ = librosa.pyin(
            y, fmin=F0_MIN, fmax=F0_MAX, sr=sr,
            frame_length=N_FFT, hop_length=HOP)
    except Exception:
        return 0.0
    S, freqs = _stft_mag(y)
    n = min(S.shape[1], len(f0))
    vals = []
    df = freqs[1] - freqs[0]
    for i in range(n):
        if not vflag[i] or np.isnan(f0[i]):
            continue
        f = f0[i]
        h1 = _harmonic_amp(S[:, i], freqs, f, df)
        h2 = _harmonic_amp(S[:, i], freqs, 2 * f, df)
        if h1 > 0 and h2 > 0:
            vals.append(20 * np.log10(h1) - 20 * np.log10(h2))
    return float(np.mean(vals)) if vals else 0.0


def _harmonic_amp(col: np.ndarray, freqs: np.ndarray, fh: float, df: float, tol_bins: int = 3) -> float:
    if fh >= freqs[-1]:
        return 0.0
    center = int(round(fh / df))
    lo = max(0, center - tol_bins)
    hi = min(len(col), center + tol_bins + 1)
    return float(np.max(col[lo:hi])) if hi > lo else 0.0


def _log_mel(y: np.ndarray, sr: int = ANALYSIS_SR, n_mels: int = 80) -> np.ndarray:
    import librosa
    M = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=n_mels,
        fmax=sr / 2)
    return librosa.power_to_db(M + EPS)


def smoothness_metrics(y: np.ndarray, sr: int = ANALYSIS_SR) -> dict:
    logmel = _log_mel(y, sr)
    d = np.diff(logmel, axis=1)
    dmag = np.sqrt((d ** 2).sum(axis=0))
    out = {
        "mel_delta_mean": float(np.mean(dmag)),
        "mel_delta_var": float(np.var(dmag)),
        "mel_delta_p95": float(np.percentile(dmag, 95)),
    }
    out.update(_jitter_shimmer(y, sr))
    return out


def _jitter_shimmer(y: np.ndarray, sr: int) -> dict:
    import librosa
    try:
        f0, vflag, _ = librosa.pyin(
            y, fmin=F0_MIN, fmax=F0_MAX, sr=sr,
            frame_length=N_FFT, hop_length=HOP)
    except Exception:
        return {"jitter": 0.0, "shimmer": 0.0}
    f0v = f0[np.isfinite(f0)]
    jitter = 0.0
    if len(f0v) > 2:
        periods = 1.0 / f0v
        jitter = float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + EPS))
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    rmsv = rms[rms > (rms.max() * 0.1 + EPS)]
    shimmer = 0.0
    if len(rmsv) > 2:
        shimmer = float(np.mean(np.abs(np.diff(rmsv))) / (np.mean(rmsv) + EPS))
    return {"jitter": jitter, "shimmer": shimmer}


def boundary_discontinuity(y: np.ndarray, sr: int = ANALYSIS_SR, chunk_ms: float = 20.0) -> float:
    logmel = _log_mel(y, sr)
    d = np.sqrt((np.diff(logmel, axis=1) ** 2).sum(axis=0))
    if len(d) < 4:
        return 0.0
    frames_per_chunk = max(1, int(round((chunk_ms / 1000.0) * sr / HOP)))
    idx = np.arange(len(d))
    boundary = (idx % frames_per_chunk) == 0
    if boundary.sum() == 0 or (~boundary).sum() == 0:
        return 1.0
    return float((d[boundary].mean() + EPS) / (d[~boundary].mean() + EPS))


def analyze(y: np.ndarray, sr: int = ANALYSIS_SR, full: bool = True) -> dict:
    out = {"duration_s": float(len(y) / sr), "rms": float(np.sqrt(np.mean(y ** 2) + EPS))}
    out.update(spectral_metrics(y, sr))
    out["cpp_db"] = cpp_mean(y, sr)
    out["hnr_db"] = hnr_mean(y, sr)
    if full:
        out["h1h2_db"] = h1h2_mean(y, sr)
        out.update(smoothness_metrics(y, sr))
        out["boundary_20ms"] = boundary_discontinuity(y, sr, 20.0)
    return out


def _match_len(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 4096) -> int:
    from scipy.signal import fftconvolve
    n = min(len(a), len(b))
    if n < 2048:
        return 0
    corr = fftconvolve(a[:n], b[:n][::-1], mode="full")
    mid = n - 1
    lo, hi = mid - max_lag, mid + max_lag + 1
    return int(np.argmax(corr[lo:hi]) - max_lag)


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lag = _best_lag(a, b)
    if lag > 0:
        b = b[lag:]
    elif lag < 0:
        a = a[-lag:]
    return _match_len(a, b)


def compare(orig: np.ndarray, recon: np.ndarray, sr: int = ANALYSIS_SR, full: bool = True) -> dict:
    orig, recon = _match_len(orig, recon)
    mo = analyze(orig, sr, full=full)
    mr = analyze(recon, sr, full=full)

    def d(k: str) -> float:
        return mr.get(k, 0.0) - mo.get(k, 0.0)

    def ratio(k: str) -> float:
        return mr.get(k, 0.0) / (mo.get(k, 0.0) + EPS)

    out = {
        "centroid_delta_hz": d("centroid_hz"),
        "rolloff85_delta_hz": d("rolloff85_hz"),
        "hf_preserve": ratio("hf_ratio"),
        "brilliance_preserve": ratio("band_brilliance"),
        "eight_k_cliff_ratio": ratio("eight_k_cliff"),
        "cpp_delta_db": d("cpp_db"),
        "hnr_delta_db": d("hnr_db"),
        "h1h2_delta_db": d("h1h2_db"),
        "sib_delta": d("sib_ratio"),
        "hf_flatness_delta": d("hf_flatness"),
        "mel_delta_var_ratio": ratio("mel_delta_var"),
    }
    oa, ra = _align(orig, recon)
    out["log_spectral_distance_db"] = _lsd(oa, ra)
    out["mel_l1_db"] = _mel_l1(oa, ra, sr)
    return {"orig": mo, "recon": mr, "delta": out}


def _lsd(a: np.ndarray, b: np.ndarray) -> float:
    import librosa
    A = 20 * np.log10(np.abs(librosa.stft(a, n_fft=N_FFT, hop_length=HOP)) + EPS)
    B = 20 * np.log10(np.abs(librosa.stft(b, n_fft=N_FFT, hop_length=HOP)) + EPS)
    n = min(A.shape[1], B.shape[1])
    return float(np.sqrt(np.mean((A[:, :n] - B[:, :n]) ** 2)))


def _mel_l1(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    A = _log_mel(a, sr)
    B = _log_mel(b, sr)
    n = min(A.shape[1], B.shape[1])
    return float(np.mean(np.abs(A[:, :n] - B[:, :n])))


def _round(d, n: int = 4):
    if isinstance(d, dict):
        return {k: _round(v, n) for k, v in d.items()}
    if isinstance(d, float):
        return round(d, n)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("analyze")
    pa.add_argument("wavs", nargs="+")
    pa.add_argument("--fast", action="store_true", help="skip pyin-based metrics")
    pc = sub.add_parser("compare")
    pc.add_argument("orig")
    pc.add_argument("recon")
    args = ap.parse_args()

    if args.cmd == "analyze":
        res = {}
        for w in args.wavs:
            y = load_wav(w)
            res[Path(w).name] = _round(analyze(y, full=not args.fast))
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.cmd == "compare":
        o = load_wav(args.orig)
        r = load_wav(args.recon)
        print(json.dumps(_round(compare(o, r)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
