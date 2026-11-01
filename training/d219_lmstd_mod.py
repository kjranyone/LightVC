"""D219(学習なし・results/conv_p0/d219_prereg.yaml): lmstd の上乗せ(RRQS18P − G03S10ZH)を log-mel の時間変調の帯に分ける。
前処理は d109 v2 と同じ(先頭 8 秒・RMS 0.05・STFT 2048 / hop 240・mel 40 帯・発声中のフレーム)。帯の分け方はゼロ位相の帯域通過(解析専用)。
    uv run python d219_lmstd_mod.py --root <dev の親> --out ../results/conv_p0/d219.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf

import nvoc as N
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
SETS = ("dev", "dev2", "dev3")
FORMS = {"RRQS18P": "d139", "RRQA18P": "d139", "G03S10ZH": "d114"}
BANDS = ((0.0, 4.0), (4.0, 12.0), (12.0, 25.0), (25.0, 100.0))
FPS = N.SR / N.HOP


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_pool", type=int, default=150)
    a = ap.parse_args()
    import librosa
    RT = Path(a.root)
    fb = librosa.filters.mel(sr=N.SR, n_fft=2048, n_mels=40, fmin=50, fmax=10000)
    sos = [ss.butter(4, hi, "lowpass", fs=FPS, output="sos") if lo == 0 else ss.butter(4, lo, "highpass", fs=FPS, output="sos") if hi >= FPS / 2 else
           ss.butter(4, (lo, hi), "bandpass", fs=FPS, output="sos") for lo, hi in BANDS]

    def decomp(x: np.ndarray, want: float | None) -> dict:
        x = x[: int(8.0 * N.SR)].astype(np.float32)
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        n = len(x) // N.HOP
        e = 10 * np.log10((x[: n * N.HOP].reshape(n, N.HOP) ** 2).mean(1) + 1e-12)
        act = e > e.max() - 40
        _, _, Z = ss.stft(x, N.SR, nperseg=2048, noverlap=2048 - N.HOP)
        P = np.abs(Z[:, :n]) ** 2
        act = act[: P.shape[1]]
        lm = 10 * np.log10(fb @ P + 1e-10)
        lms = float(lm[:, act].std(1).mean())
        if want is not None:
            assert abs(lms - want) < 1e-4, (lms, want)
        r = {"lmstd": lms, "var": float(lm[:, act].var(1).mean())}
        for (lo, hi), s_ in zip(BANDS, sos):
            r[f"{lo:g}-{hi:g}"] = float(ss.sosfiltfilt(s_, lm - lm.mean(1, keepdims=True), axis=1)[:, act].var(1).mean())
        return r

    rows: dict = {c: [] for c in FORMS}
    chk = 0
    for D in SETS:
        S = RT / D
        cache = json.loads((S / "d109_cache.json").read_text())
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            k = f.stem[:-4]
            for c, d in FORMS.items():
                w = S / d / f"{k}_{c}.wav"
                st = cache.get(f"2|set|{w}|{w.stat().st_mtime_ns}")
                x, sr = sf.read(w, dtype="float32")
                assert sr == N.SR
                r = decomp(x, st["lmstd"] if st else None)
                chk += st is not None
                r.update({"set": D, "key": k, "shimmer": st["shimmer"] if st else float("nan")})
                rows[c].append(r)
        print(D, "done", flush=True)
    S = RT / "dev"
    excl = set((S / "excl.txt").read_text().split()) if (S / "excl.txt").exists() else set()
    base = ROOT / "female-dataset"
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(109).shuffle(pool)
    real = []
    for spk in pool:
        if len(real) >= a.n_pool:
            break
        ws = sorted((base / spk).glob("*.wav"))
        if len(ws) < 3 or any(sf.info(str(w)).samplerate < 44100 for w in ws[:3]):
            continue
        x = load48(ws[len(ws) // 2])
        if len(x) < 3 * N.SR:
            continue
        rest = np.concatenate([load48(w) for w in ws if w != ws[len(ws) // 2]][:6])[: int(30 * N.SR)]
        if len(rest) < 8 * N.SR:
            continue
        real.append(decomp(x, None))
    keys = ["var"] + [f"{lo:g}-{hi:g}" for lo, hi in BANDS]
    rep: dict = {"checked_against_d109": chk, "n_real": len(real)}
    rep["levels_median"] = {c: {kk: round(float(np.median([r[kk] for r in rows[c]])), 3) for kk in keys + ["lmstd"]} for c in FORMS}
    rep["real"] = {kk: {"p15": round(float(np.percentile([r[kk] for r in real], 15)), 3), "p50": round(float(np.median([r[kk] for r in real])), 3),
                        "p85": round(float(np.percentile([r[kk] for r in real], 85)), 3)} for kk in keys + ["lmstd"]}
    rng = np.random.default_rng(0)
    for c in ("RRQS18P", "RRQA18P"):
        d = {kk: np.array([x[kk] - y[kk] for x, y in zip(rows[c], rows["G03S10ZH"])]) for kk in keys}
        assert all(x["key"] == y["key"] for x, y in zip(rows[c], rows["G03S10ZH"]))
        med = {kk: float(np.median(v)) for kk, v in d.items()}
        bs = [sum(float(np.median(d[kk][ii])) for kk in keys[3:]) / sum(float(np.median(d[kk][ii])) for kk in keys[1:])
              for ii in (rng.integers(0, len(d["var"]), len(d["var"])) for _ in range(2000))]
        F = sum(med[kk] for kk in keys[3:]) / sum(med[kk] for kk in keys[1:])
        rep[f"{c}-G03S10ZH"] = {"median_diff": {kk: round(v, 3) for kk, v in med.items()}, "F_fast": round(F, 3), "F_fast_ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                                "share": {kk: round(med[kk] / sum(med[k2] for k2 in keys[1:]), 3) for kk in keys[1:]}}
    dsh = np.array([x["shimmer"] - y["shimmer"] for x, y in zip(rows["RRQS18P"], rows["G03S10ZH"])])
    dfa = np.array([sum(x[kk] - y[kk] for kk in keys[3:]) for x, y in zip(rows["RRQS18P"], rows["G03S10ZH"])])
    dsl = np.array([sum(x[kk] - y[kk] for kk in keys[1:3]) for x, y in zip(rows["RRQS18P"], rows["G03S10ZH"])])
    ok = np.isfinite(dsh)
    rep["shimmer_corr"] = {"fast": round(float(np.corrcoef(dsh[ok], dfa[ok])[0, 1]), 3), "slow": round(float(np.corrcoef(dsh[ok], dsl[ok])[0, 1]), 3), "n": int(ok.sum())}
    F = rep["RRQS18P-G03S10ZH"]["F_fast"]
    rep["verdict"] = "FAST" if F >= 0.6 else "SLOW" if F <= 0.4 else "MIXED"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
