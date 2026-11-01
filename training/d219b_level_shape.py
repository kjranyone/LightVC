"""D219 の記述の追加(事後と明記・results/conv_p0/d219_verdict.yaml): log-mel の時間分散(mel 40 帯の平均)を、大きさ(フレームごとの帯の平均)と形(帯の平均を引いた残り)に分ける。
mean_b var(lm_b) = var(level) + mean_b var(lm_b − level)(厳密に足し算)。それぞれの 0–4 Hz の分も出す。目標本人の参照(d37/_work/ref_TTT.wav の先頭 8 秒)とも比べる。
    uv run python d219b_level_shape.py --root <dev の親> --out ../results/conv_p0/d219b.json
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
FPS = N.SR / N.HOP


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import librosa
    RT = Path(a.root)
    fb = librosa.filters.mel(sr=N.SR, n_fft=2048, n_mels=40, fmin=50, fmax=10000)
    lp = ss.butter(4, 4.0, "lowpass", fs=FPS, output="sos")

    def split(x: np.ndarray) -> dict:
        x = x[: int(8.0 * N.SR)].astype(np.float32)
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        n = len(x) // N.HOP
        e = 10 * np.log10((x[: n * N.HOP].reshape(n, N.HOP) ** 2).mean(1) + 1e-12)
        act = e > e.max() - 40
        _, _, Z = ss.stft(x, N.SR, nperseg=2048, noverlap=2048 - N.HOP)
        P = np.abs(Z[:, :n]) ** 2
        act = act[: P.shape[1]]
        lm = 10 * np.log10(fb @ P + 1e-10)
        lv = lm.mean(0)
        sh = lm - lv[None]
        slow = lambda z: ss.sosfiltfilt(lp, z - z.mean(-1, keepdims=True), axis=-1)
        return {"lmstd": float(lm[:, act].std(1).mean()), "var": float(lm[:, act].var(1).mean()), "level": float(lv[act].var()), "shape": float(sh[:, act].var(1).mean()),
                "level_slow": float(slow(lv)[act].var()), "shape_slow": float(slow(sh)[:, act].var(1).mean())}

    rows: dict = {c: [] for c in FORMS}
    tref: dict = {}
    for D in SETS:
        S = RT / D
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            k = f.stem[:-4]
            ti = int(k.split("_t")[1])
            if (D, ti) not in tref:
                tref[(D, ti)] = split(load48(S / "d37/_work" / f"ref_{ti:03d}.wav"))
            for c, d in FORMS.items():
                x, sr = sf.read(S / d / f"{k}_{c}.wav", dtype="float32")
                rows[c].append({**split(x), "set": D, "key": k, "t": ti})
        print(D, flush=True)
    S = RT / "dev"
    excl = set((S / "excl.txt").read_text().split()) if (S / "excl.txt").exists() else set()
    base = ROOT / "female-dataset"
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(109).shuffle(pool)
    real = []
    for spk in pool:
        if len(real) >= 150:
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
        real.append(split(x))
    keys = ["var", "level", "shape", "level_slow", "shape_slow", "lmstd"]
    q = lambda v: {"p15": round(float(np.percentile(v, 15)), 3), "p50": round(float(np.median(v)), 3), "p85": round(float(np.percentile(v, 85)), 3)}
    rep: dict = {"real_pool": {kk: q([r[kk] for r in real]) for kk in keys}, "target_refs": {kk: q([r[kk] for r in tref.values()]) for kk in keys}}
    rep["forms_median"] = {c: {kk: round(float(np.median([r[kk] for r in rows[c]])), 3) for kk in keys} for c in FORMS}
    rep["diff_median"] = {f"{c}-G03S10ZH": {kk: round(float(np.median([x[kk] - y[kk] for x, y in zip(rows[c], rows["G03S10ZH"])])), 3) for kk in keys} for c in ("RRQS18P", "RRQA18P")}
    rep["minus_target_ref_median"] = {c: {kk: round(float(np.median([r[kk] - tref[(r["set"], r["t"])][kk] for r in rows[c]])), 3) for kk in keys} for c in FORMS}
    rep["corr_with_target_ref"] = {c: {kk: round(float(np.corrcoef([r[kk] for r in rows[c]], [tref[(r["set"], r["t"])][kk] for r in rows[c]])[0, 1]), 3) for kk in keys} for c in FORMS}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
