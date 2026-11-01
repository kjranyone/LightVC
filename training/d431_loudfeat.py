"""D431: 音量の条件 c0n の作り直しの設計の証拠。実女声の区間(e2e/wav・40 本)で、c0n(CheapTrick の mel 包絡の DCT c0 = 学習の Front)と候補の音量の特徴
(L4k = 100 Hz〜4 kHz の帯域のパワーの log・L8k = 〜8 kHz・Lfull = 全帯域・mel の平均 0〜4 kHz)が、録音の経路の摂動(低域通過 8k / 5k・2 kHz の棚 ±9 dB)でどれだけずれるか(フレームの平均のずれ・dB 相当)と、
元の c0n とのフレームの相関(音量の動きを保つか)を測る。CPU だけ。
    uv run python d431_loudfeat.py --out ../results/conv_p0/d431.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import nvoc as N
import pae as PA
import train_rvoc as TR

PERTS = {"lp8k": (8000.0, 0.0), "lp5k": (5000.0, 0.0), "shelf+9": (0.0, 9.0), "shelf-9": (0.0, -9.0)}


def band_logpow(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    T = (len(x) - 2 * PA.A) // N.HOP
    w = np.hanning(1024)
    f = np.fft.rfftfreq(1024, 1 / N.SR)
    sel = (f >= lo) & (f < hi)
    out = np.zeros(T)
    for t in range(T):
        c = PA.A + t * N.HOP
        seg = x[c - 512:c + 512]
        if len(seg) < 1024:
            seg = np.pad(seg, (0, 1024 - len(seg)))
        out[t] = 10 * np.log10((np.abs(np.fft.rfft(seg * w)) ** 2)[sel].sum() + 1e-10)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    ws = sorted(glob.glob("/home/kojirotanaka/lightvc_scratch_20261013/e2e/wav/*.wav"))[: a.n]
    feats = {"c0n_x10dB": None, "L4k": (100, 4000), "L8k": (100, 8000), "Lfull": (50, 24000)}
    shift: dict = {}
    corr: dict = {}
    for w in ws:
        x, sr = sf.read(w, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        x = x[: 48000 * 4]
        T = (len(x) - 2 * PA.A) // N.HOP
        f0 = np.full(T, 220.0, np.float32)

        def all_feats(y: np.ndarray) -> dict:
            out = {"c0n_x10dB": PA.envelope(y, f0)[0] / 10 * (20 / np.log(10)) / 1.0}
            for k, v in feats.items():
                if v is not None:
                    out[k] = band_logpow(y, *v)
            return out
        F0 = all_feats(x)
        e = F0["Lfull"]
        voiced = e > np.percentile(e, 40)
        for k in F0:
            corr.setdefault(k, []).append(float(np.corrcoef(F0[k][voiced], F0["L4k"][voiced])[0, 1]))
        for pn, prm in PERTS.items():
            F1 = all_feats(TR.channel_apply(x, prm))
            for k in F0:
                shift.setdefault((pn, k), []).append(float((F1[k] - F0[k])[voiced].mean()))
    rep = {"n_clips": len(ws), "note": "c0n_x10dB = 学習の c0(自然対数の包絡の平均)を dB に換算(× 20 / ln 10)", "shift_db": {}, "corr_with_L4k": {k: round(float(np.mean(v)), 3) for k, v in corr.items()}}
    for (pn, k), v in shift.items():
        rep["shift_db"].setdefault(pn, {})[k] = round(float(np.mean(v)), 2)
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
