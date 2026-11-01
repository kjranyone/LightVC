"""D222(記述・results/conv_p0/d222_prereg.yaml): mel 40 帯ごとの時間方向の std の、出力 − 目標本人の参照の差(d109 v2 と同じ前処理)。
    uv run python d222_band_std.py --root <dev の親> --out ../results/conv_p0/d222.json
--lifter Q(記述の追加・事後): パワースペクトルの対数をケプストラムで Q 標本(48 kHz)より上を切って倍音の構造を除いた包絡で同じ測定(包絡の動きと倍音の帯またぎを分ける)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf

import nvoc as N
from d6_unit_cov import load48

SETS = ("dev", "dev2", "dev3")
FORMS = {"G03S10ZH": "d114", "RRQS18P": "d139", "RRQSD18P": "d139", "RRQSX18P": "d139"}
EDGES = (0.0, 500.0, 1500.0, 4000.0, 10001.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lifter", type=int, default=0)
    a = ap.parse_args()
    import librosa
    RT = Path(a.root)
    fb = librosa.filters.mel(sr=N.SR, n_fft=2048, n_mels=40, fmin=50, fmax=10000)
    fc = librosa.mel_frequencies(n_mels=42, fmin=50, fmax=10000)[1:-1]
    grp = np.digitize(fc, EDGES[1:-1])

    def bstd(x: np.ndarray) -> np.ndarray:
        x = x[: int(8.0 * N.SR)].astype(np.float32)
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        n = len(x) // N.HOP
        e = 10 * np.log10((x[: n * N.HOP].reshape(n, N.HOP) ** 2).mean(1) + 1e-12)
        act = e > e.max() - 40
        _, _, Z = ss.stft(x, N.SR, nperseg=2048, noverlap=2048 - N.HOP)
        P = np.abs(Z[:, :n]) ** 2
        act = act[: P.shape[1]]
        if a.lifter:
            L = np.log(P + 1e-12)
            cep = np.fft.irfft(L, axis=0)
            cep[a.lifter: cep.shape[0] - a.lifter + 1] = 0.0
            P = np.exp(np.fft.rfft(cep, axis=0).real)
        return (10 * np.log10(fb @ P[:, act] + 1e-10)).std(1)

    d: dict = {c: [] for c in FORMS}
    tr: dict = {}
    for D in SETS:
        S = RT / D
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            k = f.stem[:-4]
            ti = int(k.split("_t")[1])
            if (D, ti) not in tr:
                tr[(D, ti)] = bstd(load48(S / "d37/_work" / f"ref_{ti:03d}.wav"))
            for c, dd in FORMS.items():
                x, _ = sf.read(S / dd / f"{k}_{c}.wav", dtype="float32")
                d[c].append(bstd(x) - tr[(D, ti)])
    rep: dict = {"band_center_hz": [round(float(v)) for v in fc], "groups": [f"{EDGES[i]:g}-{EDGES[i + 1]:g}" for i in range(len(EDGES) - 1)],
                 "n_bands_per_group": [int((grp == g).sum()) for g in range(len(EDGES) - 1)]}
    for c in FORMS:
        m = np.median(np.stack(d[c]), 0)
        contrib = [float(m[grp == g].sum() / 40) for g in range(len(EDGES) - 1)]
        rep[c] = {"lmstd_minus_target_ref": round(float(np.median([x.mean() for x in d[c]])), 3), "per_band_median": [round(float(v), 3) for v in m],
                  "group_contrib_to_lmstd": [round(v, 3) for v in contrib], "group_share": [round(v / sum(contrib), 3) for v in contrib]}
    sh = rep["G03S10ZH"]["group_share"][0]
    rep["P_low"] = sh
    rep["verdict"] = "LOW" if sh >= 0.5 else "MIDHIGH" if sh <= 0.3 else "MIXED"
    print(json.dumps({k: v for k, v in rep.items() if k != "band_center_hz"}, indent=1))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
