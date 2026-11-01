"""D545(耳の所見「ジャギー」の候補 = 高域の倍音の崩れ・折り返し像): 倍音間コントラスト(eval_zsvc.contrast と同じ定義: 有声のフレームの log|STFT|(nfft 4096)の帯の中の p90 − p10)を
帯 0.5〜3 kHz(既存)・3〜5 kHz・5〜8 kHz で。有声 = WORLD(dio + stonemask・16 kHz・5 ms)。群 = NAME=<dir>:<接尾辞>・REAL_cen。発話ごと・耳の 3 試行・対の差(群 − 先頭)。
    uv run python d545_hf_contrast.py --set devff5 --groups SRC=<d56>:SRC CP=<dir>:CP CAND=<dir>:OP15 --cen --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P

SR = 48000
BANDS = {"c05_3": (500, 3000), "c3_5": (3000, 5000), "c5_8": (5000, 8000)}
EAR = {"t01": "p0076_t017", "t02": "p0063_t023", "t03": "p0056_t001"}


def measure(x: np.ndarray) -> dict:
    import pyworld as pw
    x16 = resample_poly(x.astype(np.float64), 16000, SR)
    f0, _ = pw.dio(x16, 16000, f0_floor=90, f0_ceil=1100, frame_period=5.0)
    f0 = pw.stonemask(x16, f0, _, 16000)
    f, _, Z = ss.stft(x.astype(np.float64), SR, nperseg=4096, noverlap=4096 - 240)
    L = np.log(np.abs(Z) + 1e-6)
    T = min(L.shape[1], len(f0))
    v = f0[:T] > 0
    out = {}
    for nm, (lo, hi) in BANDS.items():
        if v.sum() < 5:
            out[nm] = float("nan")
            continue
        Lb = L[(f >= lo) & (f <= hi)][:, :T][:, v]
        out[nm] = float(np.mean(np.percentile(Lb, 90, axis=0) - np.percentile(Lb, 10, axis=0)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--cen", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = P.H / a.set
    G = {g.split("=")[0]: (Path(g.split("=")[1].split(":")[0]), g.split(":")[-1]) for g in a.groups}
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    rows: dict = {g: {} for g in G}
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        for g, (d, sfx) in G.items():
            p = d / f"{k}_{sfx}.wav"
            if p.exists():
                x, sr = sf.read(str(p), dtype="float32", always_2d=True)
                x = x.mean(1)[: 8 * sr]
                rows[g][k] = measure(resample_poly(x, SR, sr) if sr != SR else x)
    if a.cen:
        rows["REAL_cen"] = {}
        for t_ in sorted({r["ti"] for r in pairs}):
            p = S / "d37" / f"cen_{t_:03d}.wav"
            if p.exists():
                x, sr = sf.read(str(p), dtype="float32", always_2d=True)
                x = x.mean(1)
                for o in range(0, max(1, len(x) - 8 * sr), 8 * sr):
                    rows["REAL_cen"][f"{t_}_{o}"] = measure(x[o:o + 8 * sr])
    ms = list(BANDS)
    rep = {"summary_median_p15_p85": {g: {m: [round(float(np.nanpercentile([v[m] for v in d.values()], q)), 3) for q in (50, 15, 85)] for m in ms} for g, d in rows.items()}}
    ref = list(G)[0]
    rep["paired_minus_" + ref] = {g: {m: [round(float(np.nanmedian([rows[g][k][m] - rows[ref][k][m] for k in rows[g] if k in rows[ref]])), 3),
                                          round(float(np.nanmean([rows[g][k][m] < rows[ref][k][m] for k in rows[g] if k in rows[ref]])), 3)] for m in ms} for g in G if g != ref}
    rep["ear"] = {t: {g: {m: round(rows[g][k][m], 3) for m in ms} for g in G if k in rows[g]} for t, k in EAR.items()}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps({**rep, "per_utt": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
