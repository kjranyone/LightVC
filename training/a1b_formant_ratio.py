"""A-1b 男→女のフォルマント比(Fant の k 因子と同じ量)を、A-1 の整列フレーム包絡から測る。

A-1 の帯域別 α はフレームごとの探索が悪条件(狭い帯域に山が 1〜2 個で、どの α でも相関が高い: F1 域の 46% が探索端)だったため、
ここでは包絡の山(F1: 200–1000Hz・F2: 800–2800Hz・F3: 1800–3800Hz の各帯で最大の極大)を拾い、整列した男女フレームの比の
中央値を話者ペアごと・母音クラスごとに集計する。山が無い帯のフレームは除外。

    uv run python a1b_formant_ratio.py   # results/artic_a1/formant_ratio.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
A1 = ROOT / "results/artic_a1"
FGRID = np.geomspace(150.0, 7000.0, 160)
FB = {"F1": (200.0, 1000.0), "F2": (800.0, 2800.0), "F3": (1800.0, 3800.0)}


def peaks(E: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (FGRID >= lo) & (FGRID <= hi)
    idx = np.nonzero(m)[0]
    sub = E[:, idx]
    loc = (sub[:, 1:-1] > sub[:, :-2]) & (sub[:, 1:-1] >= sub[:, 2:])
    val = np.where(loc, sub[:, 1:-1], -np.inf)
    k = np.argmax(val, 1)
    ok = np.isfinite(val[np.arange(len(k)), k])
    i = idx[1:-1][k]
    a, b, c = E[np.arange(len(i)), i - 1], E[np.arange(len(i)), i], E[np.arange(len(i)), i + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
    lf = np.log(FGRID[i]) + off * (np.log(FGRID[1]) - np.log(FGRID[0]))
    return np.where(ok, np.exp(lf), np.nan)


def main() -> int:
    z = np.load(A1 / "vtl_warp_frames.npz")
    Em, Ef, lab, pid = z["Em"].astype(np.float64), z["Ef"].astype(np.float64), z["lab"], z["pid"]
    meta = json.loads((A1 / "vtl_warp.json").read_text())
    rep = {"method": "包絡の山(帯内最大の極大・放物線補間)の男女比。整列は A-1 の DTW。", "overall": {}, "per_pair": [], "by_vowel_cluster": {}}
    R = {}
    for fn, (lo, hi) in FB.items():
        pm, pf = peaks(Em, lo, hi), peaks(Ef, lo, hi)
        R[fn] = pf / pm
        ok = np.isfinite(R[fn])
        rep["overall"][fn] = {"ratio_median": round(float(np.nanmedian(R[fn])), 4),
                              "ratio_iqr": [round(float(np.nanpercentile(R[fn], 25)), 3), round(float(np.nanpercentile(R[fn], 75)), 3)],
                              "male_median_hz": round(float(np.nanmedian(pm))), "female_median_hz": round(float(np.nanmedian(pf))),
                              "frac_frames": round(float(ok.mean()), 3)}
    for i, pr in enumerate(meta["pairs"]):
        s = pid == i
        rep["per_pair"].append({"pair": pr, **{fn: round(float(np.nanmedian(R[fn][s])), 4) for fn in FB}})
    pp = {fn: np.array([p[fn] for p in rep["per_pair"]]) for fn in FB}
    rep["pair_summary"] = {fn: {"median": round(float(np.median(v)), 4), "p10": round(float(np.percentile(v, 10)), 4),
                                "p90": round(float(np.percentile(v, 90)), 4)} for fn, v in pp.items()}
    for k in sorted(set(lab.tolist())):
        s = lab == k
        rep["by_vowel_cluster"][f"c{k}"] = {"n": int(s.sum()),
                                            **{f"male_{fn}_hz": round(float(np.nanmedian(peaks(Em[s], *FB[fn])))) for fn in FB},
                                            **{f"{fn}_ratio": round(float(np.nanmedian(R[fn][s])), 4) for fn in FB}}
    (A1 / "formant_ratio.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("overall", "pair_summary")}, ensure_ascii=False, indent=1))
    print(json.dumps(rep["by_vowel_cluster"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
