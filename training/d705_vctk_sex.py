"""D705(paper_mf §3 A の測り・規則は results/conv_p0/d705/prereg.yaml・学習なし・描画なし): devvctkm の対(同じ番号 j の女声 FE pi = j と男声 ME pi = 100 + j は同じ文の区間・同じ目標)で、
同じ出力の高さ・同じ言語・同じ録音・同じ内容・同じ目標の男声に固有の分 = v1(ME k12)− v1(FE k0)を測る。2 × 2(性別 × k0 / k12)・入力そのもの・出力の f0 も並べる。
    uv run python d705_vctk_sex.py --arm Y58 --out ../results/conv_p0/d705/d705.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from d692_eval import f0_med
from kansei_v1 import KanseiV1

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
S = H / "devvctkm"


def boot(d: np.ndarray, rng: np.random.Generator) -> list:
    bs = [np.median(rng.choice(d, len(d))) for _ in range(2000)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="Y58")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = json.loads((S / "devset.json").read_text())["pairs"]
    fe = {r["pi"]: r for r in P if r["kind"] == "FE"}
    me = {r["pi"] - 100: r for r in P if r["kind"] == "ME"}
    J = sorted(j for j in set(fe) & set(me) if fe[j]["ti"] == me[j]["ti"] and fe[j]["block"] == me[j]["block"])
    key = lambda r: f"p{r['pi']:04d}_t{r['ti']:03d}"
    w = lambda r, k: S / "gate" / f"RUST{k}{a.arm}" / f"{key(r)}_RUST{k}{a.arm}.wav"
    km = KanseiV1("gen")
    rng = np.random.default_rng(705)
    v = {(sx, k): np.array(km.margin([w((fe if sx == "F" else me)[j], k) for j in J])) for sx in ("F", "M") for k in (0, 12)}
    src = {sx: np.array(km.margin([S / "d56" / f"{key((fe if sx == 'F' else me)[j])}_SRC.wav" for j in J])) for sx in ("F", "M")}
    f0 = {(sx, k): np.array([f0_med(w((fe if sx == "F" else me)[j], k)) for j in J]) for sx, k in (("F", 0), ("M", 12))}
    d = v[("M", 12)] - v[("F", 0)]
    st = 12 * np.log2(f0[("M", 12)] / f0[("F", 0)])
    rep: dict = {"arm": a.arm, "fold": "gen", "n_pairs": len(J),
                 "median": {f"{'FE' if sx == 'F' else 'ME'}_k{k}": round(float(np.median(x)), 3) for (sx, k), x in v.items()},
                 "in_band": {f"{'FE' if sx == 'F' else 'ME'}_k{k}": round(float((x <= 0).mean()), 3) for (sx, k), x in v.items()},
                 "src_median": {f"{'FE' if sx == 'F' else 'ME'}": round(float(np.median(x)), 3) for sx, x in src.items()},
                 "f0_out_median_hz": {"FE_k0": round(float(np.nanmedian(f0[("F", 0)])), 1), "ME_k12": round(float(np.nanmedian(f0[("M", 12)])), 1)},
                 "pitch_mismatch_semitones_median": round(float(np.nanmedian(st)), 2),
                 "male_eqpitch": {"paired_median": round(float(np.median(d)), 3), "ci95": boot(d, rng), "frac_male_worse": round(float((d > 0).mean()), 3)},
                 "pitch_up_female": round(float(np.median(v[("F", 12)] - v[("F", 0)])), 3), "pitch_up_male": round(float(np.median(v[("M", 12)] - v[("M", 0)])), 3)}
    m = rep["male_eqpitch"]["paired_median"]
    rep["verdict_rule"] = ("pitch_unmatched" if abs(rep["pitch_mismatch_semitones_median"]) > 2.0 else
                           "male_small" if m <= 0.5 else "male_large" if m >= 1.5 else "male_mid")
    rep["domain_far"] = rep["median"]["FE_k0"] >= 2.0
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
