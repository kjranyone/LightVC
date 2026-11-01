"""D704(paper_mf E-C1d・規則は results/conv_p0/d704/prereg.yaml)の評価: 官能指数 v1(gen fold)。
(a) 符号化の代価 = 同じ鍵の v1(mp3 の組)− v1(元の組)(Y58・N67・N71・VCTK の男声 k12 / k5・devff2 k0)。
(b) mp3 の組での覆いの移り = N71 − N67 の対の差の DiD(女声の対照 = devff2_mp3 k0)。
(c) 2 人目の日本語の本物の男声(d704_nmk)の水準と N71 − N67 の DiD(女声の対照 = 元の devff2 k0)。外郎売の水準と並べる。
    uv run python d704_eval.py --out ../results/conv_p0/d704/d704.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kansei_v1 import KanseiV1

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
ARMS = ("Y58", "N67", "N71")
COND = {"vctk_k12": ("devvctkm", "ME", "RUST12"), "vctk_k5": ("devvctkm", "ME", "RUST5Q3"), "ff2_k0": ("devff2", None, "RUST0"),
        "uiro_k12": ("d487_uiro", None, "RUST12"), "uiro_k5": ("d487_uiro", None, "RUST5Q3"), "nmk_k12": ("d704_nmk", None, "RUST12"), "nmk_k5": ("d704_nmk", None, "RUST5Q3")}


def keys(s: str, kind: str | None) -> list:
    return [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def ci(d: np.ndarray, rng: np.random.Generator, ctl: np.ndarray | None = None) -> list:
    bs = [np.median(rng.choice(d, len(d))) - (np.median(rng.choice(ctl, len(ctl))) if ctl is not None else 0.0) for _ in range(2000)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    km = KanseiV1("gen")
    rng = np.random.default_rng(704)
    V: dict = {}

    def v(s: str, kind: str | None, d: str, arm: str, ks: list) -> np.ndarray:
        if (s, d, arm) not in V:
            V[(s, d, arm)] = np.array(km.margin([H / s / "gate" / f"{d}{arm}" / f"{k}_{d}{arm}.wav" for k in ks]))
        return V[(s, d, arm)]

    rep: dict = {"fold": "gen", "codec": {}, "mp3_cov": {}, "nmk": {}}
    for nm in ("vctk_k12", "vctk_k5", "ff2_k0"):
        s, kind, d = COND[nm]
        ks = keys(f"{s}_mp3", kind)
        assert set(ks) <= set(keys(s, kind))
        rep["codec"][nm] = {"n": len(ks)}
        for arm in ARMS:
            c, m = v(s, kind, d, arm, ks), v(f"{s}_mp3", kind, d, arm, ks)
            rep["codec"][nm][arm] = {"clean_median": round(float(np.median(c)), 3), "mp3_median": round(float(np.median(m)), 3),
                                     "cost_paired_median": round(float(np.median(m - c)), 3), "cost_ci95": ci(m - c, rng), "frac_worse": round(float((m > c).mean()), 3)}
        print(nm, rep["codec"][nm], flush=True)
    sm = lambda nm: (f"{COND[nm][0]}_mp3", COND[nm][1], COND[nm][2])
    g = {}
    for nm in ("vctk_k12", "vctk_k5", "ff2_k0"):
        s, kind, d = sm(nm)
        ks = keys(s, kind)
        g[nm] = v(s, kind, d, "N71", ks) - v(s, kind, d, "N67", ks)
    for nm in ("vctk_k12", "vctk_k5"):
        rep["mp3_cov"][nm] = {"paired_diff_median": round(float(np.median(g[nm])), 3), "frac_better": round(float((g[nm] < 0).mean()), 3),
                              "DiD": round(float(np.median(g[nm]) - np.median(g["ff2_k0"])), 3), "DiD_ci95": ci(g[nm], rng, g["ff2_k0"])}
    rep["mp3_cov"]["control_ff2_mp3_k0"] = round(float(np.median(g["ff2_k0"])), 3)
    print("mp3_cov", rep["mp3_cov"], flush=True)
    s, kind, d = COND["ff2_k0"]
    kf = keys(s, kind)
    gc = v(s, kind, d, "N71", kf) - v(s, kind, d, "N67", kf)
    for nm in ("nmk_k12", "nmk_k5", "uiro_k12", "uiro_k5"):
        s, kind, d = COND[nm]
        ks = keys(s, kind)
        lv = {arm: v(s, kind, d, arm, ks) for arm in ARMS}
        gn = lv["N71"] - lv["N67"]
        rep["nmk"][nm] = {"n": len(ks)} | {f"{arm}_median": round(float(np.median(lv[arm])), 3) for arm in ARMS} | {
            "N71_N67_paired_median": round(float(np.median(gn)), 3), "DiD": round(float(np.median(gn) - np.median(gc)), 3), "DiD_ci95": ci(gn, rng, gc),
            "N67_Y58_paired_median": round(float(np.median(lv["N67"] - lv["Y58"])), 3)}
        print(nm, rep["nmk"][nm], flush=True)
    rep["nmk"]["control_ff2_k0"] = round(float(np.median(gc)), 3)
    y = [rep["codec"][n]["Y58"]["cost_paired_median"] for n in ("vctk_k12", "vctk_k5")]
    ra = "codec_large" if max(y) >= 0.5 else "codec_small" if max(y) < 0.2 else "codec_mid"
    c_ = rep["mp3_cov"]["control_ff2_mp3_k0"]
    b12 = rep["mp3_cov"]["vctk_k12"]["DiD"]
    rb = "invalid_control" if c_ > 0.5 else "codec_carrier" if b12 >= -0.3 else "codec_neutral" if b12 <= -0.6 else "partial"
    n12 = rep["nmk"]["nmk_k12"]["DiD"]
    rc = "ja_transfer" if n12 <= -0.6 else "ja_none" if n12 >= -0.3 else "partial"
    rep["uiro_vs_nmk_Y58"] = [round(rep["nmk"][f"uiro_k{k}"]["Y58_median"] - rep["nmk"][f"nmk_k{k}"]["Y58_median"], 3) for k in (12, 5)]
    rep["verdict_rule"] = {"a": ra, "b": rb, "c": rc}
    rep["ear_candidate"] = False
    print(json.dumps({"verdict_rule": rep["verdict_rule"], "Y58_codec_cost": y, "mp3_DiD_k12": b12, "nmk_DiD_k12": n12, "uiro_minus_nmk_Y58": rep["uiro_vs_nmk_Y58"]}, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
