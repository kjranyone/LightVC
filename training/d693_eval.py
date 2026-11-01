"""D693 の評価(規則は results/conv_p0/d693/prereg.yaml): 声の低い実女声の入力を高い声の目標(LX)と別の声の低い女声(LL)へ k0 で変換した官能指数 v1(gen fold)の
同じ入力どうしの差 LX − LL(表の声区の開きの罰)・B1R の表での回復 LXB1R − LX・開きとの Spearman・妥当性(LL・自己変換・入力 SRC)・CER(記録だけ)。
    uv run python d693_eval.py --out ../results/conv_p0/d693/d693.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from d692_eval import f0_med
from kansei_v1 import KanseiV1

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")


def wav(s: str, k: str, d: str) -> Path:
    return H / s / "gate" / d / f"{k}_{d}.wav"


def boot(d: np.ndarray, g: np.ndarray, rng: np.random.Generator) -> list:
    us = sorted(set(g))
    idx = {u: np.flatnonzero(g == u) for u in us}
    bs = [np.median(d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])]) for _ in range(2000)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    km = KanseiV1("gen")
    rng = np.random.default_rng(693)
    lx = json.loads((H / "d693_lx" / "devset.json").read_text())["pairs"]
    ll = {(r["pi"], r["j"]): r for r in json.loads((H / "d693_ll" / "devset.json").read_text())["pairs"]}
    kx = [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in lx]
    kl = [f"p{r['pi']:04d}_t{ll[(r['pi'], r['j'])]['ti']:03d}" for r in lx]
    v = {"LX": np.array(km.margin([wav("d693_lx", k, "RUST0Y58") for k in kx])),
         "LXB1R": np.array(km.margin([wav("d693_lx", k, "RUST0B1R") for k in kx])),
         "LL": np.array(km.margin([wav("d693_ll", k, "RUST0Y58") for k in kl]))}
    self_ = json.loads((H / "d688_lowself2" / "devset.json").read_text())["pairs"]
    v["SELF"] = np.array(km.margin([wav("d688_lowself2", f"p{r['pi']:04d}_t{r['ti']:03d}", "RUST0Y58") for r in self_]))
    v["SRC"] = np.array(km.margin([H / "d688_lowself2" / "d56" / f"p{r['pi']:04d}_t{r['ti']:03d}_SRC.wav" for r in self_]))
    g = np.array([r["pi"] for r in lx])
    en = lambda s, ti: float(np.exp(json.loads((H / s / "rr_en" / f"en_{ti:03d}" / "enroll.json").read_text())["mu"]))
    f0t_x = np.array([en("d693_lx", r["ti"]) for r in lx])
    f0t_l = np.array([en("d693_ll", ll[(r["pi"], r["j"])]["ti"]) for r in lx])
    f0o = {"LX": np.array([f0_med(wav("d693_lx", k, "RUST0Y58")) for k in kx]), "LL": np.array([f0_med(wav("d693_ll", k, "RUST0Y58")) for k in kl]),
           "LXB1R": np.array([f0_med(wav("d693_lx", k, "RUST0B1R")) for k in kx])}
    rep: dict = {"fold": "gen", "n_pairs": len(lx), "n_inputs": len(set(g))}
    for nm, x in v.items():
        rep[nm] = {"median": round(float(np.median(x)), 3), "in_band": round(float((x <= 0).mean()), 3), "n": len(x)}
    for nm, x in f0o.items():
        rep[nm]["f0_out_median"] = round(float(np.nanmedian(x)), 1)
    rep["f0_T_median"] = {"LX": round(float(np.median(f0t_x)), 1), "LL": round(float(np.median(f0t_l)), 1)}
    dB = v["LX"] - v["LL"]
    dR = v["LXB1R"] - v["LX"]
    rep["delta_B"] = {"paired_median": round(float(np.median(dB)), 3), "ci95_by_input": boot(dB, g, rng), "frac_LX_worse": round(float((dB > 0).mean()), 3)}
    rep["b1r_recovery"] = {"paired_median": round(float(np.median(dR)), 3), "ci95_by_input": boot(dR, g, rng), "frac_better": round(float((dR < 0).mean()), 3),
                           "fraction_of_delta_B": round(float(-np.median(dR) / np.median(dB)), 3) if np.median(dB) > 0 else None}
    gap = np.log2(f0t_x / f0o["LX"]) - np.log2(f0t_l / f0o["LL"])
    ok = np.isfinite(gap)
    rho, p = spearmanr(gap[ok], dB[ok])
    rep["dose"] = {"spearman_gapdiff_deltaB": round(float(rho), 3), "p": round(float(p), 4), "n": int(ok.sum()),
                   "gap_LX_semitones_median": round(float(np.nanmedian(12 * np.log2(f0t_x / f0o["LX"]))), 2),
                   "gap_LL_semitones_median": round(float(np.nanmedian(12 * np.log2(f0t_l / f0o["LL"]))), 2)}
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        for r, k, k2 in zip(lx, kx, kl):
            (t / f"{k}_SRC.wav").symlink_to(H / "d693_lx" / "d56" / f"{k}_SRC.wav")
            (t / f"{k}_LX.wav").symlink_to(wav("d693_lx", k, "RUST0Y58"))
            (t / f"{k}_LXB1R.wav").symlink_to(wav("d693_lx", k, "RUST0B1R"))
            (t / f"{k}_LL.wav").symlink_to(wav("d693_ll", k2, "RUST0Y58"))
        oj = t / "cer.json"
        r_ = subprocess.run([sys.executable, str(Path(__file__).parent / "d23_cer.py"), "--dirs", str(t), "--names", "lowx", "--out", str(oj)], capture_output=True, text=True)
        assert r_.returncode == 0, r_.stderr[-1500:]
        c = json.loads(oj.read_text())["lowx"]["per"]
    rep["cer"] = {nm: round(float(np.median(list(c.get(nm, {}).values()))), 4) for nm in ("LX", "LXB1R", "LL") if c.get(nm)}
    ll_ok = rep["LL"]["median"] <= 0.5
    m = rep["delta_B"]["paired_median"]
    rep["verdict_rule"] = "invalid_base" if not ll_ok else "B_large" if m >= 2.0 else "B_small" if m <= 1.0 else "B_mid"
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
