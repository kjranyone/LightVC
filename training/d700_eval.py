"""D700 / D701 の評価(腕 67 = paper_mf E-A1b・腕 68 = E-C1・規則は results/diag_ppgvoc67/prereg.yaml・results/diag_ppgvoc68/prereg.yaml): 腕(N67 / N68)と基準(N65 / N67)の官能指数 v1(gen fold)の対の差・
DiD = 男声の対の差 − 女声の対照(devff2 k0)の対の差(入力の再標本の 95% 区間)・外郎売と devff2 の CER の対の差(記録)。Y58 との差も並べる。
    uv run python d700_eval.py --out ../results/diag_ppgvoc67/d700.json
    uv run python d700_eval.py --base N67 --arm N68 --out ../results/diag_ppgvoc68/d701.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from kansei_v1 import KanseiV1

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
COND = [("uiro_k12", "d487_uiro", None, "RUST12"), ("uiro_k5", "d487_uiro", None, "RUST5Q3"), ("vctkME_k12", "devvctkm", "ME", "RUST12"), ("vctkME_k5", "devvctkm", "ME", "RUST5Q3"),
        ("devff2_k0", "devff2", None, "RUST0")]
BASE, ARM = "N65", "N67"
RULES = {"N67": ("f0v_main", "timing"), "N68": ("coverage_main", "no_transfer"), "S70": ("ms_ok", "ms_none"), "N71": ("coverage_main", "no_transfer")}


def keys(s: str, kind: str | None) -> list:
    return [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def cer() -> dict:
    out = {}
    jobs = [("d487_uiro", "RUST12", "uiro_k12"), ("d487_uiro", "RUST5Q3", "uiro_k5"), ("devff2", "RUST0", "devff2_k0")]
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        for s, d, nm in jobs:
            u = t / nm
            u.mkdir()
            for k in keys(s, None):
                (u / f"{k}_SRC.wav").symlink_to(H / s / "d56" / f"{k}_SRC.wav")
                for a in (BASE, ARM):
                    (u / f"{k}_{a}.wav").symlink_to(H / s / "gate" / f"{d}{a}" / f"{k}_{d}{a}.wav")
        oj = t / "cer.json"
        r_ = subprocess.run([sys.executable, str(Path(__file__).parent / "d23_cer.py"), "--dirs", *[str(t / nm) for *_, nm in jobs], "--names", *[nm for *_, nm in jobs], "--out", str(oj)],
                            capture_output=True, text=True)
        assert r_.returncode == 0, r_.stderr[-1500:]
        c = json.loads(oj.read_text())
    for *_, nm in jobs:
        b, x = c[nm]["per"][BASE], c[nm]["per"][ARM]
        kk = sorted(set(b) & set(x))
        out[nm] = {"n": len(kk), f"{BASE}_median": round(float(np.median([b[k] for k in kk])), 4), f"{ARM}_median": round(float(np.median([x[k] for k in kk])), 4),
                   "paired_diff_median": round(float(np.median([x[k] - b[k] for k in kk])), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="N65")
    ap.add_argument("--arm", default="N67", choices=tuple(RULES))
    ap.add_argument("--guard", default="", help="spk_guard.py の出力(N71: coverage_main は守りが通るときだけ)")
    a = ap.parse_args()
    global BASE, ARM
    BASE, ARM = a.base, a.arm
    km = KanseiV1("gen")
    rng = np.random.default_rng(700)
    rep: dict = {"fold": "gen", "base": BASE, "arm": ARM}
    dd = {}
    for nm, s, kd, d in COND:
        ks = keys(s, kd)
        f = lambda sx: [H / s / "gate" / f"{d}{sx}" / f"{k}_{d}{sx}.wav" for k in ks]
        vb, va, vy = (np.array(km.margin(f(x))) for x in (BASE, ARM, "Y58"))
        dd[nm] = va - vb
        rep[nm] = {"n": len(ks), "Y58_median": round(float(np.median(vy)), 3), f"{BASE}_median": round(float(np.median(vb)), 3), f"{ARM}_median": round(float(np.median(va)), 3),
                   "paired_diff_median": round(float(np.median(dd[nm])), 3), "frac_better": round(float((dd[nm] < 0).mean()), 3), "pass": round(float((va <= 0).mean()), 3),
                   "vs_Y58_paired_median": round(float(np.median(va - vy)), 3)}
        print(nm, rep[nm], flush=True)
    ctl = dd["devff2_k0"]
    male = [n for n, *_ in COND[:4]]
    for nm in male:
        bs = [np.median(rng.choice(dd[nm], len(dd[nm]))) - np.median(rng.choice(ctl, len(ctl))) for _ in range(2000)]
        rep[nm]["DiD"] = round(float(np.median(dd[nm]) - np.median(ctl)), 3)
        rep[nm]["DiD_ci95"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    rep["cer"] = cer()
    print("cer", rep["cer"], flush=True)
    c_ = rep["devff2_k0"]["paired_diff_median"]
    u = [rep["uiro_k12"]["DiD"], rep["uiro_k5"]["DiD"]]
    ok_, no_ = RULES[ARM]
    if ARM == "S70":
        allc = [rep[n]["paired_diff_median"] for n, *_ in COND]
        cer_ok = all(rep["cer"][n]["paired_diff_median"] <= 0.03 for n in ("uiro_k12", "uiro_k5"))
        v = "ms_ok" if all(x <= -0.3 for x in allc) and cer_ok else "ms_cer_cost" if all(x <= -0.3 for x in allc) else "ms_none" if float(np.median(allc)) >= -0.1 else "partial"
    elif c_ > 0.5:
        v = "invalid_control"
    elif all(x <= -1.0 for x in u):
        v = ok_
    elif ARM in ("N68", "N71") and u[0] <= -1.0 and u[1] > -0.5:
        v = "k12_only"
    elif all(x >= -0.5 for x in u):
        v = no_
    else:
        v = "partial"
    if a.guard:
        g_ = json.loads(Path(a.guard).read_text())
        rep["guard_ok"] = g_["guard_ok"]
        if v == "coverage_main" and not g_["guard_ok"]:
            v = "coverage_male_drift"
    if ARM == "N68":
        p64 = {"uiro_k12": 0.117, "uiro_k5": 0.619, "vctkME_k12": -0.370, "vctkME_k5": -0.147, "devff2_k0": -0.010}
        rep["interaction_vs_arm64"] = {n: round(rep[n]["paired_diff_median"] - p64[n], 3) for n in p64}
    rep["verdict_rule"] = v
    rep["ear_candidate"] = all(rep[n][f"{ARM}_median"] <= 0 for n in male)
    print(json.dumps({"DiD": [rep[n]["DiD"] for n in male], "control": c_, "verdict_rule": v, "ear_candidate": rep["ear_candidate"]}, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
