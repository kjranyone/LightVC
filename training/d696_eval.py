"""D696 / D697 の評価(paper_mf E-A1t2・規則は results/conv_p0/d696/prereg.yaml・results/diag_ppgvoc66/prereg.yaml): 腕(NR = P58 から R だけを外す・N66 = 腕 66)と Y58 の
官能指数 v1(gen fold)の対の差・差の差 DiD = 男声の対の差 − 女声の対照(devff2 k0)の対の差(入力の再標本の 95% 区間)・外郎売と devff2 の CER の対の差。N65(D695)を並べる。
    uv run python d696_eval.py --arm NR --out ../results/conv_p0/d696/d696.json
    uv run python d696_eval.py --arm N66 --out ../results/diag_ppgvoc66/d697.json
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
ROOT = Path(__file__).resolve().parent.parent
COND = [("uiro_k12", "d487_uiro", None, "RUST12"), ("uiro_k5", "d487_uiro", None, "RUST5Q3"), ("vctkME_k12", "devvctkm", "ME", "RUST12"), ("vctkME_k5", "devvctkm", "ME", "RUST5Q3"),
        ("devff2_k0", "devff2", None, "RUST0")]


def pairs(s: str, kind: str | None) -> list:
    return [r for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def cer(arm: str) -> dict:
    out = {}
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        jobs = [("d487_uiro", "RUST12", "uiro_k12"), ("d487_uiro", "RUST5Q3", "uiro_k5"), ("devff2", "RUST0", "devff2_k0")]
        for s, d, nm in jobs:
            u = t / nm
            u.mkdir()
            for r in pairs(s, None):
                k = f"p{r['pi']:04d}_t{r['ti']:03d}"
                (u / f"{k}_SRC.wav").symlink_to(H / s / "d56" / f"{k}_SRC.wav")
                for a in ("Y58", arm):
                    (u / f"{k}_{a}.wav").symlink_to(H / s / "gate" / f"{d}{a}" / f"{k}_{d}{a}.wav")
        oj = t / "cer.json"
        r_ = subprocess.run([sys.executable, str(Path(__file__).parent / "d23_cer.py"), "--dirs", *[str(t / nm) for *_, nm in jobs], "--names", *[nm for *_, nm in jobs], "--out", str(oj)],
                            capture_output=True, text=True)
        assert r_.returncode == 0, r_.stderr[-1500:]
        c = json.loads(oj.read_text())
    for *_, nm in jobs:
        b, x = c[nm]["per"]["Y58"], c[nm]["per"][arm]
        kk = sorted(set(b) & set(x))
        out[nm] = {"n": len(kk), "Y58_median": round(float(np.median([b[k] for k in kk])), 4), "median": round(float(np.median([x[k] for k in kk])), 4),
                   "paired_diff_median": round(float(np.median([x[k] - b[k] for k in kk])), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("NR", "N66"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    km = KanseiV1("gen")
    rng = np.random.default_rng(696)
    rep: dict = {"fold": "gen", "base": "Y58", "arm": a.arm}
    dd = {}
    for nm, s, kd, d in COND:
        keys = [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in pairs(s, kd)]
        f = lambda sx: [H / s / "gate" / f"{d}{sx}" / f"{k}_{d}{sx}.wav" for k in keys]
        vb = np.array(km.margin(f("Y58")))
        va = np.array(km.margin(f(a.arm)))
        dd[nm] = va - vb
        rep[nm] = {"n": len(keys), "Y58_median": round(float(np.median(vb)), 3), "median": round(float(np.median(va)), 3), "paired_diff_median": round(float(np.median(dd[nm])), 3),
                   "frac_better": round(float((dd[nm] < 0).mean()), 3), "pass": round(float((va <= 0).mean()), 3)}
        print(nm, rep[nm], flush=True)
    ctl = dd["devff2_k0"]
    male = [n for n, *_ in COND[:4]]
    for nm in male:
        bs = [np.median(rng.choice(dd[nm], len(dd[nm]))) - np.median(rng.choice(ctl, len(ctl))) for _ in range(2000)]
        rep[nm]["DiD"] = round(float(np.median(dd[nm]) - np.median(ctl)), 3)
        rep[nm]["DiD_ci95"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    d695 = json.loads((ROOT / "results/diag_ppgvoc65/d695.json").read_text())
    rep["ref_N65"] = {nm: {"paired_diff_median": d695[nm]["paired_diff_median"], "DiD": d695[nm].get("DiD")} for nm, *_ in COND}
    rep["cer"] = cer(a.arm)
    print("cer", rep["cer"], flush=True)
    c_ = rep["devff2_k0"]["paired_diff_median"]
    u = [rep["uiro_k12"]["DiD"], rep["uiro_k5"]["DiD"]]
    if c_ > 0.5:
        v = "invalid_control"
    elif a.arm == "NR":
        v = "r_main" if all(x <= -1.0 for x in u) else "r_minor" if all(x >= -0.5 for x in u) else "r_partial"
    else:
        ok_cer = all(rep["cer"][n]["paired_diff_median"] <= 0.03 for n in ("uiro_k12", "uiro_k5"))
        v = ("content_free_gain" if ok_cer else "gain_with_cer_cost") if all(x <= -1.0 for x in u) else "onehot_needed" if all(x >= -0.5 for x in u) else "partial"
    rep["verdict_rule"] = v
    rep["ear_candidate"] = all(rep[n]["median"] <= 0 for n in male)
    print(json.dumps({"DiD": [rep[n]["DiD"] for n in male], "control": c_, "verdict_rule": v, "ear_candidate": rep["ear_candidate"]}, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
