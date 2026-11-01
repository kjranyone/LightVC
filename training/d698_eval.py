"""D698 の評価(paper_mf E-A1b0・規則は results/conv_p0/d698/prereg.yaml): 生成器 N65 の官能指数 v1(gen fold)で
(c) 同じ出力の高さでの男声との差 gap = 外郎売 N65 の中央値 − devff2 LOW(k −8 / k −15)の中央値(各群を再標本した 95% 区間)・低い高さの罰(devff2 の k0 との対の差・記録)、
(a) F1 → 教師の f0・有声の差し替え T65 の対の差 T65 − N65 と DiD_TF = 男声の対の差 − 女声(devff2 k0)の対の差、を出す。CER は記録。
    uv run python d698_eval.py --out ../results/conv_p0/d698/d698.json
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
LOW = [("devff2_km8", "RUST-8", "uiro_k12"), ("devff2_km15", "RUST-15Q3", "uiro_k5")]


def pairs(s: str, kind: str | None) -> list:
    return [r for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def keys(s: str, kind: str | None) -> list:
    return [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in pairs(s, kind)]


def wavs(s: str, d: str, sx: str, ks: list) -> list:
    return [H / s / "gate" / f"{d}{sx}" / f"{k}_{d}{sx}.wav" for k in ks]


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
                for a in ("N65", "T65"):
                    (u / f"{k}_{a}.wav").symlink_to(H / s / "gate" / f"{d}{a}" / f"{k}_{d}{a}.wav")
        oj = t / "cer.json"
        r_ = subprocess.run([sys.executable, str(Path(__file__).parent / "d23_cer.py"), "--dirs", *[str(t / nm) for *_, nm in jobs], "--names", *[nm for *_, nm in jobs], "--out", str(oj)],
                            capture_output=True, text=True)
        assert r_.returncode == 0, r_.stderr[-1500:]
        c = json.loads(oj.read_text())
    for *_, nm in jobs:
        b, x = c[nm]["per"]["N65"], c[nm]["per"]["T65"]
        kk = sorted(set(b) & set(x))
        out[nm] = {"n": len(kk), "N65_median": round(float(np.median([b[k] for k in kk])), 4), "T65_median": round(float(np.median([x[k] for k in kk])), 4),
                   "paired_diff_median": round(float(np.median([x[k] - b[k] for k in kk])), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    km = KanseiV1("gen")
    rng = np.random.default_rng(698)
    rep: dict = {"fold": "gen", "generator": "N65"}
    n65, dd = {}, {}
    for nm, s, kd, d in COND:
        ks = keys(s, kd)
        vb = np.array(km.margin(wavs(s, d, "N65", ks)))
        va = np.array(km.margin(wavs(s, d, "T65", ks)))
        n65[nm], dd[nm] = vb, va - vb
        rep[nm] = {"n": len(ks), "N65_median": round(float(np.median(vb)), 3), "T65_median": round(float(np.median(va)), 3), "paired_diff_median": round(float(np.median(dd[nm])), 3),
                   "frac_better": round(float((dd[nm] < 0).mean()), 3), "T65_pass": round(float((va <= 0).mean()), 3)}
        print(nm, rep[nm], flush=True)
    ctl = dd["devff2_k0"]
    for nm in [n for n, *_ in COND[:4]]:
        bs = [np.median(rng.choice(dd[nm], len(dd[nm]))) - np.median(rng.choice(ctl, len(ctl))) for _ in range(2000)]
        rep[nm]["DiD_TF"] = round(float(np.median(dd[nm]) - np.median(ctl)), 3)
        rep[nm]["DiD_TF_ci95"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    ks = keys("devff2", None)
    for nm, d, male in LOW:
        v = np.array(km.margin(wavs("devff2", d, "N65", ks)))
        pen = v - n65["devff2_k0"]
        bs = [np.median(rng.choice(n65[male], len(n65[male]))) - np.median(rng.choice(v, len(v))) for _ in range(2000)]
        rep[nm] = {"n": len(ks), "median": round(float(np.median(v)), 3), "pass": round(float((v <= 0).mean()), 3), "low_penalty_paired_median": round(float(np.median(pen)), 3),
                   "male": male, "male_N65_median": rep[male]["N65_median"], "gap": round(float(np.median(n65[male]) - np.median(v)), 3),
                   "gap_ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
        print(nm, rep[nm], flush=True)
    rep["cer"] = cer()
    print("cer", rep["cer"], flush=True)
    g = [rep[nm]["gap"] for nm, *_ in LOW]
    c_vt = "low_main" if all(x <= 1.0 for x in g) else "male_main" if all(x >= 2.0 for x in g) else "mixed"
    c_ = rep["devff2_k0"]["paired_diff_median"]
    u = [rep["uiro_k12"]["DiD_TF"], rep["uiro_k5"]["DiD_TF"]]
    a_vt = "invalid_control" if c_ > 0.5 else "f1_main" if all(x <= -1.0 for x in u) else "f1_minor" if all(x >= -0.5 for x in u) else "f1_partial"
    rep["verdict_rule"] = {"c": c_vt, "a": a_vt}
    rep["ear_candidate"] = all(rep[n]["T65_median"] <= 0 for n in ("uiro_k12", "uiro_k5", "vctkME_k12", "vctkME_k5"))
    print(json.dumps({"gap": g, "DiD_TF": u, "control": c_, "verdict_rule": rep["verdict_rule"], "ear_candidate": rep["ear_candidate"]}, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
