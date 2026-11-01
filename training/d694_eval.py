"""D694 の評価(規則は results/conv_p0/d694/prereg.yaml): Y58(基準)と CQNR(C2 定数 + 事後 one-hot + R なし)・NEUT(CQNR + c0 を女声の単位ごとの平均)の
官能指数 v1(gen fold)の対の差。男声の入力 4 条件(外郎売・VCTK の男声 × k12 / k5)と女声の対照(devff2 k0)。外郎売の CER は記録だけ。
    uv run python d694_eval.py --out ../results/conv_p0/d694/d694.json
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
ARMS = ("CQNR", "NEUT")


def pairs(s: str, kind: str | None) -> list:
    return [r for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    km = KanseiV1("gen")
    rep: dict = {"fold": "gen", "base": "Y58"}
    for nm, s, kd, d in COND:
        keys = [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in pairs(s, kd)]
        f = lambda sx: [H / s / "gate" / f"{d}{sx}" / f"{k}_{d}{sx}.wav" for k in keys]
        vb = np.array(km.margin(f("Y58")))
        rep[nm] = {"n": len(keys), "Y58_median": round(float(np.median(vb)), 3)}
        va = {}
        for arm in ARMS:
            va[arm] = np.array(km.margin(f(arm)))
            dd = va[arm] - vb
            rep[nm][arm] = {"median": round(float(np.median(va[arm])), 3), "paired_diff_median": round(float(np.median(dd)), 3), "frac_better": round(float((dd < 0).mean()), 3),
                            "pass": round(float((va[arm] <= 0).mean()), 3)}
        rep[nm]["c0_part_NEUT_minus_CQNR"] = round(float(np.median(va["NEUT"] - va["CQNR"])), 3)
        print(nm, rep[nm], flush=True)
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        for r in pairs("d487_uiro", None):
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            (t / f"{k}_SRC.wav").symlink_to(H / "d487_uiro" / "d56" / f"{k}_SRC.wav")
            for d, tag in (("RUST12", "k12"), ("RUST5Q3", "k5")):
                for arm in ("Y58",) + ARMS:
                    (t / f"{k}_{arm}{tag}.wav").symlink_to(H / "d487_uiro" / "gate" / f"{d}{arm}" / f"{k}_{d}{arm}.wav")
        oj = t / "cer.json"
        r_ = subprocess.run([sys.executable, str(Path(__file__).parent / "d23_cer.py"), "--dirs", str(t), "--names", "uiro", "--out", str(oj)], capture_output=True, text=True)
        assert r_.returncode == 0, r_.stderr[-1500:]
        c = json.loads(oj.read_text())["uiro"]["per"]
    for tag in ("k12", "k5"):
        b = c.get(f"Y58{tag}", {})
        for arm in ARMS:
            x = c.get(f"{arm}{tag}", {})
            kk = sorted(set(b) & set(x))
            rep[f"cer_uiro_{tag}_{arm}"] = {"n": len(kk), "paired_diff_median": round(float(np.median([x[k] - b[k] for k in kk])), 4), "median": round(float(np.median([x[k] for k in kk])), 4)}
    male = [n for n, *_ in COND[:4]]
    d = lambda n: rep[n]["NEUT"]["paired_diff_median"]
    ctl = rep["devff2_k0"]["NEUT"]["paired_diff_median"]
    rep["verdict_rule"] = ("invalid_control" if ctl > 1.0 else "interface" if sum(d(n) <= -2.0 for n in male) >= 3
                           else "not_interface" if sum(d(n) >= -1.0 for n in male) >= 3 else "partial")
    print(json.dumps({"control": ctl, "verdict_rule": rep["verdict_rule"]}, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
