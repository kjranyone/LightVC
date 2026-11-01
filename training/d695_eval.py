"""D695 の評価(腕 65 = diag_ppgvoc65・paper_mf E-A1t・規則は results/diag_ppgvoc65/prereg.yaml): 学習の中で界面を中立にした生成器 N65(事後 one-hot・C2 定数・c0 = 単位の表・R なし)と
Y58 の官能指数 v1(gen fold)の対の差。男声 4 条件(外郎売・VCTK の男声 × k12 / k5)と女声の対照(devff2 k0)。主の量 = 差の差 DiD = 男声の対の差 − 女声の対の差(入力で束ねた区間つき)。
外郎売の CER は記録だけ。D694 の推論だけの中立化 NEUT の値を並べる(参考)。
    uv run python d695_eval.py --out ../results/diag_ppgvoc65/d695.json
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
ARM = "N65"


def pairs(s: str, kind: str | None) -> list:
    return [r for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    km = KanseiV1("gen")
    rng = np.random.default_rng(695)
    rep: dict = {"fold": "gen", "base": "Y58", "arm": ARM}
    dd = {}
    for nm, s, kd, d in COND:
        keys = [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in pairs(s, kd)]
        f = lambda sx: [H / s / "gate" / f"{d}{sx}" / f"{k}_{d}{sx}.wav" for k in keys]
        vb = np.array(km.margin(f("Y58")))
        va = np.array(km.margin(f(ARM)))
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
    d694 = json.loads((ROOT / "results/conv_p0/d694/d694.json").read_text())
    rep["ref_D694_NEUT_inference_only"] = {nm: d694[nm]["NEUT"]["paired_diff_median"] for nm, *_ in COND}
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        for r in pairs("d487_uiro", None):
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            (t / f"{k}_SRC.wav").symlink_to(H / "d487_uiro" / "d56" / f"{k}_SRC.wav")
            for d, tag in (("RUST12", "k12"), ("RUST5Q3", "k5")):
                for arm in ("Y58", ARM):
                    (t / f"{k}_{arm}{tag}.wav").symlink_to(H / "d487_uiro" / "gate" / f"{d}{arm}" / f"{k}_{d}{arm}.wav")
        oj = t / "cer.json"
        r_ = subprocess.run([sys.executable, str(Path(__file__).parent / "d23_cer.py"), "--dirs", str(t), "--names", "uiro", "--out", str(oj)], capture_output=True, text=True)
        assert r_.returncode == 0, r_.stderr[-1500:]
        c = json.loads(oj.read_text())["uiro"]["per"]
    for tag in ("k12", "k5"):
        b, x = c.get(f"Y58{tag}", {}), c.get(f"{ARM}{tag}", {})
        kk = sorted(set(b) & set(x))
        rep[f"cer_uiro_{tag}"] = {"n": len(kk), "Y58_median": round(float(np.median([b[k] for k in kk])), 4), "median": round(float(np.median([x[k] for k in kk])), 4),
                                  "paired_diff_median": round(float(np.median([x[k] - b[k] for k in kk])), 4)}
    did = [rep[n]["DiD"] for n in male]
    rep["verdict_rule"] = "interface" if sum(v <= -2.0 for v in did) >= 3 else "style" if sum(v >= -1.0 for v in did) >= 3 else "partial"
    rep["quality_axis"] = "neutral_ok" if rep["devff2_k0"]["paired_diff_median"] <= 0.5 else "neutral_costly"
    rep["ear_candidate"] = all(rep[n]["median"] <= 0 for n in male)
    print(json.dumps({"DiD": did, "control": rep["devff2_k0"]["paired_diff_median"], "verdict_rule": rep["verdict_rule"], "quality_axis": rep["quality_axis"],
                      "ear_candidate": rep["ear_candidate"]}, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
