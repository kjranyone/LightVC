"""D691(腕 64 の評価・規則は results/diag_ppgvoc64/prereg.yaml の measurement / rule): P58 と P64 の Rust の描画(関門の組・gate/RUST<k>[Y64])の官能指数 v1(gen を外した fold)と、
外郎売の CER(d23_cer・入力 SRC の書き起こしが基準)を対で比べる。
    uv run python d691_eval64.py --sfx_a Y58 --sfx_b Y64 --out ../results/diag_ppgvoc64/d691.json
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
COND = [("uiro_k12", "d487_uiro", None, "RUST12"), ("uiro_k5", "d487_uiro", None, "RUST5Q3"), ("vctkME_k12", "devvctkm", "ME", "RUST12"),
        ("vctkME_k5", "devvctkm", "ME", "RUST5Q3"), ("vctkFE_k0", "devvctkm", "FE", "RUST0"), ("devff2_k0", "devff2", None, "RUST0"), ("lowself_k0", "d688_lowself2", None, "RUST0")]
MAIN = ("uiro_k12", "uiro_k5", "vctkME_k12", "vctkME_k5")


def keys(s: str, kind: str | None) -> list:
    return [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfx_a", default="Y58", help="P58 の描画の接尾辞(P64 と同じ手順で描き直したもの)")
    ap.add_argument("--sfx_b", default="Y64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_cer", action="store_true")
    a = ap.parse_args()
    km = KanseiV1("gen")
    rep: dict = {"fold": "gen", "sfx_a": a.sfx_a, "sfx_b": a.sfx_b}
    med = {}
    for nm, s, kd, d in COND:
        ks = keys(s, kd)
        pa = [H / s / "gate" / f"{d}{a.sfx_a}" / f"{k}_{d}{a.sfx_a}.wav" for k in ks]
        pb = [H / s / "gate" / f"{d}{a.sfx_b}" / f"{k}_{d}{a.sfx_b}.wav" for k in ks]
        ok = [i for i in range(len(ks)) if pa[i].exists() and pb[i].exists()]
        va = np.array(km.margin([pa[i] for i in ok]))
        vb = np.array(km.margin([pb[i] for i in ok]))
        dd = vb - va
        med[nm] = (float(np.median(va)), float(np.median(vb)))
        rep[nm] = {"n": len(ok), "n_expected": len(ks), "P58_median": round(med[nm][0], 3), "P64_median": round(med[nm][1], 3), "paired_diff_median": round(float(np.median(dd)), 3),
                   "frac_better": round(float((dd < 0).mean()), 3), "P58_pass": round(float((va <= 0).mean()), 3), "P64_pass": round(float((vb <= 0).mean()), 3)}
        print(nm, rep[nm], flush=True)
    rep["maleness"] = {"P58": round(med["vctkME_k12"][0] - med["vctkFE_k0"][0], 3), "P64": round(med["vctkME_k12"][1] - med["vctkFE_k0"][1], 3)}
    if not a.no_cer:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            for k in keys("d487_uiro", None):
                (t / f"{k}_SRC.wav").symlink_to(H / "d487_uiro" / "d56" / f"{k}_SRC.wav")
                for d, tag in (("RUST12", "k12"), ("RUST5Q3", "k5")):
                    for arm, sx in (("P58", a.sfx_a), ("P64", a.sfx_b)):
                        f = H / "d487_uiro" / "gate" / f"{d}{sx}" / f"{k}_{d}{sx}.wav"
                        if f.exists():
                            (t / f"{k}_{arm}{tag}.wav").symlink_to(f)
            oj = t / "cer.json"
            r = subprocess.run([sys.executable, str(Path(__file__).parent / "d23_cer.py"), "--dirs", str(t), "--names", "uiro", "--out", str(oj)], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-1500:]
            c = json.loads(oj.read_text())["uiro"]
        for tag in ("k12", "k5"):
            pa, pb = c["per"].get(f"P58{tag}", {}), c["per"].get(f"P64{tag}", {})
            kk = sorted(set(pa) & set(pb))
            rep[f"cer_uiro_{tag}"] = {"n": len(kk), "P58_mean": round(float(np.mean([pa[k] for k in kk])), 4), "P64_mean": round(float(np.mean([pb[k] for k in kk])), 4),
                                      "diff_mean": round(float(np.mean([pb[k] - pa[k] for k in kk])), 4)}
    m = [rep[n]["paired_diff_median"] for n in MAIN]
    worse = rep["devff2_k0"]["paired_diff_median"] > 0.3 or rep["maleness"]["P64"] > rep["maleness"]["P58"] + 0.3 or any(rep.get(f"cer_uiro_{t}", {}).get("diff_mean", 0.0) > 0.03 for t in ("k12", "k5"))
    if all(rep[n]["P64_median"] <= 0 for n in MAIN):
        v = "ear"
    elif worse:
        v = "worse"
    elif all(x <= -1.0 for x in m):
        v = "progress"
    else:
        v = "flat"
    rep["verdict_rule"] = v
    print(json.dumps({k: rep[k] for k in ("maleness", "verdict_rule")}, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
