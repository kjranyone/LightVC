"""D692 の評価(規則は results/conv_p0/d692/prereg.yaml): Y58(基準)と B1R / B1X(登録の表を低い声区の形へ写す・逆向き)の官能指数 v1(gen fold)の対の差・
組ごとの差と log2(f0_T / 出力の f0)の Spearman・外郎売の CER の対の差。
    uv run python d692_eval.py --out ../results/conv_p0/d692/d692.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.stats import spearmanr

from kansei_v1 import KanseiV1

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
COND = [("uiro_k12", "d487_uiro", None, "RUST12"), ("uiro_k5", "d487_uiro", None, "RUST5Q3"), ("vctkME_k12", "devvctkm", "ME", "RUST12"), ("vctkME_k5", "devvctkm", "ME", "RUST5Q3")]
ARMS = ("B1R", "B1X")


def pairs(s: str, kind: str | None) -> list:
    return [r for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def f0_med(p: Path) -> float:
    import pyworld
    x, sr = sf.read(p, dtype="float64")
    g = math.gcd(sr, 16000)
    x = resample_poly(x, 16000 // g, sr // g)
    f, t = pyworld.dio(x, 16000, f0_floor=60, f0_ceil=1000, frame_period=5.0)
    f = pyworld.stonemask(x, f, t, 16000)
    v = f[f > 0]
    return float(np.median(v)) if len(v) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    km = KanseiV1("gen")
    rep: dict = {"fold": "gen", "base": "Y58"}
    dose_x, dose_y = [], []
    for nm, s, kd, d in COND:
        rs = pairs(s, kd)
        keys = [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in rs]
        f = lambda sx: [H / s / "gate" / f"{d}{sx}" / f"{k}_{d}{sx}.wav" for k in keys]
        pb = f("Y58")
        pa = {arm: f(arm) for arm in ARMS}
        ok = [i for i in range(len(keys)) if pb[i].exists() and all(pa[arm][i].exists() for arm in ARMS)]
        vb = np.array(km.margin([pb[i] for i in ok]))
        rep[nm] = {"n": len(ok), "n_expected": len(keys), "Y58_median": round(float(np.median(vb)), 3)}
        f0t = [float(np.exp(json.loads((H / s / "rr_en" / f"en_{rs[i]['ti']:03d}" / "enroll.json").read_text())["mu"])) for i in ok]
        f0o = [f0_med(pb[i]) for i in ok]
        for arm in ARMS:
            va = np.array(km.margin([pa[arm][i] for i in ok]))
            dd = va - vb
            rep[nm][arm] = {"median": round(float(np.median(va)), 3), "paired_diff_median": round(float(np.median(dd)), 3), "frac_better": round(float((dd < 0).mean()), 3),
                            "pass": round(float((va <= 0).mean()), 3)}
            if arm == "B1R":
                for i in range(len(ok)):
                    if np.isfinite(f0o[i]):
                        dose_x.append(math.log2(f0t[i] / f0o[i]))
                        dose_y.append(float(dd[i]))
        rep[nm]["f0_out_median"] = round(float(np.nanmedian(f0o)), 1)
        rep[nm]["f0_T_median"] = round(float(np.median(f0t)), 1)
        print(nm, rep[nm], flush=True)
    rho, p = spearmanr(dose_x, dose_y)
    rep["dose_B1R"] = {"n": len(dose_x), "spearman": round(float(rho), 3), "p": round(float(p), 4)}
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        for r in pairs("d487_uiro", None):
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            (t / f"{k}_SRC.wav").symlink_to(H / "d487_uiro" / "d56" / f"{k}_SRC.wav")
            for d, tag in (("RUST12", "k12"), ("RUST5Q3", "k5")):
                for arm in ("Y58",) + ARMS:
                    fp = H / "d487_uiro" / "gate" / f"{d}{arm}" / f"{k}_{d}{arm}.wav"
                    if fp.exists():
                        (t / f"{k}_{arm}{tag}.wav").symlink_to(fp)
        oj = t / "cer.json"
        r_ = subprocess.run([sys.executable, str(Path(__file__).parent / "d23_cer.py"), "--dirs", str(t), "--names", "uiro", "--out", str(oj)], capture_output=True, text=True)
        assert r_.returncode == 0, r_.stderr[-1500:]
        c = json.loads(oj.read_text())["uiro"]["per"]
    for tag in ("k12", "k5"):
        b = c.get(f"Y58{tag}", {})
        for arm in ARMS:
            x = c.get(f"{arm}{tag}", {})
            kk = sorted(set(b) & set(x))
            rep[f"cer_uiro_{tag}_{arm}"] = {"n": len(kk), "paired_diff_median": round(float(np.median([x[k] - b[k] for k in kk])), 4),
                                            "Y58_median": round(float(np.median([b[k] for k in kk])), 4), "median": round(float(np.median([x[k] for k in kk])), 4)}
    main4 = [n for n, *_ in COND]
    d = lambda arm, n: rep[n][arm]["paired_diff_median"]
    cer_bad = {n: rep[f"cer_uiro_{n.split('_')[1]}_B1R"]["paired_diff_median"] > 0.03 for n in main4 if n.startswith("uiro")}
    sup = all(d("B1R", n) <= -0.5 and not cer_bad.get(n, False) for n in main4) and sum(d("B1X", n) >= -0.1 for n in main4) >= 3
    nons = sum(d("B1X", n) <= -0.5 for n in main4) >= 2
    notr = all(abs(d("B1R", n)) < 0.3 and abs(d("B1X", n)) < 0.3 for n in main4)
    rep["verdict_rule"] = "supported" if sup else "nonspecific" if nons else "not_read" if notr else "partial"
    print(json.dumps({"dose_B1R": rep["dose_B1R"], "verdict_rule": rep["verdict_rule"]}, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
