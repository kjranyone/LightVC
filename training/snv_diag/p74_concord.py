"""段 P74 step a(results/snv_p74/prereg.yaml・学習なし): p74_conv.py の行から、系の対ごとの差(源で対・話者で束ねた 95 % の区間)と、devA(cv)で開く対 Σ が脆い源の組で同じ向きに開くかを出す。
dose=<p73_swap.py の json> を渡すと、組ごとの nz の折れ線で x の CER を乱しの用量に直し、amp の組の用量から cv の CER を予言する(副の読み)。
    cd training && uv run --no-sync python snv_diag/p74_concord.py in=results/snv_p74/conv_a.json,results/snv_p74/conv_b.json [amp=x:d_rv6] [dose=results/snv_p74/dose.json] [out=results/snv_p74/concord.json]
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def boot(d: np.ndarray, g: np.ndarray, n: int = 2000) -> tuple[float, float, float]:
    ks = sorted(set(g))
    by = [d[g == k] for k in ks]
    rng = np.random.default_rng(0)
    ms = [float(np.concatenate([by[i] for i in rng.integers(0, len(ks), len(ks))]).mean()) for _ in range(n)]
    return round(float(d.mean()), 4), round(float(np.percentile(ms, 2.5)), 4), round(float(np.percentile(ms, 97.5)), 4)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    rows = [r for p in opt["in"].split(",") for r in json.loads((ROOT / p).read_text())["rows"]]
    amp = opt.get("amp", "x:d_rv6")
    syss, sets = list(dict.fromkeys(r["sys"] for r in rows)), list(dict.fromkeys(r["set"] for r in rows))
    X = {(r["sys"], r["set"], r["src"]): (float(np.mean(r["x"])), r["self"], r["spk"]) for r in rows}
    mean = {nm: {s: {"x": round(float(np.mean([v[0] for k, v in X.items() if k[0] == s and k[1] == nm])), 4), "self": round(float(np.mean([v[1] for k, v in X.items() if k[0] == s and k[1] == nm])), 4),
                     "n": sum(1 for k in X if k[0] == s and k[1] == nm)} for s in syss} for nm in sets}
    val6 = {nm: {s: round(float(np.mean([v[0] for k, v in X.items() if k[0] == s and k[1] == nm and v[2] >= "em594"])), 4) for s in syss} for nm in sets if nm.startswith("x:d_")}
    pairs: dict = {nm: {} for nm in sets}
    for nm in sets:
        for a, b in itertools.combinations(syss, 2):
            src = sorted(k[2] for k in X if k[0] == a and k[1] == nm and (b, nm, k[2]) in X)
            d = np.array([X[(a, nm, s)][0] - X[(b, nm, s)][0] for s in src])
            pairs[nm][f"{a}-{b}"] = boot(d, np.array([X[(a, nm, s)][2] for s in src]))
    sig = lambda t: t[1] > 0 or t[2] < 0
    sigma = [k for k, t in pairs["cv"].items() if sig(t)]
    same = [k for k in sigma if np.sign(pairs[amp][k][0]) == np.sign(pairs["cv"][k][0])]
    strong = [k for k in same if sig(pairs[amp][k])]
    verdict = "R_ok" if len(strong) == len(sigma) else "R_weak" if len(same) == len(sigma) else "R_no"
    res = {"prereg": "results/snv_p74/prereg.yaml(step_a)", "amp": amp, "mean": mean, "val6_x": val6, "sigma": sigma, "sigma_same_sign": len(same), "sigma_sig_on_amp": len(strong), "reversed": [k for k in sigma if k not in same],
           "weak": [k for k in same if k not in strong], "amp_sig_not_in_sigma": [k for k, t in pairs[amp].items() if sig(t) and k not in sigma], "verdict": verdict, "pairs": pairs}
    if opt.get("dose"):
        D = json.loads((ROOT / opt["dose"]).read_text())["rows"]
        dose_of = lambda rep: 0.0 if rep == "copy" else int(rep[2:]) / 10
        curve = {nm: sorted((dose_of(rep), float(np.mean([r["cer"] for r in D if r["set"] == nm and r["rep"] == rep]))) for rep in dict.fromkeys(r["rep"] for r in D if r["set"] == nm)) for nm in sets}
        inv = lambda nm, c: float(np.interp(c, np.maximum.accumulate([y for _, y in curve[nm]]), [x for x, _ in curve[nm]]))
        fwd = lambda nm, d: float(np.interp(d, [x for x, _ in curve[nm]], [y for _, y in curve[nm]]))
        res["curve"] = {nm: [(x, round(y, 4)) for x, y in curve[nm]] for nm in sets}
        res["dose"] = {s: {nm: {"self": round(inv(nm, mean[nm][s]["self"]), 3), "x": round(inv(nm, mean[nm][s]["x"]), 3)} for nm in sets} for s in syss}
        res["cv_pred_from_amp"] = {s: {"pred": round(fwd("cv", res["dose"][s][amp]["x"]), 4), "actual": mean["cv"][s]["x"]} for s in syss}
    (ROOT / opt.get("out", "results/snv_p74/concord.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, ensure_ascii=False, indent=1))
    for nm in sets:
        print(nm, {k: v for k, v in pairs[nm].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
