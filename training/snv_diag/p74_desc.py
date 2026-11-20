"""段 P74 step c(results/snv_p74/prereg.yaml・学習なし・ASR なし): 源の性質(前段 v3 の特徴から)と、源の脆さ F の並び。
性質 = range_lo / range_hi(有声の c1〜c8 / c9〜c24 の時間の標準偏差)・speed(続く有声のフレームの c1〜c8 の差分の大きさ)・speed_rel(speed ÷ range_lo)・rate(仮名 / 秒・cv だけ)・
vfrac / vflip(有声の率 / 切り替わりの回数 / 秒)・f0med / f0sd / f0jump(高さの中央値 / 対数の標準偏差 / 対数で 0.08 より大きく動く率)・ap(有声の周期性 4 帯の平均)・c0rng(有声の強さの 95 % 点 − 5 % 点)。
F = (nz3 + nz5 + nz7) / 3 − copy(frag=<p73_swap.py の json>・cv の源)。話者ごとに平均し、順位相関と区間(話者の再標本 2000 回・種 0)を出す。
    cd training && uv run --no-sync python snv_diag/p74_desc.py frag=results/snv_p74/frag_cv.json [sets=cv,valm,trm,x:d_clean,jv,x:qw] [out=results/snv_p74/desc.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p18_align as AL

ROOT = Path(__file__).resolve().parents[2]
HOP = 0.005


def desc(u: dict, ref: str | None) -> dict:
    y, v, lf = u["y"], u["v"] > 0.5, u["lf"]
    pair = v[1:] & v[:-1]
    d = np.diff(y[1:9], axis=1)[:, pair]
    lo = float(np.sqrt(y[1:9][:, v].var(1).mean()))
    sp = float(np.sqrt((d ** 2).sum(0)).mean())
    dur = y.shape[1] * HOP
    c0 = y[0, v]
    return {"range_lo": lo, "range_hi": float(np.sqrt(y[9:25][:, v].var(1).mean())), "speed": sp, "speed_rel": sp / lo, "vfrac": float(v.mean()), "vflip": float((v[1:] != v[:-1]).sum() / dur),
            "f0med": float(np.exp(np.median(lf[v]))), "f0sd": float(lf[v].std()), "f0jump": float((np.abs(np.diff(lf))[pair] > 0.08).mean()), "ap": float(y[25:29][:, v].mean()),
            "c0rng": float(np.percentile(c0, 95) - np.percentile(c0, 5))} | ({"rate": len(ref) / dur} if ref is not None else {})


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    names = opt.get("sets", "cv,valm,trm,x:d_clean,jv,x:qw").split(",")
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    sets = AL.build_sets(names, 64, split)
    rows = [{"set": nm, "spk": it["spk"], "src": it["src"]} | desc(it["u"], it.get("ref") if nm == "cv" else None) for nm, its in sets.items() for it in its if it["valid"]]
    keys = [k for k in rows[0] if k not in ("set", "spk", "src")]
    spk_mean = lambda nm, k: np.array([np.mean([r[k] for r in rows if r["set"] == nm and r["spk"] == s]) for s in sorted({r["spk"] for r in rows if r["set"] == nm})])
    dist = {nm: {k: [round(float(q), 4) for q in np.percentile(spk_mean(nm, k), [10, 50, 90])] for k in keys if k != "rate" or nm == "cv"} for nm in sets}
    F = {}
    for r in json.loads((ROOT / opt["frag"]).read_text())["rows"]:
        F.setdefault(r["src"], {})[r["rep"]] = r["cer"]
    F = {s: float(np.mean([c[k] for k in ("nz3", "nz5", "nz7")]) - c["copy"]) for s, c in F.items()}
    cv = [r | {"F": F[r["src"]]} for r in rows if r["set"] == "cv" and r["src"] in F]
    sp = sorted({r["spk"] for r in cv})
    A = {k: np.array([np.mean([r[k] for r in cv if r["spk"] == s]) for s in sp]) for k in keys + ["F"]}
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(sp), len(sp)) for _ in range(2000)]
    rho = {}
    for k in keys:
        b = [spearmanr(A[k][i], A["F"][i])[0] for i in idx]
        rho[k] = [round(float(spearmanr(A[k], A["F"])[0]), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
    o = np.argsort(A["F"])
    t = len(sp) // 3
    strata = {k: [round(float(A[k][o[:t]].mean()), 4), round(float(A[k][o[t:-t]].mean()), 4), round(float(A[k][o[-t:]].mean()), 4)] for k in keys + ["F"]}
    res = {"prereg": "results/snv_p74/prereg.yaml(step_c)", "n_src": len(cv), "n_spk": len(sp), "rho_F": rho, "strata_F(下 1/3・中・上 1/3)": strata, "dist(話者の 10・50・90 %)": dist, "rows": cv}
    (ROOT / opt.get("out", "results/snv_p74/desc.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
