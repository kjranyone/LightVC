"""D232 の判定(results/conv_p0/d232_prereg.yaml): RRQSDC19LE(等化を R の形の操作者の変換で測る)の dev 3 組の 3 条件・声域の中心の合格線・hf の明るさ。
    uv run python d232_verdict.py --s <scratchpad> --root <dev の親> --out ../results/conv_p0/d232.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import d181_verdict as V
from rr17c_verdict import ci
from rr19L_verdict import MOVED, SIDE

ROOT = Path(__file__).resolve().parent.parent
SETS = ("dev", "dev2", "dev3")
CAND = "RRQSDC19LE"
KEYS = {CAND: f"d139:{CAND}", "RRQSDC19L": "d139:RRQSDC19L", "G03S10ZH": "d114:G03S10ZH"}


def per_utt(root: Path, D: str, c: str) -> list:
    cache = json.loads((root / D / "d109_cache.json").read_text())
    d, form = KEYS[c].split(":")
    return [(D, int(f.name.split("_")[1][1:]), cache[f"2|set|{f}|{f.stat().st_mtime_ns}"]) for f in sorted((root / D / d).glob(f"*_{form}.wav"))]


def margins(rows: list, bounds: dict, rng: np.random.Generator) -> dict:
    cl = np.array([f"{r[0]}:{r[1]}" for r in rows]); us = sorted(set(cl)); idx = {u: np.flatnonzero(cl == u) for u in us}
    B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(2000)]
    out = {}
    for k in MOVED:
        v = np.array([r[2][k] for r in rows], float)
        md = float(np.nanmedian(v)); bs = [np.nanmedian(v[ii]) for ii in B]
        hw = float(np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2
        l15, h15 = bounds[k]
        mg = h15 - md if SIDE[k] == "hi" else min(md - l15, h15 - md)
        out[k] = {"median": round(md, 3), "margin": round(mg, 3), "half_width": round(hw, 3), "ok": bool(mg > hw)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S, RT = Path(a.s), Path(a.root)
    rep: dict = {}
    ax = {D: json.loads((ROOT / f"results/conv_p0/d109v2_d232{D}.json").read_text()) for D in SETS}
    bounds = {k: v["real_p15_p85"] for k, v in ax["dev"][KEYS["G03S10ZH"]]["stats"].items()}
    rep["axes"] = {c: {D: {"pass": ax[D][kk]["pass"], "fail": [k for k, v in ax[D][kk]["stats"].items() if not v["pass"]]} for D in SETS} for c, kk in KEYS.items()}
    pu = {c: sum((per_utt(RT, D, c) for D in SETS), []) for c in KEYS}
    rng = np.random.default_rng(1)
    rep["margins_pooled180"] = {c: margins(pu[c], bounds, rng) for c in KEYS}
    med = lambda c, k: float(np.nanmedian([r[2][k] for r in pu[c]]))
    rep["hf_diff"] = {f"{c}-G03S10ZH": round(med(c, "hf") - med("G03S10ZH", "hf"), 3) for c in (CAND, "RRQSDC19L")}
    cer = {D: json.loads((ROOT / f"results/conv_p0/d232{D}_cer.json").read_text())[D] for D in SETS}
    rep["cer_mean_3"] = {c: round(float(np.mean([cer[D][c]["cer_mean"] for D in SETS])), 4) for c in list(KEYS) + ["KNN"]}
    ok = {D: {m["ti"] for m in json.loads((RT / D / "cen_reg/meta.json").read_text()) if m["ok"]} for D in SETS}
    R = {D: V.ranks(S / f"d232{D}_lda.npz", ROOT / f"results/conv_p0/d232{D}_tri.json") for D in SETS}
    for m in ("LDA", "ECAPA"):
        rep[f"REG_{m}"] = {}
        for c in KEYS:
            rows = [(D, int(k.split("_t")[1]), R[D][m][c][k] - R[D][m]["KNN"][k]) for D in SETS for k in sorted(set(R[D][m][c]) & set(R[D][m]["KNN"])) if int(k.split("_t")[1]) in ok[D]]
            rep[f"REG_{m}"][f"{c}-KNN"] = ci(rows, np.random.default_rng(0))
        rows = [(D, int(k.split("_t")[1]), R[D][m][CAND][k] - R[D][m]["RRQSDC19L"][k]) for D in SETS for k in sorted(R[D][m][CAND]) if int(k.split("_t")[1]) in ok[D]]
        rep[f"REG_{m}"][f"{CAND}-RRQSDC19L"] = ci(rows, np.random.default_rng(0))
    lim = rep["cer_mean_3"]["G03S10ZH"] + 0.03
    cond = {"i_axes_3sets": all(rep["axes"][CAND][D]["pass"] for D in SETS), "ii_margin": all(v["ok"] for v in rep["margins_pooled180"][CAND].values()),
            "iii_cer": rep["cer_mean_3"][CAND] <= lim,
            "iv_owner_line_reg": all(rep[f"REG_{m}"][f"{CAND}-KNN"]["rank_diff"] <= 0 and rep[f"REG_{m}"][f"{CAND}-KNN"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA"))}
    rep["conditions"] = cond
    fixed = rep["hf_diff"][f"{CAND}-G03S10ZH"] <= 0.25
    rep["verdict"] = "GO_EVAL2" if fixed and all(cond.values()) else "NO_FIX" if not fixed else "ID_LOSS"
    print(json.dumps({k: rep[k] for k in ("conditions", "hf_diff", "cer_mean_3", "verdict")}, indent=1, ensure_ascii=False))
    for m in ("LDA", "ECAPA"):
        print(m, {k: (v["rank_diff"], v["ci95"]) for k, v in rep[f"REG_{m}"].items()})
    print("hf margins", {c: rep["margins_pooled180"][c]["hf"] for c in KEYS})
    print("axes", {c: {D: v[D]["fail"] for D in SETS} for c, v in rep["axes"].items()})
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
