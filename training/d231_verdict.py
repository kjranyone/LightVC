"""D231 の最終判定(results/conv_p0/d231_prereg.yaml・評価の組・一度だけ): RRQSDC19L の (i) 4 軸 (ii) 余裕 (iii) CER (iv) 声域の中心の合格線。
    uv run python d231_verdict.py --s <scratchpad> --root <評価の組> --out ../results/conv_p0/d231.json
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
CAND = "RRQSDC19L"
DESC = ("RRQSD19L", "RRQS19L", "G03S10ZH")
KEYS = {c: f"d139:{c}" for c in (CAND, "RRQSD19L", "RRQS19L")} | {"G03S10ZH": "d114:G03S10ZH"}


def per_utt(S: Path, c: str) -> list:
    cache = json.loads((S / "d109_cache.json").read_text())
    d, form = KEYS[c].split(":")
    return [("eval", int(f.name.split("_")[1][1:]), cache[f"2|set|{f}|{f.stat().st_mtime_ns}"]) for f in sorted((S / d).glob(f"*_{form}.wav"))]


def margins(rows: list, bounds: dict, rng: np.random.Generator) -> dict:
    cl = np.array([f"{r[0]}:{r[1]}" for r in rows]); us = sorted(set(cl)); idx = {u: np.flatnonzero(cl == u) for u in us}
    B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(2000)]
    out = {}
    for k in MOVED:
        v = np.array([r[2][k] for r in rows], float)
        md = float(np.nanmedian(v))
        bs = [np.nanmedian(v[ii]) for ii in B]
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
    ax = json.loads((ROOT / "results/conv_p0/d109v2_eval231.json").read_text())
    bounds = {k: v["real_p15_p85"] for k, v in ax[KEYS["G03S10ZH"]]["stats"].items()}
    rep["axes"] = {c: {"pass": ax[kk]["pass"], "fail": [k for k, v in ax[kk]["stats"].items() if not v["pass"]],
                       "shimmer": ax[kk]["stats"]["shimmer"]["median"], "lmstd": [ax[kk]["stats"]["lmstd"]["median"], ax[kk]["stats"]["lmstd"]["out_frac"]]} for c, kk in KEYS.items()}
    rng = np.random.default_rng(1)
    rep["margins"] = {c: margins(per_utt(RT, c), bounds, rng) for c in KEYS}
    cer = json.loads((ROOT / "results/conv_p0/d231_cer.json").read_text())["eval"]
    rep["cer"] = {c: round(float(v["cer_mean"]), 4) for c, v in cer.items() if c not in ("per", "examples")}
    ok = {m["ti"] for m in json.loads((RT / "cen_reg/meta.json").read_text()) if m["ok"]}
    rep["targets_reg_ok"] = len(ok)
    for ruler, tag in (("REG", "d231reg"), ("IND", "d231ind")):
        R = V.ranks(S / f"{tag}_lda.npz", ROOT / f"results/conv_p0/{tag}_tri.json")
        for m in ("LDA", "ECAPA"):
            blk = {"levels": {c: round(float(np.mean([R[m][c][k] for k in R[m][c] if int(k.split("_t")[1]) in ok])), 2) for c in (CAND,) + DESC + ("KNN",)}}
            for c in (CAND,) + DESC:
                rows = [("eval", int(k.split("_t")[1]), R[m][c][k] - R[m]["KNN"][k]) for k in sorted(set(R[m][c]) & set(R[m]["KNN"])) if int(k.split("_t")[1]) in ok]
                blk[f"{c}-KNN"] = ci(rows, np.random.default_rng(0))
            rep[f"{ruler}_{m}"] = blk
    cond = {"i_axes": rep["axes"][CAND]["pass"], "ii_margin": all(v["ok"] for v in rep["margins"][CAND].values()),
            "iii_cer": rep["cer"][CAND] <= rep["cer"]["G03S10ZH"] + 0.03,
            "iv_owner_line": all(rep[f"REG_{m}"][f"{CAND}-KNN"]["rank_diff"] <= 0 and rep[f"REG_{m}"][f"{CAND}-KNN"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA"))}
    rep["conditions"] = cond
    rep["verdict"] = "PASS" if all(cond.values()) else "FAIL_ID" if cond["i_axes"] and cond["ii_margin"] and cond["iii_cer"] else "FAIL_NAT"
    print(json.dumps({k: rep[k] for k in ("conditions", "verdict", "cer", "targets_reg_ok")}, indent=1, ensure_ascii=False))
    for r in ("REG", "IND"):
        for m in ("LDA", "ECAPA"):
            print(r, m, rep[f"{r}_{m}"]["levels"], {c: (rep[f"{r}_{m}"][f"{c}-KNN"]["rank_diff"], rep[f"{r}_{m}"][f"{c}-KNN"]["ci95"]) for c in (CAND,) + DESC})
    print("axes", {c: (v["pass"], v["fail"]) for c, v in rep["axes"].items()})
    print("margins_fail", {c: [k for k, v in mm.items() if not v["ok"]] for c, mm in rep["margins"].items()})
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
