"""diag_rr20d の判定(results/diag_rr20d/prereg.yaml): クローンの話者を 2.2 倍にした R の製品の経路の形(RRQSDC20D・RRQSD20D)の候補の 3 条件・合格線と、rr18p との本人らしさの差・クローンの読み(rr19L_verdict.py の写し・形の名前だけ替える)。
    uv run python rr20d_verdict.py --s <scratchpad> --root <dev の親> --out ../results/diag_rr20d/verdict_eval_real.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import d181_verdict as V
from rr17c_verdict import ci

ROOT = Path(__file__).resolve().parent.parent
SETS = ("dev", "dev2", "dev3")
FORMS = ("RRQSDC20D", "RRQSD20D")
DESC = ("RRQS20D", "RRQS18P", "RRQSD18P", "RRQSDC18P")
KEYS = {c: f"d139:{c}" for c in FORMS + DESC} | {"G03S10ZH": "d114:G03S10ZH"}
SIDE = {"jump": "hi", "range": "both", "micro": "both", "vfrac": "both", "contrast": "both", "amline": "hi", "shimmer": "both", "hf": "both", "tilt": "both", "lmstd": "both",
        "ltas_d": "hi", "dtilt": "both", "dhf": "both"}
MOVED = ("contrast", "amline", "shimmer", "hf", "tilt", "lmstd", "ltas_d", "dtilt", "dhf")
PAIRS = [(c, "KNN") for c in FORMS + DESC + ("G03S10ZH",)] + [(c, "G03S10ZH") for c in FORMS + DESC] + [("RRQS20D", "RRQS18P"), ("RRQSD20D", "RRQSD18P"), ("RRQSDC20D", "RRQSDC18P"), ("RRQSDC20D", "RRQSD20D")]


def per_utt(root: Path, D: str, c: str) -> list:
    S = root / D
    cache = json.loads((S / "d109_cache.json").read_text())
    d, form = KEYS[c].split(":")
    rows = []
    for f in sorted((S / d).glob(f"*_{form}.wav")):
        rows.append((D, int(f.name.split("_")[1][1:]), cache[f"2|set|{f}|{f.stat().st_mtime_ns}"]))
    return rows


def margins(rows: list, bounds: dict, rng: np.random.Generator) -> dict:
    cl = np.array([f"{r[0]}:{r[1]}" for r in rows]); us = sorted(set(cl)); idx = {u: np.flatnonzero(cl == u) for u in us}
    B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(2000)]
    out = {}
    for k in MOVED:
        v = np.array([r[2][k] for r in rows], float)
        ok = np.isfinite(v)
        md = float(np.median(v[ok]))
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
    runs = {D: V.ranks(S / f"rr20{D}_lda.npz", ROOT / f"results/conv_p0/rr20{D}_tri.json") for D in SETS}
    rep: dict = {}
    for m in ("LDA", "ECAPA"):
        blk: dict = {f"levels_{D}": {c: round(float(np.mean(list(r.values()))), 2) for c, r in sorted(runs[D][m].items())} for D in SETS}
        for p, o in PAIRS:
            rows = {D: [(D, int(k.split("_t")[1]), R[m][p][k] - R[m][o][k]) for k in sorted(set(R[m][p]) & set(R[m][o]))] for D, R in runs.items()}
            rng = np.random.default_rng(0)
            blk[f"{p}-{o}"] = {"pooled90": ci(sum(rows.values(), []), rng), "dev_dev2": ci(rows["dev"] + rows["dev2"], rng), "dev3": ci(rows["dev3"], rng)}
        rep[m] = blk
    cer = {D: json.loads((ROOT / f"results/conv_p0/rr20{D}_cer.json").read_text())[D] for D in SETS}
    rep["cer"] = {D: {c: round(float(v["cer_mean"]), 4) for c, v in cer[D].items() if c not in ("per", "examples")} for D in SETS}
    rep["cer_mean_3"] = {c: round(float(np.mean([rep["cer"][D][c] for D in SETS])), 4) for c in rep["cer"]["dev"] if all(c in rep["cer"][D] for D in SETS)}
    ax = {D: json.loads((ROOT / f"results/conv_p0/d109v2_rr20{D}.json").read_text()) for D in SETS}
    bounds = {k: v["real_p15_p85"] for k, v in ax["dev"][KEYS["G03S10ZH"]]["stats"].items()}
    assert all(ax[D][KEYS["G03S10ZH"]]["stats"][k]["real_p15_p85"] == bounds[k] for D in SETS for k in bounds)
    rep["axes"] = {c: {D: {"pass": ax[D][kk]["pass"], "fail": [k for k, v in ax[D][kk]["stats"].items() if not v["pass"]],
                           "shimmer": ax[D][kk]["stats"]["shimmer"]["median"], "lmstd": [ax[D][kk]["stats"]["lmstd"]["median"], ax[D][kk]["stats"]["lmstd"]["out_frac"]]} for D in SETS}
                   for c, kk in KEYS.items()}
    rng = np.random.default_rng(1)
    pu = {c: sum((per_utt(RT, D, c) for D in SETS), []) for c in KEYS}
    rep["margins_pooled180"] = {c: margins(pu[c], bounds, rng) for c in KEYS}
    med = lambda c, k, D=None: float(np.nanmedian([r[2][k] for r in pu[c] if D is None or r[0] == D]))
    rep["attribution"] = {f"{p}-{o}": {k: {D: round(med(p, k, D) - med(o, k, D), 3) for D in SETS + (None,)} for k in ("lmstd", "shimmer", "contrast", "hf", "tilt")}
                          for p, o in (("RRQS20D", "RRQS18P"), ("RRQSD20D", "RRQSD18P"), ("RRQSDC20D", "RRQSDC18P"), ("RRQSDC20D", "G03S10ZH"), ("RRQSD20D", "G03S10ZH"))}
    lim = rep["cer_mean_3"]["G03S10ZH"] + 0.03
    cond = {c: {"i_axes_3sets": all(rep["axes"][c][D]["pass"] for D in SETS), "ii_margin": all(v["ok"] for v in rep["margins_pooled180"][c].values()),
                "iii_cer": rep["cer_mean_3"][c] <= lim} for c in FORMS + DESC}
    rep["conditions"] = cond
    ol = lambda c: all(rep[m][f"{c}-KNN"]["pooled90"]["rank_diff"] <= 0 and rep[m][f"{c}-KNN"]["pooled90"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA"))
    for c in FORMS + DESC:
        cond[c]["owner_line"] = ol(c)
    cand = [c for c in FORMS if all(cond[c].values())]
    rep["candidates"] = cand
    worst = lambda c: max(rep[m][f"{c}-KNN"]["pooled90"]["rank_diff"] for m in ("LDA", "ECAPA"))
    rep["choice"] = ("RRQSDC20D" if worst("RRQSD20D") - worst("RRQSDC20D") > 0.25 else "RRQSD20D") if len(cand) == 2 else cand[0] if cand else None
    up = rep["LDA"]["RRQS20D-RRQS18P"]["pooled90"]["ci95"][1] < 0
    rep["verdict"] = "CAND" if cand else "NOT_CAND_UP" if up else "NOT_CAND_FLAT"
    ve = ROOT / "results/diag_rr20d/verdict_eval.json"
    if ve.exists():
        rep["clone_unseen_minus_rr18p"] = json.loads(ve.read_text()).get("diag_rr20d_minus_diag_rr18p", {}).get("unseen_all")
    rep["owner_line_descriptive_pooled90"] = {p: "MET" if all(rep[m][f"{p}-KNN"]["pooled90"]["rank_diff"] <= 0 and rep[m][f"{p}-KNN"]["pooled90"]["ci95"][1] <= 1.0 for m in ("LDA", "ECAPA")) else "NOT_MET"
                                              for p, o in PAIRS if o == "KNN"}
    print(json.dumps({k: rep[k] for k in ("conditions", "candidates", "choice", "verdict", "owner_line_descriptive_pooled90", "attribution", "cer_mean_3")}, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
