"""N2(学習なし): 物差しの候補(UTMOS・AES の PQ / CE・WavLM の近傍の類似)の対の勝率を読み、開発の組で 1 つ選んで検査の組で判定する。
事前登録 = results/snv_n2/prereg.yaml(線・候補・選び方はそこに固定)。出力 = results/snv_n2/n2_judge.json。
    cd training && uv run --no-sync python snv_n2_judge.py <in>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/snv_n2"
FAM = {"V": ("clean", "V"), "B1": ("V", "B1"), "B2": ("V", "B2"), "B3": ("V", "B3"), "GL": ("clean", "GL"), "N10": ("clean", "N10"),
       "N20": ("clean", "N20"), "N40": ("clean", "N40"), "BZ": ("clean", "BZ"), "LP4": ("clean", "LP4"), "LP8": ("clean", "LP8"),
       "Sg": ("Vf", "Sg"), "Sx": ("Vf", "Sx")}
GATE = {"real_F": ["GL", "N20", "BZ", "LP4", "B2", "B3"], "iro_F": ["GL", "N20", "BZ", "B2", "B3"], "iro_M": ["Sg", "Sx"]}
LINE, LO = 0.90, 0.85
KNN = 5
CANDS = ("utmos", "aes_pq", "aes_ce", "knn")


def boot(w: np.ndarray, grp: np.ndarray, n: int = 2000) -> list:
    rng = np.random.default_rng(0)
    gs = np.unique(grp)
    if len(gs) < 5:
        bs = [w[rng.integers(0, len(w), len(w))].mean() for _ in range(n)]
    else:
        idx = {g: np.where(grp == g)[0] for g in gs}
        bs = [np.concatenate([w[idx[g]] for g in rng.choice(gs, len(gs))]).mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main() -> int:
    src = Path(sys.argv[1])
    its = json.loads((OUT / "items.json").read_text())
    mos = json.loads((OUT / "utmos.json").read_text())
    aes = json.loads((OUT / "aes.json").read_text())
    emb = np.load(src / "wavlm.npz")
    ref = defaultdict(list)
    for it in its:
        ref[(it["pop"], it["split"])].append(emb[f"clean|{it['name']}"])
    ref = {k: np.stack(v) for k, v in ref.items()}

    def score(c: str, it: dict, v: str) -> float:
        k = f"{v}|{it['name']}"
        if c == "utmos":
            return mos[k]
        if c == "aes_pq":
            return aes[k]["PQ"]
        if c == "aes_ce":
            return aes[k]["CE"]
        other = "test" if it["split"] == "dev" else "dev"
        return float(np.sort(ref[(it["pop"], other)] @ emb[k])[-KNN:].mean())

    res: dict = {"n_items": {f"{p}_{s}": sum(1 for it in its if it["pop"] == p and it["split"] == s) for p in ("real_F", "real_M", "iro_F", "iro_M") for s in ("dev", "test")}}
    cells: dict = {}
    for c in CANDS:
        for pop in ("real_F", "real_M", "iro_F", "iro_M"):
            for sp in ("dev", "test"):
                sel = [it for it in its if it["pop"] == pop and it["split"] == sp]
                for fam, (a, b) in FAM.items():
                    if fam in ("Sg", "Sx") and pop != "iro_M":
                        continue
                    sa = np.array([score(c, it, a) for it in sel])
                    sb = np.array([score(c, it, b) for it in sel])
                    w = (sa > sb) + 0.5 * (sa == sb)
                    grp = np.array([it["spk"] if pop.startswith("iro") else it["name"] for it in sel])
                    cells[(c, pop, sp, fam)] = {"win": round(float(w.mean()), 3), "ci95": boot(w, grp), "n": len(sel), "d_mean": round(float(np.mean(sa - sb)), 4)}

    def ok(c: str, sp: str) -> tuple[bool, float, list]:
        fails, mins = [], []
        for pop, fams in GATE.items():
            for fam in fams:
                r = cells[(c, pop, sp, fam)]
                mins.append(r["win"])
                if r["win"] < LINE or r["ci95"][0] < LO:
                    fails.append(f"{pop}:{fam} {r['win']} {r['ci95']}")
        for pop in ("real_F", "iro_F"):
            w1, w2, w3 = (cells[(c, pop, sp, b)]["win"] for b in ("B1", "B2", "B3"))
            if not (w1 <= w2 + 0.02 and w2 <= w3 + 0.02):
                fails.append(f"{pop}:mono {w1} {w2} {w3}")
        return not fails, float(min(mins)), fails

    dev = {c: ok(c, "dev") for c in CANDS}
    passed = [c for c in CANDS if dev[c][0]]
    chosen = max(passed or CANDS, key=lambda c: dev[c][1])
    test = ok(chosen, "test")
    res["dev"] = {c: {"pass": v[0], "min_win": round(v[1], 3), "fails": v[2]} for c, v in dev.items()}
    res["chosen"] = chosen
    res["chosen_from_passed_dev"] = bool(passed)
    res["test"] = {"pass": test[0], "min_win": round(test[1], 3), "fails": test[2]}
    res["test_all_candidates"] = {c: {"pass": v[0], "min_win": round(v[1], 3), "fails": v[2]} for c, v in ((c, ok(c, "test")) for c in CANDS)}
    res["cells"] = {f"{c}|{pop}|{sp}|{fam}": r for (c, pop, sp, fam), r in cells.items()}

    clean_cat: dict = {}
    for c in CANDS:
        for pop in ("real_F", "iro_F"):
            by = defaultdict(list)
            for it in its:
                if it["pop"] == pop:
                    by[it["cat"]].append(score(c, it, "clean"))
            clean_cat[f"{c}|{pop}"] = {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}
    res["clean_by_cat"] = clean_cat
    auth = {r.split(",")[0]: float(r.split(",")[1]) for r in (ROOT / "data/jnv/JNV/file_authenticity.csv").read_text().splitlines()[1:]}
    jn = [it for it in its if it["spk"].startswith("jnv_") and it["name"][4:] in auth]
    res["jnv_authenticity_spearman"] = {c: round(float(spearmanr([score(c, it, "clean") for it in jn], [auth[it["name"][4:]] for it in jn])[0]), 3) for c in CANDS}
    hf = json.loads((OUT / "hf.json").read_text())
    res["hf_median"] = {pop: {v: round(float(np.median([hf[it["name"]][v] for it in its if it["pop"] == pop and v in hf[it["name"]]])), 2)
                              for v in ("clean", "V", "B1", "B2", "B3", "GL", "N20", "BZ", "LP4", "LP8", "Vf", "Sg", "Sx") if any(v in hf[it["name"]] for it in its if it["pop"] == pop)}
                        for pop in ("real_F", "real_M", "iro_F", "iro_M")}
    for c in CANDS:
        print(c, "dev", res["dev"][c], flush=True)
    print("chosen", chosen, "test", res["test"], flush=True)
    print("auth", res["jnv_authenticity_spearman"], flush=True)
    print("hf", res["hf_median"], flush=True)
    (OUT / "n2_judge.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
