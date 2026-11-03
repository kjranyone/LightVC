"""N2b(学習なし・物差しの較正): N2 の既成の物差しの点の線形の組み合わせを、清音 対 劣化の対の順位で学び、学ばなかった劣化の型と回帰の描画(Sg / Sx)で確かめる。
事前登録 = results/snv_n2b/prereg.yaml。素材と点は N2 のもの(新しく測らない)。出力 = results/snv_n2b/n2b.json・ruler.json(重みと標準化)。
    cd training && uv run --no-sync python snv_n2b_fit.py <n2 の素材の dir>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from snv_n2_judge import FAM, boot

ROOT = Path(__file__).resolve().parent.parent
N2 = ROOT / "results/snv_n2"
OUT = ROOT / "results/snv_n2b"
FEATS = ("utmos", "aes_pq", "aes_ce", "aes_cu", "aes_pc", "knn")
TRAIN_FAM = ("B1", "B2", "B3", "N10", "N20", "BZ", "LP4", "LP8")
GROUPS = {"B": ("B1", "B2", "B3"), "N": ("N10", "N20"), "BZ": ("BZ",), "LP": ("LP4", "LP8")}
REP = {"B": "B2", "N": "N20", "BZ": "BZ", "LP": "LP4"}
GATE = {"real_F": ["N20", "BZ", "LP4", "B2", "B3"], "iro_F": ["N20", "BZ", "B2", "B3"], "iro_M": ["Sg", "Sx"]}
LINE, LO, LOFO = 0.90, 0.85, 0.80
KNN = 5


def main() -> int:
    src = Path(sys.argv[1])
    its = json.loads((N2 / "items.json").read_text())
    mos = json.loads((N2 / "utmos.json").read_text())
    aes = json.loads((N2 / "aes.json").read_text())
    emb = np.load(src / "wavlm.npz")
    ref = defaultdict(list)
    for it in its:
        ref[(it["pop"], it["split"])].append(emb[f"clean|{it['name']}"])
    ref = {k: np.stack(v) for k, v in ref.items()}

    def raw(it: dict, v: str) -> np.ndarray:
        k = f"{v}|{it['name']}"
        other = "test" if it["split"] == "dev" else "dev"
        kn = float(np.sort(ref[(it["pop"], other)] @ emb[k])[-KNN:].mean())
        return np.array([mos[k], aes[k]["PQ"], aes[k]["CE"], aes[k]["CU"], aes[k]["PC"], kn])

    cache = {}

    def phi(it: dict, v: str) -> np.ndarray:
        if (it["name"], v) not in cache:
            cache[(it["name"], v)] = raw(it, v)
        return cache[(it["name"], v)]

    dev_clean = np.stack([phi(it, "clean") for it in its if it["split"] == "dev"])
    m, s = dev_clean.mean(0), dev_clean.std(0) + 1e-6
    z = lambda it, v: (phi(it, v) - m) / s

    def pairs(split: str, fams: tuple, pops: tuple) -> np.ndarray:
        X = []
        for it in its:
            if it["split"] != split or it["pop"] not in pops:
                continue
            for f in fams:
                a, b = FAM[f]
                if f"{b}|{it['name']}" in mos:
                    X.append(z(it, a) - z(it, b))
        return np.stack(X)

    def fit(fams: tuple) -> np.ndarray:
        X = pairs("dev", fams, ("real_F", "real_M", "iro_F", "iro_M"))
        clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=5000).fit(np.concatenate([X, -X]), np.r_[np.ones(len(X)), np.zeros(len(X))])
        return clf.coef_[0]

    def win(w: np.ndarray, pop: str, split: str, fam: str) -> dict:
        sel = [it for it in its if it["pop"] == pop and it["split"] == split and f"{FAM[fam][1]}|{it['name']}" in mos]
        d = np.array([w @ (z(it, FAM[fam][0]) - z(it, FAM[fam][1])) for it in sel])
        wv = (d > 0) + 0.5 * (d == 0)
        grp = np.array([it["spk"] if pop.startswith("iro") else it["name"] for it in sel])
        return {"win": round(float(wv.mean()), 3), "ci95": boot(wv, grp), "n": len(sel)}

    res: dict = {"prereg": "results/snv_n2b/prereg.yaml", "features": FEATS, "lofo": {}}
    lofo_fail = []
    for gname, gf in GROUPS.items():
        w = fit(tuple(f for f in TRAIN_FAM if f not in gf))
        for pop in ("real_F", "iro_F"):
            r = win(w, pop, "dev", REP[gname])
            res["lofo"][f"{gname}|{pop}"] = r | {"w": [round(float(x), 3) for x in w]}
            if r["win"] < LOFO:
                lofo_fail.append(f"{gname}|{pop} {r['win']}")
    w = fit(TRAIN_FAM)
    res["w"] = dict(zip(FEATS, [round(float(x), 4) for x in w]))
    res["standardize"] = {"mean": dict(zip(FEATS, m.round(5).tolist())), "std": dict(zip(FEATS, s.round(5).tolist()))}
    cells, fails = {}, []
    for pop in ("real_F", "real_M", "iro_F", "iro_M"):
        for sp in ("dev", "test"):
            for fam in FAM:
                if fam in ("Sg", "Sx") and pop != "iro_M":
                    continue
                cells[f"{pop}|{sp}|{fam}"] = win(w, pop, sp, fam)
    for pop, fams in GATE.items():
        for fam in fams:
            r = cells[f"{pop}|test|{fam}"]
            if r["win"] < LINE or r["ci95"][0] < LO:
                fails.append(f"{pop}:{fam} {r['win']} {r['ci95']}")
    for pop in ("real_F", "iro_F"):
        w1, w2, w3 = (cells[f"{pop}|test|{b}"]["win"] for b in ("B1", "B2", "B3"))
        if not (w1 <= w2 + 0.02 and w2 <= w3 + 0.02):
            fails.append(f"{pop}:mono {w1} {w2} {w3}")
    res["cells"] = cells
    res["lofo_fails"] = lofo_fail
    res["test_fails"] = fails
    res["pass"] = not lofo_fail and not fails
    score = lambda it, v: float(w @ z(it, v))
    bycat = {}
    for pop in ("real_F", "iro_F"):
        by = defaultdict(list)
        for it in its:
            if it["pop"] == pop:
                by[it["cat"]].append(score(it, "clean"))
        bycat[pop] = {k: round(float(np.mean(v)), 3) for k, v in sorted(by.items())}
    res["clean_by_cat"] = bycat
    auth = {r.split(",")[0]: float(r.split(",")[1]) for r in (ROOT / "data/jnv/JNV/file_authenticity.csv").read_text().splitlines()[1:]}
    jn = [it for it in its if it["spk"].startswith("jnv_") and it["name"][4:] in auth]
    res["jnv_authenticity_spearman"] = round(float(spearmanr([score(it, "clean") for it in jn], [auth[it["name"][4:]] for it in jn])[0]), 3)
    (OUT / "n2b.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (OUT / "ruler.json").write_text(json.dumps({"features": FEATS, "w": res["w"], "standardize": res["standardize"], "knn": {"k": KNN, "layer": 12, "model": "microsoft/wavlm-large"}}, ensure_ascii=False, indent=1))
    print("w", res["w"], flush=True)
    print("lofo", {k: v["win"] for k, v in res["lofo"].items()}, "fails", lofo_fail, flush=True)
    for pop in ("real_F", "iro_F", "iro_M", "real_M"):
        print(pop, {f: cells[f"{pop}|test|{f}"]["win"] for f in FAM if f"{pop}|test|{f}" in cells}, flush=True)
    print("test_fails", fails, "PASS" if res["pass"] else "FAIL", flush=True)
    print("clean_by_cat", bycat, "auth", res["jnv_authenticity_spearman"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
