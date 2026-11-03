"""N0r の追加の線 CAT_x: N0c の全部で学習した primary の分類器(声に依らない種類の物差し)で、N0r の 6 種(混ざりを除く 432 本)を読む。事前登録 = results/snv_n0r/prereg.yaml。
    cd training && uv run --no-sync python snv_n0r_catx.py
"""
from __future__ import annotations

import json

import numpy as np

from snv_a0_measure import ROOT, load, to16
from snv_n0_measure import cat_feat, trim
from snv_n0c_gen import CATS


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    feat = lambda p: cat_feat(to16(*(lambda w: (trim(w[0]), w[1]))(load(p))))
    its = json.loads((ROOT / "results/snv_n0c/wav/items.json").read_text())
    X = np.stack([feat(ROOT / f"results/snv_n0c/wav/{i['voice']}_{i['cat']}_{i['t']}_{i['k']}.wav") for i in its])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X, [CATS.index(i["cat"]) for i in its])
    n0r = [i for i in json.loads((ROOT / "results/snv_n0r/wav/items.json").read_text()) if i["cat"] in CATS]
    Z = np.stack([feat(ROOT / f"results/snv_n0r/wav/{i['voice']}_{i['cat']}_{i['k']}.wav") for i in n0r])
    ok = clf.predict(Z) == np.array([CATS.index(i["cat"]) for i in n0r])
    acc = float(ok.mean())
    lo = acc - 1.96 * np.sqrt(acc * (1 - acc) / len(ok))
    res = {"acc": round(acc, 3), "ci95_lo": round(float(lo), 3), "n": int(len(ok)), "pass": bool(lo >= 0.80),
           "per_cat": {c: round(float(np.mean([o for o, i in zip(ok, n0r) if i["cat"] == c])), 3) for c in CATS},
           "by_sex": {s: round(float(np.mean([o for o, i in zip(ok, n0r) if i["sex"] == s])), 3) for s in ("F", "M")}}
    print("CAT_x", res, flush=True)
    (ROOT / "results/snv_n0r/n0r_catx.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
