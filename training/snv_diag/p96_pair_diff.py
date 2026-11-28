import json
import sys

import numpy as np


def rows(p: str) -> tuple[dict, dict]:
    r = json.load(open(p))
    (rs,) = r["rows"].values()
    return {(q["src"], q["tgt"], q["k"]): q for q in rs if q["group"] == (sys.argv[4] if len(sys.argv) > 4 else "devA")}, r["ref"]


def main() -> None:
    a, ref = rows(sys.argv[1])
    b, _ = rows(sys.argv[2])
    keys = [k for k in a if k in b and ref[k[0]]["text_vs_orig"] <= 0.1]
    spk = sorted({a[k]["spk"] for k in keys})
    out = {"trials": len(keys), "speakers": len(spk)}
    rng = np.random.default_rng(0)
    for m in ["cer", "m", "utmos"]:
        d = np.array([np.mean([b[k][m] - a[k][m] for k in keys if a[k]["spk"] == s]) for s in spk])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        out[m] = {"a": round(float(np.mean([a[k][m] for k in keys])), 4), "b": round(float(np.mean([b[k][m] for k in keys])), 4), "diff_b_minus_a": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if len(sys.argv) > 3:
        json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
