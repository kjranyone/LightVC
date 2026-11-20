"""段 P74 step b(results/snv_p74/prereg.yaml・学習なし): p74_parts.py の行から、形ごとの平均と、形の対の差(源で対・話者で束ねた 95 % の区間)を出す。
    cd training && uv run --no-sync python snv_diag/p74_parts_read.py in=results/snv_p74/parts_cv.json [out=results/snv_p74/parts_cv_read.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p74_concord import boot

ROOT = Path(__file__).resolve().parents[2]
PAIRS = (("self7", "self0"), ("x7r3", "self0"), ("x0r3", "self0"), ("x7r3", "x0r3"), ("m0r3", "self0"), ("x0r3", "m0r3"), ("x7r1", "x7r3"), ("x7r6", "x7r3"), ("selfin0", "self0"))


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    src = ROOT / opt["in"]
    rows = json.loads(src.read_text())["rows"]
    keys = [k for k in rows[0] if k not in ("set", "spk", "src", "n_enr")]
    val = {k: np.array([float(np.mean(r[k])) for r in rows]) for k in keys}
    g = np.array([r["spk"] for r in rows])
    en = [k for k in keys if "e" in k[4:]]
    pairs = PAIRS + tuple((k, k[: k.index("e", 4)]) for k in en)
    res = {"in": opt["in"], "n": len(rows), "n_spk": len(set(g)), "mean": {k: round(float(v.mean()), 4) for k, v in val.items()}, "diff": {f"{a}-{b}": boot(val[a] - val[b], g) for a, b in pairs}}
    (ROOT / opt.get("out", str(src.with_name(src.stem + "_read.json").relative_to(ROOT)))).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
