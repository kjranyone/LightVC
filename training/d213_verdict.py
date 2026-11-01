"""D213 の判定(学習なし): RRTN16C の shimmer の源を RRTNC16C(c0 = 男声のみ)・RRTNE16C(包絡 = 表のみ)・RRTNB16C(両方 R なし)と比べる(results/conv_p0/d213_prereg.yaml)。
    uv run python d213_verdict.py --out ../results/conv_p0/d213.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORMS = ("RRTN16C", "RRTNC16C", "RRTNE16C", "RRTNB16C")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ax = {D: json.loads((ROOT / f"results/conv_p0/d109v2_d213{D}.json").read_text()) for D in ("dev", "dev2")}
    rep: dict = {"stats": {D: {c: {k: v["median"] for k, v in ax[D][f"d139:{c}"]["stats"].items() if k in ("shimmer", "contrast", "amline", "hf", "tilt", "lmstd")} for c in FORMS} for D in ax},
                 "axes": {D: {c: ax[D][f"d139:{c}"]["axes"] for c in FORMS} for D in ax},
                 "real_p15_p85": {D: {k: ax[D]["d139:RRTN16C"]["stats"][k]["real_p15_p85"] for k in ("shimmer", "lmstd")} for D in ax}}
    sh = {D: {c: rep["stats"][D][c]["shimmer"] for c in FORMS} for D in ax}
    dC = {D: sh[D]["RRTN16C"] - sh[D]["RRTNC16C"] for D in ax}
    dE = {D: sh[D]["RRTN16C"] - sh[D]["RRTNE16C"] for D in ax}
    rep["drop_c0"] = {D: round(v, 3) for D, v in dC.items()}
    rep["drop_env"] = {D: round(v, 3) for D, v in dE.items()}
    if all(sh[D]["RRTNB16C"] > 0.70 for D in ax):
        v = "PATH"
    elif all(dC[D] >= 0.05 and dC[D] > dE[D] for D in ax):
        v = "C0_SRC"
    elif all(dE[D] >= 0.05 and dE[D] >= dC[D] for D in ax):
        v = "ENV_SRC"
    else:
        v = "MIXED"
    rep["verdict"] = v
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
