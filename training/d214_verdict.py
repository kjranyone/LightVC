"""D214 の判定(学習なし): results/conv_p0/d214_prereg.yaml の fork(RENDERER / C0_ROW / OTHER)。
    uv run python d214_verdict.py --out ../results/conv_p0/d214.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORMS = ("RUSTB", "PYR", "PYTB", "PYX")
SC = Path("/home/kojirotanaka/lightvc_scratch_20261013")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ax = {D: json.loads((ROOT / f"results/conv_p0/d109v2_d214{D}.json").read_text()) for D in ("dev", "dev2")}
    rep: dict = {"stats": {D: {c: {k: v["median"] for k, v in ax[D][f"d214:{c}"]["stats"].items() if k in ("shimmer", "contrast", "amline", "hf", "tilt", "lmstd")} for c in FORMS} for D in ax},
                 "axes": {D: {c: ax[D][f"d214:{c}"]["axes"] for c in FORMS} for D in ax},
                 "jitter": {D: json.loads((SC / D / "d214/jitter.json").read_text())["median"] for D in ax},
                 "rust_vs_d114_maxdiff": {D: json.loads((SC / D / "d214/jitter.json").read_text())["rust_vs_d114_maxdiff"] for D in ax}}
    sh = {D: {c: rep["stats"][D][c]["shimmer"] for c in FORMS} for D in ax}
    rep["PYR_minus_RUSTB"] = {D: round(sh[D]["PYR"] - sh[D]["RUSTB"], 3) for D in ax}
    rep["PYTB_minus_PYX"] = {D: round(sh[D]["PYTB"] - sh[D]["PYX"], 3) for D in ax}
    if all(rep["PYR_minus_RUSTB"][D] >= 0.05 for D in ax):
        v = "RENDERER"
    elif all(rep["PYTB_minus_PYX"][D] >= 0.04 for D in ax):
        v = "C0_ROW"
    else:
        v = "OTHER"
    rep["verdict"] = v
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
