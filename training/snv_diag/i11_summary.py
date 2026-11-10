"""段 I11(paper_snv rev22・results/snv_i11/prereg.yaml)の予測の照合(学習なし): 最終の物差しのファイルから、登録の予測ごとに値と合否を並べる。CER は中央値(登録どおり)・製品の CER は平均(test)。
入力 = results/snv_i11/{test.log, text_final.json, jvnv.json, ff.json, rec.json}。
    cd training && uv run --no-sync python snv_diag/i11_summary.py [out=results/snv_i11/summary.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/snv_i11"
CP, CV, I8V = "snv_i10_c_pr", "snv_i10_c_pr+vtl", "snv_i8_m13+vtl"
BGM = {"real/namikawa_0", "real/namikawa_1", "real/namikawa_2"}


def jl(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def test_rows(p: Path) -> dict:
    rows = {}
    if p.exists():
        for line in p.read_text().splitlines():
            tag, _, rest = line.partition(" ")
            if rest.startswith("{") and tag != "diff":
                try:
                    rows[tag] = json.loads(rest)
                except json.JSONDecodeError:
                    pass
    return rows


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    test, text, jv, ff, rec = test_rows(D / "test.log"), jl(D / "text_final.json"), jl(D / "jvnv.json").get("conditions", {}), jl(D / "ff.json").get("systems", {}), jl(D / "rec.json").get("summary", {})
    out: dict = {}

    def chk(name: str, val, op: str, ref) -> None:
        ok = None if val is None or ref is None else (val <= ref if op == "<=" else val >= ref if op == ">=" else val < ref)
        out[name] = {"value": None if val is None else round(float(val), 4), "op": op, "ref": None if ref is None else round(float(ref), 4), "pass": ok}

    jm = lambda t, c: jv.get(c, {}).get(t, {}).get("cer_median")
    chk("dev_jvnv.c_pr+vtl_F_k0", jm(CV, "jv_F_k0"), "<=", 0.18)
    chk("dev_jvnv.c_pr+vtl_vs_c_pr", jm(CV, "jv_F_k0"), "<=", None if jm(CP, "jv_F_k0") is None else jm(CP, "jv_F_k0") - 0.10)
    chk("dev_jvnv.i8+vtl_F_k0", jm(I8V, "jv_F_k0"), "<=", 0.16)
    rows = text.get("rows", [])
    nb = lambda t: [r["cer"] for r in rows if r["sys"] == t and r["cond"] in ("real_F_k0", "real_F_k7") and r["rel"] not in BGM]
    mcp, mcv = (float(np.median(nb(t))) if nb(t) else None for t in (CP, CV))
    chk("test_nonbgm.c_pr+vtl", mcv, "<=", None if mcp is None else mcp - 0.05)
    tm = lambda t, c: text.get("conditions", {}).get(c, {}).get(t, {}).get("cer_median")
    if CP in test and CV in test:
        chk("cost.M", test[CV]["M_mean_4k"], ">=", test[CP]["M_mean_4k"] - 0.03)
        chk("cost.g", (test[CV]["M_mean_4k"] - 0.061) / 0.574, ">=", 0.25)
        chk("cost.S", test[CV]["S"], ">=", test[CP]["S"] - 0.05)
        chk("product.CER_vs_c_pr", test[CV]["cer_k0_k7"], "<", test[CP]["cer_k0_k7"])
        chk("product.CER_line", test[CV]["cer_k0_k7"], "<=", 0.275)
    chk("cost.FF", ff.get(CV, {}).get("src_FF"), "<=", 0.15)
    chk("cost.unseen_self", tm(CV, "unseen_self"), "<=", 0.05)
    chk("cost.unseen_mF_k0", tm(CV, "unseen_mF_k0"), "<=", None if tm(CP, "unseen_mF_k0") is None else tm(CP, "unseen_mF_k0") + 0.02)
    for c in ("clean", "pink10", "music10", "rev"):
        a, b = (rec.get(c, {}).get(f"{t}|F", {}).get("cer_median") for t in (CV, CP))
        chk(f"cost.rec_{c}", a, "<=", None if b is None else b + 0.05)
    for k, v in out.items():
        print(f"{'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'n/a '}  {k}: {v['value']} {v['op']} {v['ref']}", flush=True)
    extra = {"nonbgm_median": {"c_pr": mcp, "c_pr+vtl": mcv}, "test": test, "text_conditions": text.get("conditions", {}), "jvnv": jv, "ff": ff}
    (ROOT / opt.get("out", "results/snv_i11/summary.json")).write_text(json.dumps({"checks": out} | extra, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
