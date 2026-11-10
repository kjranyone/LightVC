"""段 I16(paper_snv rev31 §5 M22・results/snv_i16/prereg.yaml)の予測の照合(学習なし): 最終の物差しのファイルから、登録の予測ごとに値と合否を並べる。CER は中央値・製品は平均。
系 = c_pr(snv_i10_c_pr)・M22 の E + M15 の C(snv_i16_combo)。対照は同じ走行の中で測り直した値。音素の無い発声のずれ(nv_probe)は記録だけ。
入力 = results/snv_i16/{bgm_dev.json, nv_probe.json, text_final.json, jvnv.json, test.log, ff.json}。
    cd training && uv run --no-sync python snv_diag/i16_summary.py [out=results/snv_i16/summary.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from i14_summary import jl, test_rows

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/snv_i16"
CP, SP = "snv_i10_c_pr", "snv_i16_combo"
BGM = ("real/namikawa_0", "real/namikawa_1", "real/namikawa_2")


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    dev, text, jv = jl(D / "bgm_dev.json").get("summary", {}), jl(D / "text_final.json"), jl(D / "jvnv.json").get("conditions", {})
    test, ff, nv = test_rows(D / "test.log"), jl(D / "ff.json").get("systems", {}), jl(D / "nv_probe.json").get("summary", {})
    rows = text.get("rows", [])
    out: dict = {}

    def chk(name: str, val, op: str, ref) -> None:
        ok = None if val is None or ref is None else (val <= ref if op == "<=" else val < ref if op == "<" else val >= ref)
        out[name] = {"value": None if val is None else round(float(val), 4), "op": op, "ref": None if ref is None else round(float(ref), 4), "pass": ok}

    dm = lambda c, s, k: dev.get(c, {}).get(f"{s}|{k}", {}).get("cer_median")
    pad = lambda c, s, k: dev.get(c, {}).get(f"{s}|pad", {}).get(k)
    add = lambda x, d: None if x is None else x + d
    med = lambda xs: float(np.median(xs)) if xs else None
    nonbgm = lambda t: med([r["cer"] for r in rows if r["sys"] == t and r["cond"] == "real_self" and r["rel"] not in BGM])
    tm = lambda t, c: text.get("conditions", {}).get(c, {}).get(t, {}).get("cer_median")
    jm = lambda c, t: jv.get(c, {}).get(t, {}).get("cer_median")

    chk("noise.clap0.self", dm("clap0", SP, "self"), "<=", 0.03)
    chk("noise.clap10.self", dm("clap10", SP, "self"), "<=", 0.05)
    chk("noise.clap10.F", dm("clap10", SP, "F"), "<=", 0.12)
    chk("noise.op5.self", dm("op5", SP, "self"), "<=", 0.04)
    chk("noise.op5.F", dm("op5", SP, "F"), "<=", 0.10)
    chk("noise.claplead15.pad_db", pad("claplead15", SP, "pad_db_median"), "<=", -25.0)
    chk("noise.claplead15.pad_vfrac", pad("claplead15", SP, "pad_vfrac_median"), "<=", 0.03)
    chk("cost.dev_clean.self", dm("clean", SP, "self"), "<=", 0.02)
    chk("cost.dev_clean.F", dm("clean", SP, "F"), "<=", add(dm("clean", CP, "F"), 0.03))
    chk("cost.dev_mus10.self", dm("mus10", SP, "self"), "<=", add(dm("mus10", CP, "self"), 0.05))
    chk("cost.irodori.unseen_self", tm(SP, "unseen_self"), "<=", 0.02)
    chk("cost.irodori.unseen_mF_k0", tm(SP, "unseen_mF_k0"), "<=", add(tm(CP, "unseen_mF_k0"), 0.03))
    chk("cost.jvnv.self", jm("jv_self", SP), "<=", add(jm("jv_self", CP), 0.03))
    chk("cost.jvnv.F_k0", jm("jv_F_k0", SP), "<=", add(jm("jv_F_k0", CP), 0.05))
    chk("cost.test_nonbgm_self", nonbgm(SP), "<=", add(nonbgm(CP), 0.05))
    if CP in test and SP in test:
        chk("cost.S", test[SP]["S"], ">=", test[CP]["S"] - 0.05)
        chk("cost.M", test[SP]["M_mean_4k"], ">=", test[CP]["M_mean_4k"] - 0.03)
        chk("product.no_regression", test[SP]["cer_k0_k7"], "<=", test[CP]["cer_k0_k7"] + 0.03)
        chk("product.CER_line", test[SP]["cer_k0_k7"], "<=", 0.275)
    chk("cost.FF", ff.get(SP, {}).get("src_FF"), "<=", 0.15)
    for k, v in out.items():
        print(f"{'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'n/a '}  {k}: {v['value']} {v['op']} {v['ref']}", flush=True)
    for t in (CP, SP):
        print(t, {c: (dm(c, t, "self"), dm(c, t, "F"), pad(c, t, "pad_db_median")) for c in dev}, flush=True)
    print("nv(記録)", json.dumps(nv, ensure_ascii=False), flush=True)
    extra = {"dev": dev, "nv": nv, "real_self": {t: {r["rel"].replace("real/", ""): round(r["cer"], 3) for r in rows if r["sys"] == t and r["cond"] == "real_self"} for t in (CP, SP)},
             "nonbgm_self": {t: nonbgm(t) for t in (CP, SP)}, "conditions": text.get("conditions", {}), "jvnv": jv, "test": test, "ff": ff}
    (ROOT / opt.get("out", "results/snv_i16/summary.json")).write_text(json.dumps({"checks": out} | extra, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
