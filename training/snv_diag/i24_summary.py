"""段 I24(学習なし・paper_snv rev39 §5 M27 + 開閉の持続・results/snv_i24/prereg.yaml)の予測の照合: 床を操作者の録音に数える比の重みの、本物の声の恒等さ・きれいな声の代価・伴奏の利得・製品の CER を登録の予測ごとに並べる。CER は中央値・製品は平均。
系 = c_pr(snv_i10_c_pr)・前段 v3 + 持続の門(200 ms で開き 500 ms で閉じる・開いたフレームだけ段 I21 の重みと重みで濾した YIN)+ c_pr(snv_i10_c_pr+mask:snv_i24_gate_hyst)。対照は同じ走行の中で測り直した値。手拍子の条件(clap.json)は記録だけ。
入力 = results/snv_i24/{real_probe.json, bgm_heldout.json, clap.json, text_final.json, jvnv.json, test.log, ff.json, window.json}。
    cd training && uv run --no-sync python snv_diag/i24_summary.py [out=results/snv_i24/summary.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from i14_summary import jl, test_rows

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/snv_i24"
CP, SP = "snv_i10_c_pr", "snv_i10_c_pr+mask:snv_i24_gate_hyst"
BGM = ("real/namikawa_0", "real/namikawa_1", "real/namikawa_2")


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    dev, clap, text, jv = jl(D / "bgm_heldout.json").get("summary", {}), jl(D / "clap.json").get("summary", {}), jl(D / "text_final.json"), jl(D / "jvnv.json").get("conditions", {})
    test, ff = test_rows(D / "test.log"), jl(D / "ff.json").get("systems", {})
    rp, win = jl(D / "real_probe.json").get("summary", {}), jl(D / "window.json")
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
    bgm = lambda t: (lambda xs: float(np.mean(xs)) if len(xs) == 3 else None)([r["cer"] for r in rows if r["sys"] == t and r["cond"] == "real_self" and r["rel"] in BGM])
    tm = lambda t, c: text.get("conditions", {}).get(c, {}).get(t, {}).get("cer_median")
    jm = lambda c, t: jv.get(c, {}).get(t, {}).get("cer_median")

    for g in ("jvnv", "real_nonbgm", "vctk"):
        chk(f"identity.{g}.y_l1", rp.get(g, {}).get("y_l1"), "<=", 0.01)
    chk("bgm.mus10.F", dm("mus10", SP, "F"), "<=", add(dm("mus10", CP, "F"), -0.25))
    chk("bgm.mus0.self", dm("mus0", SP, "self"), "<=", add(dm("mus0", CP, "self"), -0.30))
    chk("bgm.lead0.pad_db", pad("lead0", SP, "pad_db_median"), "<=", add(pad("lead0", CP, "pad_db_median"), -10.0))
    chk("bgm.lead0.F", dm("lead0", SP, "F"), "<=", add(dm("lead0", CP, "F"), -0.10))
    chk("bgm.test_bgm_self_mean", bgm(SP), "<=", add(bgm(CP), -0.15))
    chk("cost.dev_clean.self", dm("clean", SP, "self"), "<=", 0.02)
    chk("cost.dev_clean.F", dm("clean", SP, "F"), "<=", add(dm("clean", CP, "F"), 0.02))
    chk("cost.irodori.unseen_self", tm(SP, "unseen_self"), "<=", 0.02)
    chk("cost.irodori.unseen_mF_k0", tm(SP, "unseen_mF_k0"), "<=", add(tm(CP, "unseen_mF_k0"), 0.02))
    chk("cost.jvnv.self", jm("jv_self", SP), "<=", add(jm("jv_self", CP), 0.02))
    chk("cost.jvnv.F_k0", jm("jv_F_k0", SP), "<=", add(jm("jv_F_k0", CP), 0.04))
    chk("cost.test_nonbgm_self", nonbgm(SP), "<=", add(nonbgm(CP), 0.02))
    if CP in test and SP in test:
        chk("cost.S", test[SP]["S"], ">=", test[CP]["S"] - 0.03)
        chk("cost.M", test[SP]["M_mean_4k"], ">=", test[CP]["M_mean_4k"] - 0.03)
        chk("product.gain", test[SP]["cer_k0_k7"], "<=", test[CP]["cer_k0_k7"] - 0.05)
        chk("product.CER_line", test[SP]["cer_k0_k7"], "<=", 0.275)
    chk("cost.FF", ff.get(SP, {}).get("src_FF"), "<=", 0.15)
    for k, v in out.items():
        print(f"{'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'n/a '}  {k}: {v['value']} {v['op']} {v['ref']}", flush=True)
    for t in (CP, SP):
        print(t, {c: (dm(c, t, "self"), dm(c, t, "F"), pad(c, t, "pad_db_median")) for c in dev}, flush=True)
        print(" 手拍子(記録)", {c: (clap.get(c, {}).get(f"{t}|self", {}).get("cer_median"), clap.get(c, {}).get(f"{t}|F", {}).get("cer_median"), clap.get(c, {}).get(f"{t}|pad", {}).get("pad_db_median")) for c in clap}, flush=True)
        print(" 試験の伴奏の区間", {r["rel"]: round(r["cer"], 3) for r in rows if r["sys"] == t and r["cond"] == "real_self" and r["rel"] in BGM}, flush=True)
    extra = {"real_probe": rp, "window_mean": win.get("mean", {}), "dev": dev, "clap": clap, "real_self": {t: {r["rel"].replace("real/", ""): round(r["cer"], 3) for r in rows if r["sys"] == t and r["cond"] == "real_self"} for t in (CP, SP)},
             "nonbgm_self": {t: nonbgm(t) for t in (CP, SP)}, "bgm_self_mean": {t: bgm(t) for t in (CP, SP)}, "conditions": text.get("conditions", {}), "jvnv": jv, "test": test, "ff": ff}
    (ROOT / opt.get("out", "results/snv_i24/summary.json")).write_text(json.dumps({"checks": out} | extra, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
