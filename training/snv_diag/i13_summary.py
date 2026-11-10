"""段 I13(paper_snv rev25・results/snv_i13/prereg.yaml)の予測の照合(学習なし): 最終の物差しのファイルから、登録の予測ごとに値と合否を並べる。CER は中央値・伴奏の区間は区間ごとの値と 3 区間の平均・製品の CER は平均(test)。
対照(snv_i12_e_r2・snv_i10_c_pr)は同じ走行の中で測り直した値。入力 = results/snv_i13/{text_final.json, jvnv.json, rec_final.json, test.log, ff.json}。
    cd training && uv run --no-sync python snv_diag/i13_summary.py [out=results/snv_i13/summary.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/snv_i13"
L13, L16, CO, R2, CP = "snv_i13_e_l13", "snv_i13_e_l16", "snv_i13_combo", "snv_i12_e_r2", "snv_i10_c_pr"
BGM = ("real/namikawa_0", "real/namikawa_1", "real/namikawa_2")


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
    text, jv, rec = jl(D / "text_final.json"), jl(D / "jvnv.json").get("conditions", {}), jl(D / "rec_final.json").get("summary", {})
    test, ff = test_rows(D / "test.log"), jl(D / "ff.json").get("systems", {})
    rows = text.get("rows", [])
    out: dict = {}

    def chk(name: str, val, op: str, ref) -> None:
        ok = None if val is None or ref is None else (val <= ref if op == "<=" else val < ref if op == "<" else val >= ref)
        out[name] = {"value": None if val is None else round(float(val), 4), "op": op, "ref": None if ref is None else round(float(ref), 4), "pass": ok}

    med = lambda xs: float(np.median(xs)) if xs else None
    self_of = lambda t, rel: next((r["cer"] for r in rows if r["sys"] == t and r["cond"] == "real_self" and r["rel"] == rel), None)
    bgm_mean = lambda t: float(np.mean([self_of(t, b) for b in BGM])) if all(self_of(t, b) is not None for b in BGM) else None
    nonbgm_self = lambda t: med([r["cer"] for r in rows if r["sys"] == t and r["cond"] == "real_self" and r["rel"] not in BGM])
    tm = lambda t, c: text.get("conditions", {}).get(c, {}).get(t, {}).get("cer_median")
    jm = lambda c, t: jv.get(c, {}).get(t, {}).get("cer_median")
    rm = lambda c, t: rec.get(c, {}).get(f"{t}|F", {}).get("cer_median")
    add = lambda x, d: None if x is None else x + d

    chk("bgm.namikawa_1.e_l13", self_of(L13, "real/namikawa_1"), "<=", 0.75)
    chk("bgm.mean.e_l13", bgm_mean(L13), "<=", add(bgm_mean(R2), -0.10))
    chk("bgm_dose.mean.e_l13_vs_e_l16", bgm_mean(L13), "<=", add(bgm_mean(L16), 0.05))
    for t in (L13, L16, CO):
        chk(f"content.unseen_self.{t}", tm(t, "unseen_self"), "<=", 0.05)
    chk("content.nonbgm_self.e_l13", nonbgm_self(L13), "<=", add(nonbgm_self(R2), 0.05))
    chk("content.jv_self.e_l13", jm("jv_self", L13), "<=", 0.10)
    for c in ("clean", "pink10", "music10", "rev"):
        chk(f"content.rec.{c}.e_l13", rm(c, L13), "<=", add(rm(c, R2), 0.05))
    chk("voice.jvnv_F_k0.e_l13", jm("jv_F_k0", L13), "<=", add(jm("jv_F_k0", R2), 0.05))
    chk("voice.unseen_mF_k0.e_l13", tm(L13, "unseen_mF_k0"), "<=", add(tm(R2, "unseen_mF_k0"), 0.03))
    if CO in test and CP in test:
        chk("product.CER_vs_c_pr", test[CO]["cer_k0_k7"], "<", test[CP]["cer_k0_k7"])
        chk("product.CER_line", test[CO]["cer_k0_k7"], "<=", 0.275)
        chk("cost.S", test[CO]["S"], ">=", test[CP]["S"] - 0.05)
        chk("cost.g", (test[CO]["M_mean_4k"] - 0.061) / 0.574, ">=", 0.25)
    chk("cost.FF", ff.get(CO, {}).get("src_FF"), "<=", 0.15)
    chk("alt_K16.bgm.mean.e_l16", bgm_mean(L16), "<=", add(bgm_mean(R2), -0.10))
    chk("alt_K16.content.nonbgm_self.e_l16", nonbgm_self(L16), "<=", add(nonbgm_self(R2), 0.05))
    chk("alt_K16.content.jv_self.e_l16", jm("jv_self", L16), "<=", 0.10)
    for k, v in out.items():
        print(f"{'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'n/a '}  {k}: {v['value']} {v['op']} {v['ref']}", flush=True)
    systems = (L13, L16, CO, R2, CP)
    extra = {"real_self": {t: {r["rel"].replace("real/", ""): round(r["cer"], 3) for r in rows if r["sys"] == t and r["cond"] == "real_self"} for t in systems},
             "bgm_mean": {t: bgm_mean(t) for t in systems}, "nonbgm_self": {t: nonbgm_self(t) for t in systems},
             "conditions": text.get("conditions", {}), "jvnv": jv, "rec": rec, "test": test, "ff": ff}
    for t in systems:
        print(t, "bgm_mean", None if extra["bgm_mean"][t] is None else round(extra["bgm_mean"][t], 3), extra["real_self"][t], flush=True)
    (ROOT / opt.get("out", "results/snv_i13/summary.json")).write_text(json.dumps({"checks": out} | extra, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
