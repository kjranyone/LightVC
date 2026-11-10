"""段 I12(paper_snv rev24・results/snv_i12/prereg.yaml)の予測の照合(学習なし): 最終の物差しのファイルから、登録の予測ごとに値と合否を並べる。CER は中央値(登録どおり)・製品の CER は平均(test)。
入力 = results/snv_i12/{text_final.json, jvnv.json, rec_final.json, test.log, ff.json}・族の中の合成の録音の条件の基準 = results/snv_i10/rec_final.json の snv_i10_e_rec。
    cd training && uv run --no-sync python snv_diag/i12_summary.py [out=results/snv_i12/summary.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/snv_i12"
VR, R2, CO, CP, ER = "snv_i12_e_vr2", "snv_i12_e_r2", "snv_i12_combo", "snv_i10_c_pr", "snv_i10_e_rec"
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
    text, jv, rec, rec10 = jl(D / "text_final.json"), jl(D / "jvnv.json").get("conditions", {}), jl(D / "rec_final.json").get("summary", {}), jl(ROOT / "results/snv_i10/rec_final.json").get("summary", {})
    test, ff = test_rows(D / "test.log"), jl(D / "ff.json").get("systems", {})
    out: dict = {}

    def chk(name: str, val, op: str, ref) -> None:
        ok = None if val is None or ref is None else (val <= ref if op == "<=" else val >= ref)
        out[name] = {"value": None if val is None else round(float(val), 4), "op": op, "ref": None if ref is None else round(float(ref), 4), "pass": ok}

    rows = text.get("rows", [])
    med = lambda xs: float(np.median(xs)) if xs else None
    bgm_self = lambda t: med([r["cer"] for r in rows if r["sys"] == t and r["cond"] == "real_self" and r["rel"] in BGM])
    chk("bgm_self.e_r2", bgm_self(R2), "<=", 0.50)
    chk("bgm_self.e_vr2", bgm_self(VR), "<=", 0.50)
    jm = lambda t: jv.get("jv_F_k0", {}).get(t, {}).get("cer_median")
    chk("jvnv.e_vr2_F_k0", jm(VR), "<=", 0.17)
    chk("jvnv.e_r2_F_k0", jm(R2), "<=", 0.35)
    nb = lambda t: med([r["cer"] for r in rows if r["sys"] == t and r["cond"] in ("real_F_k0", "real_F_k7") and r["rel"] not in BGM])
    chk("test_nonbgm.combo", nb(CO), "<=", None if nb(CP) is None else nb(CP) - 0.03)
    for c in ("clean", "pink10", "music10", "rev"):
        a, b = rec.get(c, {}).get(f"{R2}|F", {}).get("cer_median"), rec10.get(c, {}).get(f"{ER}|F", {}).get("cer_median")
        chk(f"rec_synth.{c}", a, "<=", None if b is None else b + 0.05)
    tm = lambda t, c: text.get("conditions", {}).get(c, {}).get(t, {}).get("cer_median")
    for t in (VR, R2, CO):
        chk(f"cost.unseen_self.{t}", tm(t, "unseen_self"), "<=", 0.05)
    chk("cost.unseen_mF_k0.combo", tm(CO, "unseen_mF_k0"), "<=", None if tm(CP, "unseen_mF_k0") is None else tm(CP, "unseen_mF_k0") + 0.02)
    if CO in test and CP in test:
        chk("cost.S", test[CO]["S"], ">=", test[CP]["S"] - 0.05)
        chk("cost.g", (test[CO]["M_mean_4k"] - 0.061) / 0.574, ">=", 0.25)
        chk("product.CER_line", test[CO]["cer_k0_k7"], "<=", 0.275)
    chk("cost.FF", ff.get(CO, {}).get("src_FF"), "<=", 0.15)
    for k, v in out.items():
        print(f"{'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'n/a '}  {k}: {v['value']} {v['op']} {v['ref']}", flush=True)
    extra = {"bgm_self": {t: bgm_self(t) for t in (VR, R2, CO, CP, ER, "snv_i9_aug")}, "nonbgm": {t: nb(t) for t in (VR, R2, CO, CP, ER, "snv_i9_aug")}, "test": test, "jvnv": jv, "ff": ff}
    (ROOT / opt.get("out", "results/snv_i12/summary.json")).write_text(json.dumps({"checks": out} | extra, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
