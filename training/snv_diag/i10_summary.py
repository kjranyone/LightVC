"""段 I10(paper_snv rev21・results/snv_i10/prereg.yaml)の予測の照合(学習なし): 最終の物差しのファイルから、登録の予測ごとに値と合否を並べる。CER は中央値(登録どおり)・製品の CER は平均(test)。
入力 = results/snv_i10/{rec_final, rec_mix_c, text_final, ff_final, cside_ks_*}.json・test.log・results/snv_i9/text_final.json(対照の本物の音声)・各腕の train_log.json。族の外の条件の対照 = rec_final の snv_i9_ctl(同じ走行で測る)。
    cd training && uv run --no-sync python snv_diag/i10_summary.py [out=results/snv_i10/summary.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/snv_i10"
E, CP, CC, CO, CTL = "snv_i10_e_rec", "snv_i10_c_pr", "snv_i10_c_ctl", "snv_i10_combo", "snv_i9_ctl"


def jl(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def test_rows(p: Path) -> tuple[dict, dict]:
    rows, diff = {}, {}
    if not p.exists():
        return rows, diff
    for line in p.read_text().splitlines():
        tag, _, rest = line.partition(" ")
        if rest.startswith("{"):
            try:
                v = json.loads(rest)
            except json.JSONDecodeError:
                continue
            if tag == "diff":
                diff = v
            else:
                rows[tag] = v
    return rows, diff


def med(summary: dict, cond: str, key: str) -> float | None:
    x = summary.get(cond, {}).get(key)
    return x["cer_median"] if isinstance(x, dict) else None


def tmed(text: dict, cond: str, tag: str) -> float | None:
    x = text.get("conditions", {}).get(cond, {}).get(tag)
    return x["cer_median"] if x else None


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    rec, mix = jl(D / "rec_final.json").get("summary", {}), jl(D / "rec_mix_c.json").get("summary", {})
    base = rec
    text, text9 = jl(D / "text_final.json"), jl(ROOT / "results/snv_i9/text_final.json")
    ff = jl(D / "ff_final.json").get("systems", {})
    test, diff = test_rows(D / "test.log")
    out: dict = {}

    def chk(name: str, val, op: str, ref) -> None:
        ok = None if val is None or ref is None else (val <= ref if op == "<=" else val >= ref if op == ">=" else val < ref if op == "<" else val > ref)
        out[name] = {"value": val, "op": op, "ref": ref, "pass": ok}

    for c, lim in (("pink10", 0.15), ("rev", 0.15), ("music10", 0.35), ("clean", 0.05)):
        chk(f"M16_dev.{c}", med(rec, c, f"{E}|F"), "<=", lim)
    for c in ("babble10", "rev12", "hum15"):
        b = med(base, c, f"{CTL}|F")
        chk(f"M16_dev.{c}(対照 {b})", med(rec, c, f"{E}|F"), "<=", None if b is None else round(0.5 * b, 4))
    for c, lim in (("real_self", 0.15), ("real_M_k0", 0.30), ("real_F_k0", 0.40)):
        chk(f"M16_real.{c}", tmed(text, c, E), "<=", lim)
    chk("M16_cost.unseen_self", tmed(text, "unseen_self", E), "<=", 0.05)
    c9 = tmed(text9, "unseen_mF_k0", CTL)
    chk("M16_cost.unseen_mF_k0", tmed(text, "unseen_mF_k0", E), "<=", None if c9 is None else round(c9 + 0.02, 4))
    cc = tmed(text, "unseen_mF_k0", CC)
    chk("M15_pitch.unseen_mF_k0", tmed(text, "unseen_mF_k0", CP), "<=", 0.08)
    chk("M15_pitch.unseen_mF_k0_vs_ctl", tmed(text, "unseen_mF_k0", CP), "<=", None if cc is None else round(cc - 0.03, 4))
    ks = jl(D / f"cside_ks_{CP}.json").get("ef", {})
    chk("M15_pitch.female_k-12_F_E", ks.get("F_E_k-12_median"), "<=", 0.02)
    chk("M15_rec.music10_pv", med(mix, "music10_pv", f"{CP}|F"), "<=", 0.10)
    chk("M15_rec.rev_pv", med(mix, "rev_pv", f"{CP}|F"), "<=", 0.08)
    chk("M15_cost.unseen_self", tmed(text, "unseen_self", CP), "<=", 0.05)
    if CP in test and CC in test:
        chk("M15_cost.S", test[CP]["S"], ">=", round(test[CC]["S"] - 0.05, 4))
        chk("M15_cost.M", test[CP]["M_mean_4k"], ">=", round(test[CC]["M_mean_4k"] - 0.03, 4))
    lv = {a: (jl(ROOT / "results" / a / "train_log.json").get("evals") or [{}])[-1].get("val_l1") for a in (CP, CC)}
    chk("M15_cost.val_l1", lv[CP], "<=", None if lv[CC] is None else round(lv[CC] + 0.003, 4))
    if CO in test:
        g = (test[CO]["M_mean_4k"] - 0.061) / 0.574
        chk("product.CER_k0_k7", test[CO]["cer_k0_k7"], "<=", 0.275)
        chk("product.g", round(g, 4), ">=", 0.25)
        lo = diff.get(CO, {}).get("dM_mean_4k", {}).get("ci95_target_cluster", [None])[0]
        chk("product.dM_lower", lo, ">", 0.0)
    chk("product.FF", ff.get(CO, {}).get("src_FF"), "<=", 0.15)
    for k, v in out.items():
        print(f"{'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'n/a '}  {k}: {v['value']} {v['op']} {v['ref']}", flush=True)
    res = {"checks": out, "test": test, "diff": diff, "ff": ff}
    (ROOT / opt.get("out", "results/snv_i10/summary.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
