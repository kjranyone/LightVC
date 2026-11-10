"""段 I15(paper_snv rev30 §5 M21・results/snv_i15/prereg.yaml)の予測の照合(学習なし): 最終の物差しのファイルから、登録の予測ごとに値と合否を並べる。CER は中央値・伴奏の区間は区間ごとの値と 3 区間の平均・製品は平均。
系 = c_pr(snv_i10_c_pr)・M21 の前段 D つき(snv_i10_c_pr+sep:snv_i15_sep)・段 I14 の前段 D つき(snv_i10_c_pr+sep:snv_i14_sep・対照)。対照は同じ走行の中で測り直した値。
入力 = results/snv_i15/{post_bgwm.json, post_mask.json, real_probe.json, bgm_dev.json, text_final.json, jvnv.json, test.log, ff.json, window.json}・results/snv_i15_sep/train_log.json。
    cd training && uv run --no-sync python snv_diag/i15_summary.py [out=results/snv_i15/summary.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from i14_summary import jl, test_rows

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/snv_i15"
CP, SP, S14 = "snv_i10_c_pr", "snv_i10_c_pr+sep:snv_i15_sep", "snv_i10_c_pr+sep:snv_i14_sep"
BGM = ("real/namikawa_0", "real/namikawa_1", "real/namikawa_2")


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    dev, text, jv = jl(D / "bgm_dev.json").get("summary", {}), jl(D / "text_final.json"), jl(D / "jvnv.json").get("conditions", {})
    test, ff, win = test_rows(D / "test.log"), jl(D / "ff.json").get("systems", {}), jl(D / "window.json")
    bgwm, mask, rp = jl(D / "post_bgwm.json").get("result", {}), jl(D / "post_mask.json").get("result", {}), jl(D / "real_probe.json").get("summary", {})
    tlog = jl(ROOT / "results/snv_i15_sep/train_log.json")
    rows = text.get("rows", [])
    out: dict = {}

    def chk(name: str, val, op: str, ref) -> None:
        ok = None if val is None or ref is None else (val <= ref if op == "<=" else val < ref if op == "<" else val >= ref)
        out[name] = {"value": None if val is None else round(float(val), 4), "op": op, "ref": None if ref is None else round(float(ref), 4), "pass": ok}

    dm = lambda c, s, k: dev.get(c, {}).get(f"{s}|{k}", {}).get("cer_median")
    pad = lambda c, s, k: dev.get(c, {}).get(f"{s}|pad", {}).get(k)
    add = lambda x, d: None if x is None else x + d
    med = lambda xs: float(np.median(xs)) if xs else None
    self_of = lambda t, rel: next((r["cer"] for r in rows if r["sys"] == t and r["cond"] == "real_self" and r["rel"] == rel), None)
    bgm_mean = lambda t: float(np.mean([self_of(t, b) for b in BGM])) if all(self_of(t, b) is not None for b in BGM) else None
    nonbgm = lambda t: med([r["cer"] for r in rows if r["sys"] == t and r["cond"] == "real_self" and r["rel"] not in BGM])
    jm = lambda c, t: jv.get(c, {}).get(t, {}).get("cer_median")
    wseg = lambda t, seg: win.get("segments", {}).get(seg, {}).get(t, {}).get("cer_win")
    dv = lambda kind: None if kind not in bgwm else abs(bgwm[kind]["bg_wm"]["vfrac"] - bgwm[kind]["bg"]["vfrac"])

    chk("shortcut.wm_music_dvfrac", dv("music"), "<=", 0.005)
    chk("shortcut.wm_noise_dvfrac", dv("noise"), "<=", 0.005)
    chk("shortcut.mask40_miss", mask.get("snr40", {}).get("miss"), "<=", 0.03)
    chk("shortcut.mask40_c0", mask.get("snr40", {}).get("y_c0"), "<=", 0.30)
    chk("real.vctk_vacc", rp.get("vctk", {}).get("vacc"), ">=", 0.85)
    chk("real.vctk_miss", rp.get("vctk", {}).get("miss"), "<=", 0.12)
    chk("real.jvnv_vacc", rp.get("jvnv", {}).get("vacc"), ">=", 0.93)
    chk("noise.clap10.self", dm("clap10", SP, "self"), "<=", 0.06)
    chk("noise.clap10.F", dm("clap10", SP, "F"), "<=", 0.12)
    chk("noise.op5.self", dm("op5", SP, "self"), "<=", 0.05)
    chk("noise.op5.F", dm("op5", SP, "F"), "<=", 0.10)
    chk("noise.claplead15.pad_db", pad("claplead15", SP, "pad_db_median"), "<=", -25.0)
    chk("noise.claplead15.pad_vfrac", pad("claplead15", SP, "pad_vfrac_median"), "<=", 0.03)
    chk("noise.mus0.self", dm("mus0", SP, "self"), "<=", 0.30)
    chk("noise.lead0.self", dm("lead0", SP, "self"), "<=", 0.30)
    chk("cost.dev_clean.self", dm("clean", SP, "self"), "<=", 0.02)
    chk("cost.jvnv.self", jm("jv_self", SP), "<=", add(jm("jv_self", CP), 0.03))
    chk("cost.jvnv.F_k0", jm("jv_F_k0", SP), "<=", add(jm("jv_F_k0", CP), 0.05))
    chk("cost.test_nonbgm_self", nonbgm(SP), "<=", add(nonbgm(CP), 0.05))
    if CP in test and SP in test:
        chk("cost.S", test[SP]["S"], ">=", test[CP]["S"] - 0.05)
        chk("cost.M", test[SP]["M_mean_4k"], ">=", test[CP]["M_mean_4k"] - 0.03)
    chk("cost.FF", ff.get(SP, {}).get("src_FF"), "<=", 0.15)
    last = tlog[-1]["val"] if tlog else {}
    chk("cost.pitch_cents_clean", (last.get("clean") or {}).get("cents"), "<=", 25.0)
    chk("test.bgm.window.namikawa_1", wseg(SP, "namikawa_1"), "<=", 0.50)
    chk("test.bgm.mean", bgm_mean(SP), "<=", add(bgm_mean(CP), -0.20))
    if CP in test and SP in test:
        chk("product.CER_vs_c_pr", test[SP]["cer_k0_k7"], "<", test[CP]["cer_k0_k7"])
        chk("product.CER_line", test[SP]["cer_k0_k7"], "<=", 0.275)
    for k, v in out.items():
        print(f"{'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'n/a '}  {k}: {v['value']} {v['op']} {v['ref']}", flush=True)
    for t in (CP, SP, S14, "dcopy"):
        print(t, {c: (dm(c, t, "self"), dm(c, t, "F"), pad(c, t, "pad_db_median")) for c in dev}, flush=True)
    extra = {"dev": dev, "real_self": {t: {r["rel"].replace("real/", ""): round(r["cer"], 3) for r in rows if r["sys"] == t and r["cond"] == "real_self"} for t in (CP, SP)},
             "bgm_mean": {t: bgm_mean(t) for t in (CP, SP)}, "nonbgm_self": {t: nonbgm(t) for t in (CP, SP)}, "window_mean": win.get("mean", {}),
             "conditions": text.get("conditions", {}), "jvnv": jv, "test": test, "ff": ff, "val_last": last, "bgwm": bgwm, "mask": mask, "real_probe": rp}
    (ROOT / opt.get("out", "results/snv_i15/summary.json")).write_text(json.dumps({"checks": out} | extra, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
