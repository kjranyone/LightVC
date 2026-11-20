"""段 P15(results/snv_p15/prereg.yaml): p15_probe.py の行を、話者・目標・失敗の型・台帳の量・差し替えの利得に分けて results/snv_p15/analysis.json に書く(学習も測定もしない)。
all = devA の先頭 64 人の全部の源 / valid = 元の音の書き起こしが読み上げの文と合う源(text_vs_orig ≤ 0.1・補足 1)。
    cd training && uv run --no-sync python snv_diag/p15_analyze.py [rows=results/snv_p15/rows.json] [out=results/snv_p15/analysis.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ("f0_med", "f0_p5", "floor", "lvl_span", "mora_rate", "rate", "vq", "vrate", "tilt", "hum", "per0", "env_v_b3k", "env_v_b6k")
FIT = ("f0_med", "floor", "mora_rate", "copy")
VALID = 0.1
r3 = lambda x: round(float(x), 3)
med = lambda xs: r3(np.median(xs)) if len(xs) else None


def eta(rows: list[dict], key: str, clip: float | None) -> dict:
    spk, tgt = sorted({r["spk"] for r in rows}), sorted({r["tgt"] for r in rows})
    cell: dict = {}
    for r in rows:
        cell.setdefault((r["spk"], r["tgt"]), []).append(min(r[key], clip) if clip else r[key])
    y = np.array([[np.mean(cell[(s, t)]) for t in tgt] for s in spk])
    allv = np.array([x for v in cell.values() for x in v])
    g = allv.mean()
    n = len(allv) / y.size
    ss_t = ((allv - g) ** 2).sum()
    ss_s = y.shape[1] * n * ((y.mean(1) - y.mean()) ** 2).sum()
    ss_g = y.shape[0] * n * ((y.mean(0) - y.mean()) ** 2).sum()
    ss_c = n * ((y - y.mean()) ** 2).sum() - ss_s - ss_g
    return {"speaker": r3(ss_s / ss_t), "target": r3(ss_g / ss_t), "speaker_x_target": r3(ss_c / ss_t), "utterance_rest": r3(1 - (ss_s + ss_g + ss_c) / ss_t)}


def loo_r2(X: np.ndarray, y: np.ndarray) -> float:
    Xr = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])] + [np.ones(len(y))])
    yr = rankdata(y)
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        pred[i] = Xr[i] @ np.linalg.lstsq(Xr[m], yr[m], rcond=None)[0]
    return float(1 - ((yr - pred) ** 2).sum() / ((yr - yr.mean()) ** 2).sum())


def boot(d: dict, fn, n: int = 2000) -> list:
    spk = sorted(d)
    rng = np.random.default_rng(0)
    bs = [fn([x for s in rng.choice(spk, len(spk)) for x in d[s]]) for _ in range(n)]
    return [r3(np.percentile(bs, 2.5)), r3(np.percentile(bs, 97.5))]


def part(R: dict, ok: set, copy: dict, selfc: dict, led: dict, vsp: dict) -> dict:
    A: dict = {}
    b4 = [r for r in R["base4"] if r["src"] in ok]
    spk = list(dict.fromkeys(r["spk"] for r in b4))
    full = [s for s in spk if len([r for r in b4 if r["spk"] == s]) == 16]
    A["n"] = {"sources": len(ok), "speakers": len(spk), "trials": len(b4)}
    A["variance"] = {f"k{k}": {"cer": eta([r for r in b4 if r["k"] == k and r["spk"] in full], "cer", None), "cer_clip1": eta([r for r in b4 if r["k"] == k and r["spk"] in full], "cer", 1.0),
                               "m": eta([r for r in b4 if r["k"] == k and r["spk"] in full], "m", None)} for k in (0, 7)}
    A["by_target"] = {t: {f"k{k}": {"cer_median": med([r["cer"] for r in b4 if r["tgt"] == t and r["k"] == k]), "frac_le_0.2": r3(np.mean([r["cer"] <= 0.2 for r in b4 if r["tgt"] == t and r["k"] == k])),
                                    "M": r3(np.mean([r["m"] for r in b4 if r["tgt"] == t and r["k"] == k]))} for k in (0, 7)} for t in sorted({r["tgt"] for r in b4})}
    A["base4"] = {"cer_median": med([r["cer"] for r in b4]), "ci95_spk": boot({s: [r["cer"] for r in b4 if r["spk"] == s] for s in spk}, np.median), "cer_median_k0": med([r["cer"] for r in b4 if r["k"] == 0]), "cer_median_k7": med([r["cer"] for r in b4 if r["k"] == 7]),
                  "frac_le_0.2": r3(np.mean([r["cer"] <= 0.2 for r in b4])), "M_k7": r3(np.mean([r["m"] for r in b4 if r["k"] == 7])), "M_k0": r3(np.mean([r["m"] for r in b4 if r["k"] == 0]))}
    S = {s: {"cer": float(np.mean([min(r["cer"], 1.5) for r in b4 if r["spk"] == s])), "m7": float(np.mean([r["m"] for r in b4 if r["spk"] == s and r["k"] == 7])), "copy": float(np.mean([copy[x] for x in ok if x.startswith(s + "_")])),
             "self": float(np.mean([selfc[x] for x in ok if x.startswith(s + "_")]))} | {k: led[s][k] for k in LEDGER} | vsp.get(s, {}) for s in spk}
    y = np.array([S[s]["cer"] for s in spk])
    A["speaker_cer_quantiles"] = {q: r3(np.percentile(y, q)) for q in (10, 25, 50, 75, 90)}
    A["spearman_with_speaker_cer"] = {}
    for k in [k for k in S[spk[0]] if k != "cer"]:
        c = spearmanr([S[s][k] for s in spk], y)
        A["spearman_with_speaker_cer"][k] = [r3(c.statistic), r3(c.pvalue)]
    A["spearman_with_speaker_cross_minus_self"] = {k: r3(spearmanr([S[s][k] for s in spk], y - np.array([S[s]["self"] for s in spk])).statistic) for k in S[spk[0]] if k not in ("cer", "self")}
    fit = lambda ks: r3(loo_r2(np.array([[S[s][k] for k in ks] for s in spk]), y))
    A["loo_r2_rank"] = {"+".join(FIT): fit(FIT), "f0_med+floor+mora_rate": fit(FIT[:3]), "copy": fit(("copy",)), "self": fit(("self",)), "copy+self": fit(("copy", "self"))}
    f0 = np.array([S[s]["f0_med"] for s in spk])
    ter = {s: int(np.searchsorted(np.percentile(f0, [100 / 3, 200 / 3]), S[s]["f0_med"])) for s in spk}
    A["by_pitch_tertile"] = {str(t): {"f0_med_hz": r3(100 * 2 ** (np.median([S[s]["f0_med"] for s in spk if ter[s] == t]) / 12)), "cer_median": med([r["cer"] for r in b4 if ter[r["spk"]] == t]), "self": med([selfc[x] for x in ok if ter[x.rsplit("_", 1)[0]] == t])} for t in (0, 1, 2)}

    def kind(r: dict) -> str:
        if copy[r["src"]] > 0.2:
            return "copy_fail"
        if selfc[r["src"]] > 0.2:
            return "self_fail"
        return "cross_fail" if r["cer"] > 0.2 else "ok"

    tot = sum(min(r["cer"], 1.0) for r in b4)
    A["failure_types"] = {k: {"trials": r3(np.mean([kind(r) == k for r in b4])), "error_mass": r3(sum(min(r["cer"], 1.0) for r in b4 if kind(r) == k) / tot), "cer_median": med([r["cer"] for r in b4 if kind(r) == k])} for k in ("copy_fail", "self_fail", "cross_fail", "ok")}
    A["sources"] = {"copy_gt_0.2": r3(np.mean([copy[s] > 0.2 for s in ok])), "self_gt_0.2": r3(np.mean([selfc[s] > 0.2 for s in ok])), "self_median": med([selfc[s] for s in ok]), "copy_median": med([copy[s] for s in ok])}
    A["oracle_levels"] = {"cross→self(試行の CER を源の自己の CER に)": med([selfc[r["src"]] for r in b4]), "cross→copy": med([copy[r["src"]] for r in b4]),
                          "self が保たれる源(≤ 0.2)だけの cross": med([r["cer"] for r in b4 if copy[r["src"]] <= 0.2 and selfc[r["src"]] <= 0.2])}
    nref = sum(r["n_ref"] for r in b4)
    A["ops_per_ref_char"] = {k: r3(sum(r[k] for r in b4) / nref) for k in ("sub", "del", "ins")}

    base = {(r["src"], r["tgt"]): r for r in b4 if r["k"] == 7}
    rp: dict = {}

    def paired(rows: list[dict], ref: dict) -> dict:
        P = [(r, ref[(r["src"], r["tgt"])]) for r in rows if (r["src"], r["tgt"]) in ref]
        d = {s: [min(a["cer"], 1.0) - min(b["cer"], 1.0) for a, b in P if a["spk"] == s] for s in dict.fromkeys(a["spk"] for a, _ in P)}
        return {"n": len(P), "cer_median": med([a["cer"] for a, _ in P]), "ref_cer_median": med([b["cer"] for _, b in P]), "d_median": r3(np.median([a["cer"] for a, _ in P]) - np.median([b["cer"] for _, b in P])),
                "paired_diff_mean_clip1": r3(np.mean([x for v in d.values() for x in v])), "paired_diff_mean_ci95_spk": boot(d, np.mean), "frac_le_0.2": r3(np.mean([a["cer"] <= 0.2 for a, _ in P])), "ref_frac_le_0.2": r3(np.mean([b["cer"] <= 0.2 for _, b in P])),
                "M": r3(np.mean([a["m"] for a, _ in P])), "ref_M": r3(np.mean([b["m"] for _, b in P])),
                "by_pitch_tertile": {str(t): r3(np.median([a["cer"] for a, _ in P if ter[a["spk"]] == t]) - np.median([b["cer"] for a, b in P if ter[a["spk"]] == t])) for t in (0, 1, 2)}}

    for c in ("tref8", "tref14", "enr4", "enr2", "enr8s", "enr10", "src1", "wcopy"):
        if c in R:
            rp[c] = paired(R[c], base)
    if "whigh" in R and "wcopy" in R:
        rp["whigh_vs_wcopy"] = paired(R["whigh"], {(r["src"], r["tgt"]): r for r in R["wcopy"] if r["src"] in ok})
        rp["whigh_vs_base"] = paired(R["whigh"], base)
    A["replacement"] = rp
    return A


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    res = json.loads((ROOT / opt.get("rows", "results/snv_p15/rows.json")).read_text())
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())
    led = json.loads((ROOT / "results/snv_d5/ledger/cv.json").read_text())
    R = res["rows"]
    srcs = list(dict.fromkeys(r["src"] for r in R["base4"]))
    copy = {r["src"]: r["cer"] for r in dv["rows"]["copy"] if r["group"] == "devA"}
    selfc = {r["src"]: r["cer"] for r in R["self"]}
    tvo = {s: dv["ref"][s]["text_vs_orig"] for s in srcs}
    A: dict = {"prereg": "results/snv_p15/prereg.yaml", "sys": res["sys"], "summary": res["summary"], "valid_rule": f"text_vs_orig ≤ {VALID}",
               "marginal_sources": {"n": sum(v > VALID for v in tvo.values()), "copy_gt_0.2": r3(np.mean([copy[s] > 0.2 for s in srcs if tvo[s] > VALID])), "p8_median": med([r["cer"] for r in R["base4"] if tvo[r["src"]] > VALID])}}
    A["all"] = part(R, set(srcs), copy, selfc, led, res.get("vspace", {}))
    A["valid"] = part(R, {s for s in srcs if tvo[s] <= VALID}, copy, selfc, led, res.get("vspace", {}))
    fam: dict = {}
    for c in [c for c in R if c.startswith(("fam_", "org_"))]:
        X, Sf = [r for r in R[c] if r["tgt"] is not None], [r for r in R[c] if r["tgt"] is None]
        fam[c] = {"n": len(X), "cross_median": med([r["cer"] for r in X]), "ci95_spk": boot({s: [r["cer"] for r in X if r["spk"] == s] for s in {r["spk"] for r in X}}, np.median), "frac_le_0.2": r3(np.mean([r["cer"] <= 0.2 for r in X])), "M": r3(np.mean([r["m"] for r in X]))}
        if Sf:
            fam[c] |= {"self_median": med([r["cer"] for r in Sf]), "self_frac_le_0.2": r3(np.mean([r["cer"] <= 0.2 for r in Sf])), "cross_minus_self": r3(np.median([r["cer"] for r in X]) - np.median([r["cer"] for r in Sf]))}
    A["generated_voices"] = fam
    jv = {c: R[c] for c in ("jv6", "jv12", "jv24", "jv48") if c in R}
    if jv:
        b = {(r["src"], r["tgt"]): r["cer"] for r in jv.get("jv6", [])}
        A["jvnv_enroll"] = {c: {"n": len(rs), "cer_median": med([r["cer"] for r in rs]), "cer_mean": r3(np.mean([r["cer"] for r in rs])), "M": r3(np.mean([r["m"] for r in rs])), "paired_diff_mean_vs_jv6": r3(np.mean([r["cer"] - b[(r["src"], r["tgt"])] for r in rs])) if b else None,
                                    "by_spk": {s: med([r["cer"] for r in rs if r["spk"] == s]) for s in ("M1", "M2")}} for c, rs in jv.items()}
    A["jvnv_build"] = {c: {"n": len(R[c]), "cer_median": med([r["cer"] for r in R[c]]), "cer_mean": r3(np.mean([min(r["cer"], 1.0) for r in R[c]])), "frac_le_0.2": r3(np.mean([r["cer"] <= 0.2 for r in R[c]])), "M": r3(np.mean([r["m"] for r in R[c]]))}
                       for c in ("jv_wcopy", "jv_wlow", "jv_ch", "jv_wlow_ch") if c in R}
    rp = A["valid"]["replacement"]
    G = {"G_pitch": -rp["whigh_vs_wcopy"]["d_median"], "G_tref": -rp["tref14"]["d_median"], "G_enr8s": -rp["enr8s"]["d_median"]} | ({"G_enr10": -rp["enr10"]["d_median"]} if "enr10" in rp else {})
    G["G_fam"] = r3(rp["enr2"]["cer_median"] - fam["fam_g0"]["cross_median"])
    A["estimates_valid"] = G
    (ROOT / opt.get("out", "results/snv_p15/analysis.json")).write_text(json.dumps(A, ensure_ascii=False, indent=1))
    show = lambda d: json.dumps(d, ensure_ascii=False)
    for name in ("all", "valid"):
        P = A[name]
        print(f"== {name}", show(P["n"]), show(P["base4"]))
        print("  ばらつき k7", show(P["variance"]["k7"]["cer"]), "| M", show(P["variance"]["k7"]["m"]))
        print("  目標ごと", show({t: [v["k0"]["cer_median"], v["k7"]["cer_median"], v["k7"]["M"]] for t, v in P["by_target"].items()}))
        print("  話者の CER との相関", show({k: v[0] for k, v in P["spearman_with_speaker_cer"].items()}))
        print("  (他の声 − 自己)との相関", show(P["spearman_with_speaker_cross_minus_self"]))
        print("  一つ抜きの R²", show(P["loo_r2_rank"]), "| 高さの 3 分位", show(P["by_pitch_tertile"]))
        print("  失敗の型", show(P["failure_types"]))
        print("  源", show(P["sources"]), "| 上限", show(P["oracle_levels"]), "| 誤りの内訳", show(P["ops_per_ref_char"]))
        for c, v in P["replacement"].items():
            print(f"  {c:15s} 中央値 {v['cer_median']} 対 {v['ref_cer_median']}(差 {v['d_median']})対の差の平均 {v['paired_diff_mean_clip1']} {v['paired_diff_mean_ci95_spk']} ≤0.2 {v['frac_le_0.2']} 対 {v['ref_frac_le_0.2']} M {v['M']} 対 {v['ref_M']} 高さ別 {show(v['by_pitch_tertile'])}")
    print("== 物差しの外の源", show(A["marginal_sources"]))
    print("== 生成した声", *[f"\n  {c}: {show(v)}" for c, v in fam.items()])
    if jv:
        print("== JVNV の登録の量", *[f"\n  {c}: {show(v)}" for c, v in A["jvnv_enroll"].items()])
    print("== JVNV に足す", show(A["jvnv_build"]))
    print("== 見積もり(有効な源)", show(G))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
