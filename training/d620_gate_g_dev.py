"""D620(推論だけ): 関門 G(gate_mf_v4 の G_axes と同じ: d109 v2 の 4 軸・受け継ぎ = 入力 SRC の中央値が破った境界と同じ側か境界から幅の 0.1 以内 かつ |候補 − 入力| ≤ 幅の 0.25・
vfrac は d567 の d109s_out / d109s_in = オーナー判断 10-24)+ オーナー判断 10-26(k5・k7 の ltas_d は記録だけ)を開発の組(dev2・dev3)で再現する。規則は results/conv_p0/d620/prereg.yaml。
    uv run python d620_gate_g_dev.py --vers RY58G RY58GQ6 RY58GQ3 RY58H4G --out ../results/conv_p0/d620/gate_g_dev.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_REC: set = {"ltas_d"}
REC_POP: list = []
AX = {"f0": ["jump", "range", "micro", "vfrac"], "gabi": ["contrast", "amline", "shimmer"], "muffled": ["hf", "tilt", "lmstd"], "target": ["ltas_d", "dtilt", "dhf"]}


def judge(c: dict, s: dict, vf: dict, k: int, refk: dict | None = None) -> dict:
    res = {}
    for ax, ks in AX.items():
        defects, rec = [], []
        for st in ks:
            x = c["stats"].get(st)
            if x is None:
                continue
            lo, hi = x["real_p15_p85"]
            w = hi - lo
            cm, sm = x["median"], s["stats"][st]["median"]
            if st == "vfrac":
                cm, sm = vf["d109s_out"], vf["d109s_in"]
                ok = lo <= cm <= hi and x["out_frac"] <= 0.15
            elif st == "shimmer" and refk is not None and k in refk:
                ok = cm <= refk[k]["stats"]["shimmer"]["median"] + 0.25 * w
                if not ok:
                    defects.append({"stat": st, "cand": round(cm, 3), "refk5": refk[k]["stats"]["shimmer"]["median"], "rule": f"REFK{k} + 0.25w", "inherited": False})
                continue
            else:
                ok = x["pass"]
            if ok:
                continue
            bound = hi if cm > hi else lo
            side = (sm - bound) * (cm - bound) > 0 or abs(sm - bound) <= 0.1 * w
            inh = side and abs(cm - sm) <= 0.25 * w
            d = {"stat": st, "cand": round(cm, 3), "src": round(sm, 3), "p15_p85": [lo, hi], "out_frac": x["out_frac"], "inherited": bool(inh)}
            (rec if (k in (5, 7) and (st in TARGET_REC)) else defects).append(d)
        res[ax] = {"pass": all(d_["inherited"] for d_ in defects), "failing_stats": defects, "record_only": rec}
    res["n_fail_axes"] = sum(not v["pass"] for v in res.values() if isinstance(v, dict))
    res["record_population"] = {st: c["stats"][st]["pass"] for st in REC_POP if st in c["stats"]}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vers", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d109_tag", default="d620_d109")
    ap.add_argument("--vfrac_dir", default="d620")
    ap.add_argument("--k5_shimmer_refk", default="", help="オーナー判断 10-27: k5 の shimmer を同じ高さの参照 REFK5 との差で(REFK5 の d109 の行がある tag の接頭辞・例 d624_d109)")
    ap.add_argument("--target_rec_all", action="store_true", help="オーナー判断 10-27: k5・k7 の目標との差の軸(ltas_d・dtilt・dhf)を全部記録だけ")
    ap.add_argument("--hf_tilt_target", action="store_true", help="オーナー判断 10-29: こもりの軸の hf / tilt を目標本人との差 dhf / dtilt の旗に置き換え(hf / tilt は記録)・目標との差の軸は ltas_d だけ")
    ap.add_argument("--k7_shimmer_refk", default="", help="オーナー判断 10-29: k7 の shimmer を同じ高さの参照 REFK7 との差で(REFK7 の d109 の行がある tag の接頭辞)")
    a = ap.parse_args()
    if a.target_rec_all and not a.hf_tilt_target:
        TARGET_REC.update({"dtilt", "dhf"})
    if a.hf_tilt_target:
        AX["muffled"] = ["dhf", "dtilt", "lmstd"]
        AX["target"] = ["ltas_d"]
        REC_POP.extend(["hf", "tilt"])
    out: dict = {}
    for s in ("dev2", "dev3"):
        d = json.loads((ROOT / f"results/conv_p0/{a.d109_tag}_{s}.json").read_text())
        src = d["d56:SRC"]
        for v in a.vers:
            for k in (5, 7, 12):
                nm = f"OURS{k}{v}"
                c = d[f"d570/{nm}:{nm}"]
                vf = json.loads((ROOT / f"results/conv_p0/{a.vfrac_dir}/vfrac_{nm}.json").read_text())[s]["loose_vs_strict_median"]
                refk = {}
                if a.k5_shimmer_refk:
                    rj = json.loads((ROOT / f"results/conv_p0/{a.k5_shimmer_refk}_{s}.json").read_text())
                    refk[5] = rj["d570/REFK5:REFK5"]
                if a.k7_shimmer_refk:
                    rj = json.loads((ROOT / f"results/conv_p0/{a.k7_shimmer_refk}_{s}.json").read_text())
                    refk[7] = rj["d570/REFK7:REFK7"]
                refk = refk or None
                r = judge(c, src, vf, k, refk)
                out.setdefault(v, {})[f"{s}|k{k}"] = r
                print(s, nm, "fail axes", r["n_fail_axes"], {ax: ("P" if r[ax]["pass"] else "F") for ax in AX}, [f"{x['stat']}({x['cand']}/{x.get('src', x.get('refk5'))})" for ax in AX for x in r[ax]["failing_stats"] if not x["inherited"]], flush=True)
    summ = {v: sum(r["n_fail_axes"] for r in out[v].values()) for v in a.vers}
    print("total failing axes", summ, flush=True)
    out["_total"] = summ
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
