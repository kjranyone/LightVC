"""D628: 耳 mf_v5_ear で錨 GU* に 12 / 12 で負けた差を、d109 v2 の発話ごとの統計で同じ入力の対(候補 対 錨)ごとに探す。
z = |x − 本物の中央値| / (本物の p85 − p15)(片側の統計は max(0, x − 中央値) / w)・d = z(候補) − z(錨)(正 = 錨が本物に近い = 耳と同じ向き)。規則は results/conv_p0/d628/prereg.yaml。
    uv run python d628_pair_axes.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
S = Path("/home/kojirotanaka/lightvc_scratch_20261013/mfv5")
OUT = ROOT / "results/conv_p0/d628"
CAND = {5: "RUST5Q3", 7: "RUST7Q3", 12: "RUST12"}
ONE = {"jump", "amline", "ltas_d"}


def main() -> int:
    cache = json.loads((S / "d109_cache.json").read_text())
    pool = [v for k, v in cache.items() if k.startswith("2|pool|")]
    keys = list(pool[0])
    R = {k: np.array([r[k] for r in pool], float) for k in keys}
    med = {k: float(np.nanmedian(R[k])) for k in keys}
    w = {k: float(np.subtract(*np.nanpercentile(R[k], [85, 15]))) for k in keys}

    def per(sysname: str) -> dict:
        out = {}
        for f in sorted((S / "gate" / sysname).glob(f"p*_{sysname}.wav")):
            ck = f"2|set|{f}|{f.stat().st_mtime_ns}"
            if ck in cache:
                out[f.name[:10]] = cache[ck]
        return out

    def z(k: str, x: float) -> float:
        return max(0.0, x - med[k]) / w[k] if k in ONE else abs(x - med[k]) / w[k]

    tally = json.loads((ROOT / "results/earbattery/mf_v5_ear/tally.json").read_text())["trials"]
    ear = [(v["pair"], v["k"], t) for t, v in tally.items()]
    rep = {"n_pool": len(pool), "real_median": med, "real_w": w, "stats": {}}
    data = {k: (per(CAND[k]), per(f"GU{k}")) for k in CAND}
    for k in CAND:
        print("k", k, "files cand", len(data[k][0]), "GU", len(data[k][1]), flush=True)
    for s in keys:
        r = {"ear": {}, "gate": {}}
        de = []
        for p, k, t in ear:
            c, g = data[k][0][p][s], data[k][1][p][s]
            d = z(s, c) - z(s, g)
            de.append(d)
            r["ear"][t] = {"k": k, "cand": round(c, 3), "GU": round(g, 3), "d": round(d, 3)}
        de = np.array(de)
        r["ear_npos"] = int((de > 0).sum())
        ok_gate = True
        for k in CAND:
            ps = sorted(set(data[k][0]) & set(data[k][1]))
            dd = np.array([z(s, data[k][0][p][s]) - z(s, data[k][1][p][s]) for p in ps])
            dd = dd[np.isfinite(dd)]
            fr, md = float((dd > 0).mean()), float(np.median(dd))
            cm, gm = float(np.nanmedian([data[k][0][p][s] for p in ps])), float(np.nanmedian([data[k][1][p][s] for p in ps]))
            r["gate"][k] = {"n": len(dd), "frac_pos": round(fr, 3), "median_d": round(md, 3), "cand_median": round(cm, 3), "GU_median": round(gm, 3)}
            ok_gate &= fr >= 0.75 and md >= 0.25
        r["rule_E"] = bool(r["ear_npos"] >= 11 and ok_gate)
        rep["stats"][s] = r
        g = r["gate"]
        print(f"{s:8s} real {med[s]:7.3f} w {w[s]:6.3f} | ear d>0 {r['ear_npos']:2d}/12 | " + " ".join(f"k{k}: cand {g[k]['cand_median']:7.3f} GU {g[k]['GU_median']:7.3f} pos {g[k]['frac_pos']:.2f} md {g[k]['median_d']:+.2f}" for k in CAND) + f" | E {r['rule_E']}", flush=True)
    rep["rule_E_stats"] = [s for s, r in rep["stats"].items() if r["rule_E"]]
    rep["fork"] = "some_E" if rep["rule_E_stats"] else "none_E"
    (OUT / "pair_axes.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("rule_E", rep["rule_E_stats"], "fork", rep["fork"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
