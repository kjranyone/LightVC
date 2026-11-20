"""段 P18 A(results/snv_p18/prereg.yaml・学習なし): C の入力の差し替え。C の構造のうち 大域の z と 鍵・値(k, v)を、自分の登録(S)と目標の参照(T)から別々に取る。
  SS = 全て自分 / TT = 全て目標(基準)/ zT_kS = z は目標・鍵と値は自分 / zS_kT = z は自分・鍵と値は目標。高さは全て k 7。
組: cv = devA の先頭 n 人 × s0・s1 / org_g0 = 生成した対照の声 / org_w_ch = 生成した低い声 + 録音の鎖 v2。目標は割り当ての 2 声。
行: mode・tgt・cer・top(出力の ECAPA + LDA が A0F の 24 声の中で最も近い声)。M = top が目標と同じ率。SS は目標を持たない(top だけ)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p18_mix.py [sets=org_g0,cv,org_w_ch] [n=64] [sys=snv_p8_wst] [out=results/snv_p18/mix_rows.json] [redo=1]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_rig as G

ROOT = G.ROOT
SETS = ("org_g0", "cv", "org_w_ch")
MODES = ("SS", "TT", "zT_kS", "zS_kT")
K = 7


def mix(Sz: dict, Sk: dict) -> dict:
    return {"z": Sz["z"], "k": Sk["k"], "v": Sk["v"], "rmask": Sk["rmask"]}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for m in MODES:
        R = [r for r in rows if r["mode"] == m and r["valid"]]
        c = np.array([r["cer"] for r in R])
        out[m] = {"n": len(R), "cer_median": round(float(np.median(c)), 4), "cer_mean": round(float(c.mean()), 4), "frac_le_0.2": round(float((c <= 0.2).mean()), 3)}
        if m == "SS":
            out[m]["M_floor"] = round(float(np.mean([r["top"] in r["tgts"] for r in R])) / 2, 3)
        else:
            out[m]["M"] = round(float(np.mean([r["top"] == r["tgt"] for r in R])), 3)
    return out


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    n, tag = int(opt.get("n", 64)), opt.get("sys", "snv_p8_wst")
    out = ROOT / opt.get("out", "results/snv_p18/mix_rows.json")
    res = json.loads(out.read_text()) if out.exists() else {"prereg": "results/snv_p18/prereg.yaml", "sys": tag, "n": n, "k": K, "rows": {}, "summary": {}}
    assert res["sys"] == tag, (res["sys"], tag)
    sets = [c for c in opt.get("sets", ",".join(SETS)).split(",") if c not in res["rows"] or opt.get("redo") == "1"]
    assert all(c in SETS for c in sets), sets
    items = {c: G.cv_items(n) if c == "cv" else G.org_items(c[4:]) for c in sets}
    rig = G.Rig(tag)
    s = rig.s
    assert s.p3, "P3 / P8 の型の系だけ"
    for c in sets:
        rows = []
        for it in items[c]:
            tgs = rig.assigned(it)
            SS = s.structure(f"self|p18|{it['spk']}|{it['src']}", it["enr"])
            rows.append(rig.row(it, s.convert(it["u"], it["enr"], SS, K), mode="SS", tgt=None, tgts=list(tgs), k=K))
            for v in tgs:
                TT = s.structure(f"t|{v}|3", rig.trefs[v])
                for mode, S in (("TT", TT), ("zT_kS", mix(TT, SS)), ("zS_kT", mix(SS, TT))):
                    rows.append(rig.row(it, s.convert(it["u"], it["enr"], S, K), mode=mode, tgt=v, k=K))
        res["rows"][c] = rows
        res["summary"][c] = summarize(rows)
        print(c, json.dumps(res["summary"][c], ensure_ascii=False), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
