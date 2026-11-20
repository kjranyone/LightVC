"""段 P25 step a 補足 1(results/snv_p25/prereg.yaml の supplement_1・学習なし): 操作者の登録の統計と有声の判定を、音量の門で選び直す探り。
条件(系は同じ・操作者の側だけを替える。門の閾 = 登録の有声のフレームの c0 の中央値 − D dB = 登録の定数):
  base = そのまま
  sgate<D> = 登録の統計(白くする平均と共分散・級ごとの平均)を取るフレームを、有声かつ c0 ≥ 閾 に絞る(流しの演奏の計算と有声の判定は元のまま)
  vgate<D> = 源と登録の有声の判定そのものを同じ門で消す(c0 < 閾 のフレームは高さ・有声を 0 に。包絡と周期性は元のまま)
組: devA の先頭 n 人 × s0・s1。自己(構造 = 自分の登録・k 0)と他の声の構造(割り当ての目標 2 声 × k 7)。CER と 1 位の声は probe_rig.Rig(large-v3・ECAPA + LDA)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p25_gate_probe.py [sys=snv_p24_c] [n=64] [conds=base,sgate12,vgate12] [out=results/snv_p25/gate_rows.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_rig as G
import snv_p1_eval as EV

ROOT = G.ROOT


def thr_of(enr: list[dict], d: float) -> float:
    c0 = np.concatenate([u["y"][0] for u in enr]) * 10.0
    v = np.concatenate([u["v"] for u in enr]) > 0.5
    return float(np.median(c0[v]) - d)


def vgated(u: dict, thr: float) -> dict:
    g = (u["y"][0] * 10.0 >= thr).astype(np.float32)
    return u | {"v": u["v"] * g, "f0": u["f0"] * g, "lf": u["lf"] * g}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag, n = opt.get("sys", "snv_p24_c"), int(opt.get("n", 64))
    conds = opt.get("conds", "base,sgate12,vgate12").split(",")
    out = ROOT / opt.get("out", "results/snv_p25/gate_rows.json")
    res = json.loads(out.read_text()) if out.exists() else {"prereg": "results/snv_p25/prereg.yaml(supplement_1)", "sys": tag, "n": n, "rows": {}}
    rig = G.Rig(tag)
    s = rig.s
    items = G.cv_items(n)
    enr_of = {it["spk"]: it["enr"] for it in items}
    for cond in conds:
        if cond in res["rows"] and opt.get("redo") != "1":
            continue
        s.scache.clear()
        kind, d = cond[:5], float(cond[5:] or 0)
        gen: dict = {}
        if kind == "sgate":
            for spk, enr in enr_of.items():
                ry, rv, _, _ = EV.struct_inputs(enr)
                s.scache[tuple(id(u) for u in enr)] = s.Pl.mean_of(ry, rv, (ry[:, 0] * 10.0 >= thr_of(enr, d)))
        if kind == "vgate":
            gen = {spk: [vgated(u, thr_of(enr, d)) for u in enr] for spk, enr in enr_of.items()}
        rows = []
        for it in items:
            enr = gen.get(it["spk"], it["enr"])
            u = vgated(it["u"], thr_of(it["enr"], d)) if kind == "vgate" else it["u"]
            w = s.convert(u, enr, s.structure(f"self|{cond}|{it['spk']}", enr), 0)
            rows.append(rig.row(it, w, mode="self", tgt=None, k=0))
            for v in rig.assigned(it):
                w = s.convert(u, enr, s.structure(f"t|{v}", rig.trefs[v]), 7)
                rows.append(rig.row(it, w, mode="cross", tgt=v, k=7))
        res["rows"][cond] = rows
        V = [r for r in rows if r["valid"]]
        print(cond, {m: {"n": len(R), "cer_median": round(float(np.median([r["cer"] for r in R])), 3), "cer_clip1": round(float(np.mean([min(r["cer"], 1.0) for r in R])), 3)} | ({"M": round(float(np.mean([r["top"] == r["tgt"] for r in R])), 3)} if m == "cross" else {})
                     for m in ("self", "cross") for R in ([r for r in V if r["mode"] == m],)}, flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
