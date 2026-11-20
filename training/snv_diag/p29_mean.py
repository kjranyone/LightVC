"""段 P29 step a の a3b(results/snv_p29/prereg.yaml・学習なし・描画なし): 粗い形の平均の不足は、源の残りか、学習の声の平均への縮みか。
組 = p29_same.py と同じ(段 D の検証の声の同じ文の 242 組・k 0)。出力の発話の平均の移動 ō − s̄ を、A = 目標の参照の平均 − 源の登録の平均 と B = 目標の参照の平均 − 学習の女声の平均 に回帰する。
ō − s̄ ≈ a·A − b·B(組 × 係数を積んだ最小二乗・帯 lo = c1〜c12・hi = c13〜c24)。1 − a = 源の残り・b = 学習の声の平均への縮み。天井の発話(目標の声が同じ文を読んだ発話)でも同じ回帰をする(物差しの較正)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p29_mean.py snv_p24_c snv_i10_c_pr [out=results/snv_p29/mean.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p29_same as S
import probe_rig as G
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = G.ROOT
BANDS = S.BANDS


def fit(rows: list[dict], key: str, lo: int, hi: int, sx: str) -> dict:
    R = [r for r in rows if r["u"].startswith(sx)]
    us = sorted({r["u"] for r in R})

    def ab(sel: list[dict]) -> np.ndarray:
        X = np.stack([np.concatenate([r["A"][lo - 1:hi - 1] for r in sel]), -np.concatenate([r["B"][lo - 1:hi - 1] for r in sel])], 1)
        return np.linalg.lstsq(X, np.concatenate([r[key][lo - 1:hi - 1] for r in sel]), rcond=None)[0]

    a, b = ab(R)
    g, bs = np.random.default_rng(0), []
    for _ in range(1000):
        pick = g.choice(us, len(us))
        bs.append(ab([r for u in pick for r in R if r["u"] == u]))
    bs = np.array(bs)
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
    return {"n": len(R), "a": [round(float(a), 3), ci(bs[:, 0])], "b": [round(float(b), 3), ci(bs[:, 1])], "corr(A, B)": round(float(np.corrcoef(np.concatenate([r["A"][lo - 1:hi - 1] for r in R]), np.concatenate([r["B"][lo - 1:hi - 1] for r in R]))[0, 1]), 3)}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tags = [a for a in sys.argv[1:] if "=" not in a]
    out = ROOT / opt.get("out", "results/snv_p29/mean.json")
    split = json.loads((ROOT / S.SPLIT).read_text())
    vm = lambda us: S.vmean(us, 1, 25)
    pop = np.mean([vm([T.load_utt(T.FEAT / f"e1/{v}/{t}.npz") for t in ("t00", "t01", "t02")]) for v in split["train"] if v.startswith("ef")], 0)
    P = S.pairs()
    voices = sorted({p["u"] for p in P} | {p["v"] for p in P})
    enr = {v: [T.load_utt(T.FEAT / f"e1/{v}/{t}.npz") for t in ("t00", "t01", "t02")] for v in voices}
    du = {(v, s): T.load_utt(T.FEAT / f"d/{v}/{s}.npz") for p in P for v, s in ((p["u"], p["su"]), (p["v"], p["sv"]))}
    base = [{"u": p["u"], "v": p["v"], "A": vm(enr[p["v"]]) - vm(enr[p["u"]]), "B": vm(enr[p["v"]]) - pop, "truth": vm([du[(p["v"], p["sv"])]]) - vm([du[(p["u"], p["su"])]])} for p in P]
    res = {"prereg": "results/snv_p29/prereg.yaml(a3b)", "pop_n": sum(v.startswith("ef") for v in split["train"]), "truth(天井の発話)": {sx: {b: fit(base, "truth", lo, hi, sx) for b, (lo, hi) in BANDS.items()} for sx in ("ef", "em")}, "systems": {}}
    print("truth", json.dumps(res["truth(天井の発話)"], ensure_ascii=False), flush=True)
    gen = EV.load_gen()
    for tag in tags:
        s_ = EV.System(tag, gen)
        rows = []
        for p, q in zip(P, base):
            s = du[(p["u"], p["su"])]
            yo = s_.convert(s, enr[p["u"]], s_.structure(f"t|{p['v']}", enr[p["v"]]), 0, with_y=True)[1][0].cpu().numpy()
            m = s["v"][:yo.shape[1]] > 0
            rows.append(q | {"out": yo[1:25, m].mean(1) - vm([s])})
        res["systems"][tag] = {sx: {b: fit(rows, "out", lo, hi, sx) for b, (lo, hi) in BANDS.items()} for sx in ("ef", "em")}
        print(tag, json.dumps(res["systems"][tag], ensure_ascii=False), flush=True)
        del s_
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
