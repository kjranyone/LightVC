"""D194(学習なし・d192 / d193 の dcorr の天井): 同じ話者・同じ文・同じ手本のクローンを乱数だけ変えてもう 1 本作り(名前 = z + 元の名前・gen_realx_clone.py --premade)、
2 本の目標の残差(目標 − 表)の向きの相関 ρ を測る。目標 = 期待値 + 乱数の揺れ とすると、期待値を完全に当てる予測器の corr(予測 − 表, 目標 − 表) の上限 ≈ √ρ。
2 本目は 1 本目の時間へ C1 の事後の DTW で写す(train_rr の交差ペアと同じ整列器)。表は各本の時間で train_rr の Pairs と同じ作り方。
    uv run python d194_clone_ceiling.py --make        # data/realx/items_clone_ceiling.json(items_clone_multi の未知 120 項目)
    uv run python d194_clone_ceiling.py --out ../results/conv_p0/d194.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results/conv_p0/d194.json"))
    a = ap.parse_args()
    cm = json.loads((ROOT / "data/realx/items_clone_multi.json").read_text())
    base = [it for it in cm if it["group"] == "unseen"]
    if a.make:
        out = [it | {"clone": "z" + it["clone"], "group": "ceiling"} for it in base]
        (ROOT / "data/realx/items_clone_ceiling.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print("items", len(out))
        return 0
    import librosa
    RRm.CFG["align"] = "c1"
    st = RRm.Store()
    rows = []
    for k, it in enumerate(base):
        it2 = it | {"clone": "z" + it["clone"]}
        if ("realx_clone", "clone", it2["clone"]) not in st.f or ("realx_clone", "clone", it["clone"]) not in st.f:
            continue
        b1 = RRm.Pairs(st, [], [], 1, 0.0, 7 + k, fixed=True, clonex=[it], p_clonex=1.0)[0]
        b2 = RRm.Pairs(st, [], [], 1, 0.0, 7 + k, fixed=True, clonex=[it2], p_clonex=1.0)[0]
        f1, f2 = st.f[("realx_clone", "clone", it["clone"])], st.f[("realx_clone", "clone", it2["clone"])]
        _, wp = librosa.sequence.dtw(X=np.sqrt(RRm.dense(f1["pk"], f1["pv"])), Y=np.sqrt(RRm.dense(f2["pk"], f2["pv"])), metric="euclidean")
        j = np.zeros(f1["pk"].shape[0], int)
        for i_, j_ in wp[::-1]:
            j[i_] = j_
        n1, n2 = b1["y"].shape[1], b2["y"].shape[1]
        j = np.minimum(j[:n1], n2 - 1)
        r1 = b1["y"] - b1["tab"]
        r2 = (b2["y"] - b2["tab"])[:, j]
        v = b1["yv"] & b2["yv"][j]
        if v.sum() < 50:
            continue
        rho = float(np.corrcoef(r1[:, v].ravel(), r2[:, v].ravel())[0, 1])
        l1 = float(np.abs(b1["y"][:, v] - b2["y"][:, j][:, v]).mean())
        l1t = float(np.abs(b1["y"][:, v] - b1["tab"][:, v]).mean())
        rows.append((it["target"], rho, l1, l1t))
    spk = np.array([r[0] for r in rows]); X = np.array([r[1:] for r in rows])
    us = sorted(set(spk)); idx = {u: np.flatnonzero(spk == u) for u in us}
    rng = np.random.default_rng(0)
    bs = np.array([X[np.concatenate([idx[s] for s in rng.choice(us, len(us))])].mean(0) for _ in range(5000)])
    ci = np.percentile(bs, [2.5, 97.5], axis=0).round(4).T.tolist()
    rho = float(X[:, 0].mean())
    rep = {"n": len(rows), "speakers": len(us), "rho_two_seeds": round(rho, 4), "rho_ci95": ci[0], "sqrt_rho_ceiling": round(float(np.sqrt(max(rho, 0))), 4),
           "l1_between_seeds": round(float(X[:, 1].mean()), 4), "l1_ci95": ci[1], "l1_clone_vs_table": round(float(X[:, 2].mean()), 4),
           "note": "d192(未知・同じ 120 項目): dcorr 正しい記憶 rr12c 0.430・差し替え 0.376・既知の保留(別の項目)0.507。L1 rr12c 1.276・表 1.387"}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
