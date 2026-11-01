"""D196(学習なし・D194 の天井の分解): 未知 120 項目(話者 A・文 s)ごとに、同じ文を別の話者 B の手本で作ったクローン(fp32・40 段の組だけ)を最大 3 本とり、
B の残差(B の表に対する)を A の時間へ C1 の事後の DTW で写して ρ_x = corr(r_A, r_B) を測る。ρ_same(D194)との差が話者に固有の再現する分の目安。
事前登録 = results/conv_p0/d196_prereg.yaml
    uv run python d196_clone_split.py --out ../results/conv_p0/d196.json
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


def warp(fa: dict, fb: dict) -> np.ndarray:
    import librosa
    _, wp = librosa.sequence.dtw(X=np.sqrt(RRm.dense(fa["pk"], fa["pv"])), Y=np.sqrt(RRm.dense(fb["pk"], fb["pv"])), metric="euclidean")
    j = np.zeros(fa["pk"].shape[0], int)
    for i_, j_ in wp[::-1]:
        j[i_] = j_
    return j


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/conv_p0/d196.json"))
    ap.add_argument("--per", type=int, default=3)
    a = ap.parse_args()
    RRm.CFG["align"] = "c1"
    st = RRm.Store()
    D = ROOT / "data/realx"
    cm = json.loads((D / "items_clone_multi.json").read_text())
    pool = json.loads((D / "items_clone.json").read_text()) + cm + [it for it in json.loads((D / "items_clone_scale.json").read_text()) if it["group"] == "unseen2"]
    has = lambda it: ("realx_clone", "clone", it["clone"]) in st.f and ("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC") in st.f
    pool = [it for it in pool if has(it)]
    base = [it for it in cm if it["group"] == "unseen" and has(it)]
    rows = []
    for k, it in enumerate(base):
        bs = sorted([m for m in pool if m["item"] == it["item"] and m["ti"] == it["ti"] and m["target"] != it["target"]], key=lambda m: m["clone"])[: a.per]
        if not bs:
            continue
        b1 = RRm.Pairs(st, [], [], 1, 0.0, 7 + k, fixed=True, clonex=[it], p_clonex=1.0)[0]
        f1 = st.f[("realx_clone", "clone", it["clone"])]
        r1 = b1["y"] - b1["tab"]
        n1 = r1.shape[1]
        rs = []
        for m in bs:
            b2 = RRm.Pairs(st, [], [], 1, 0.0, 7 + k, fixed=True, clonex=[m], p_clonex=1.0)[0]
            j = np.minimum(warp(f1, st.f[("realx_clone", "clone", m["clone"])])[:n1], b2["y"].shape[1] - 1)
            v = b1["yv"] & b2["yv"][j]
            if v.sum() < 50:
                continue
            r2 = (b2["y"] - b2["tab"])[:, j]
            rs.append(float(np.corrcoef(r1[:, v].ravel(), r2[:, v].ravel())[0, 1]))
        if rs:
            rows.append((it["target"], float(np.mean(rs)), len(rs)))
    spk = np.array([r[0] for r in rows]); X = np.array([r[1] for r in rows])
    us = sorted(set(spk)); idx = {u: np.flatnonzero(spk == u) for u in us}
    rng = np.random.default_rng(0)
    bs_ = np.array([X[np.concatenate([idx[s] for s in rng.choice(us, len(us))])].mean() for _ in range(5000)])
    ci = np.percentile(bs_, [2.5, 97.5]).round(4).tolist()
    d194 = json.loads((ROOT / "results/conv_p0/d194.json").read_text())
    rx, rs_ = float(X.mean()), d194["rho_two_seeds"]
    rep = {"n": len(rows), "speakers": len(us), "pairs": int(sum(r[2] for r in rows)), "rho_cross_speaker": round(rx, 4), "rho_cross_ci95": ci,
           "rho_same_d194": rs_, "sqrt_rho_cross": round(float(np.sqrt(max(rx, 0))), 4), "sqrt_rho_same": round(float(np.sqrt(rs_)), 4),
           "speaker_specific_dcorr_ceiling": round(float(np.sqrt(rs_) - np.sqrt(max(rx, 0))), 4),
           "note": "R の未知: 差し替えた記憶の dcorr 0.376〜0.384・正しい記憶 0.43〜0.45・dcorr_mem 0.053〜0.066(rr12c / rr13c)"}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
