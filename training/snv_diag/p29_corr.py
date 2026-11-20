"""段 P29 step a の a3f(results/snv_p29/prereg.yaml・学習なし・評価だけ): 族の中の、声を跨いだ符号の対応と、変換の不足(oracle − 変換)の関係。
組 = p29_same.py と同じ 242 組。白くした符号 W(系の演奏の前・各々の登録の統計)の最近傍(余弦)が、p29_oracle.py の DTW で揃えた位置の ±8 フレームの内にある率 = 揃えた対応。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p29_corr.py [sys=snv_p24_c] [out=results/snv_p29/corr.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p29_oracle as O
import p29_same as S
import probe_rig as G
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = G.ROOT
DEV = EV.DEV
TOL = 8


@torch.no_grad()
def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag = opt.get("sys", "snv_p24_c")
    out = ROOT / opt.get("out", "results/snv_p29/corr.json")
    s_ = EV.System(tag, EV.load_gen())
    Pl = s_.Pl
    P = S.pairs()
    voices = sorted({p["u"] for p in P} | {p["v"] for p in P})
    enr = {v: [T.load_utt(T.FEAT / f"e1/{v}/{t}.npz") for t in ("t00", "t01", "t02")] for v in voices}
    st = {}
    for v in voices:
        ry, rv, rm, _ = EV.struct_inputs(enr[v])
        st[v] = Pl.mean_of(ry, rv, rm)
    du = {(v, s): T.load_utt(T.FEAT / f"d/{v}/{s}.npz") for p in P for v, s in ((p["u"], p["su"]), (p["v"], p["sv"]))}
    tt = lambda u, k: torch.from_numpy(u[k])[None].to(DEV)
    wc = {}

    def W(v: str, s: str) -> torch.Tensor:
        if (v, s) not in wc:
            u = du[(v, s)]
            wc[(v, s)] = torch.nn.functional.normalize(Pl.white(tt(u, "y"), tt(u, "v"), st[v])[0].T, dim=1)
        return wc[(v, s)]

    rows = []
    for p in P:
        s, t = du[(p["u"], p["su"])], du[(p["v"], p["sv"])]
        j = torch.from_numpy(O.align(s, t)).to(DEV)
        a, b = W(p["u"], p["su"]), W(p["v"], p["sv"])
        n = min(len(a), len(j))
        tv = torch.from_numpy(t["v"][: len(b)] > 0).to(DEV)
        sel = torch.from_numpy(s["v"][:n] > 0).to(DEV) & tv[j[:n].clamp(max=len(b) - 1)]
        sim = (a[:n][sel] @ b.T).masked_fill(~tv[None], -2.0)
        nn = sim.argmax(1)
        jj = j[:n][sel]
        hit = ((nn - jj).abs() <= TOL).float()
        al = sim.gather(1, jj.clamp(max=len(b) - 1)[:, None])[:, 0]
        rows.append({"u": p["u"], "v": p["v"], "hit": round(float(hit.mean()), 4), "cos_aligned": round(float(al.mean()), 4), "cos_all": round(float(sim[sim > -1.5].mean()), 4), "n": int(sel.sum())})
    orc = json.loads((ROOT / "results/snv_p29/oracle.json").read_text())["rows"]
    same = json.loads((ROOT / "results/snv_p29/same.json").read_text())["rows"].get(tag)
    res = {"prereg": "results/snv_p29/prereg.yaml(a3f)", "sys": tag, "tol_frames": TOL, "summary": {}, "rows": rows}
    for sx in ("ef", "em"):
        I = [i for i, r in enumerate(rows) if r["u"].startswith(sx)]
        o = {"n": len(I), "揃えた対応": round(float(np.mean([rows[i]["hit"] for i in I])), 3), "余弦(揃えた位置 / 全体)": [round(float(np.mean([rows[i]["cos_aligned"] for i in I])), 3), round(float(np.mean([rows[i]["cos_all"] for i in I])), 3)]}
        if same is not None:
            h = np.array([rows[i]["hit"] for i in I])
            gap = np.array([orc[i]["ct_oracle"] - same[i]["ct"] for i in I])
            us = np.array([rows[i]["u"] for i in I])
            g, bs = np.random.default_rng(0), []
            uu = sorted(set(us))
            for _ in range(2000):
                idx = np.concatenate([np.flatnonzero(us == u) for u in g.choice(uu, len(uu))])
                bs.append(spearmanr(h[idx], gap[idx]).statistic)
            o["順位相関(揃えた対応, oracle − 変換)"] = [round(float(spearmanr(h, gap).statistic), 3), [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]]
            q = np.argsort(h)
            k = len(q) // 3
            o["oracle − 変換(対応の下 1/3・中・上 1/3)"] = [round(float(gap[q[:k]].mean()), 4), round(float(gap[q[k:2 * k]].mean()), 4), round(float(gap[q[2 * k:]].mean()), 4)]
            o["揃えた対応(同じ 3 群)"] = [round(float(h[q[:k]].mean()), 3), round(float(h[q[k:2 * k]].mean()), 3), round(float(h[q[2 * k:]].mean()), 3)]
        res["summary"][sx] = o
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
