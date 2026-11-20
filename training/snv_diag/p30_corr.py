"""段 P30(results/snv_p30/prereg.yaml・学習なし・評価だけ): 符号の文脈 c の、族の中の声を跨いだ「揃えた対応」。
組 = 段 D の検証の声どうしの、同じ文の (源 u, 目標 v = 検証の女声) 242 組(p29_same.pairs と同じ)。表現 = CPC の文脈 c − その声の登録(E1 の t00〜t02 の有声のフレーム)の平均。
源の有声のフレーム i の、目標の発話の有声のフレームの中の最近傍(余弦)が、DTW(snv_p30_align.align)で揃えた位置 j(i) の ±8 フレーム(40 ms)の内にある率。
検証の声の対応は、この物差しにだけ使う(学習に使わない)。最初の符号との対の差(源の声で束ねる)を並べる。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p30_corr.py snv_p24_xcpc snv_p30_dcpc@step10k [out=results/snv_p30/corr.json] [redo=1]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p29_same as S
import snv_p30_align as AL
import train_snv_p1 as T
import train_snv_p3 as P3

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda"
TOL = 8


@torch.no_grad()
def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tags = [a for a in sys.argv[1:] if "=" not in a]
    out = ROOT / opt.get("out", "results/snv_p30/corr.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {"prereg": "results/snv_p30/prereg.yaml", "tol_frames": TOL, "summary": {}, "rows": {}}
    P = S.pairs()
    voices = sorted({p["u"] for p in P} | {p["v"] for p in P})
    enr = {v: [T.load_utt(T.FEAT / f"e1/{v}/{t}.npz") for t in ("t00", "t01", "t02")] for v in voices}
    du = {(v, s): T.load_utt(T.FEAT / f"d/{v}/{s}.npz") for p in P for v, s in ((p["u"], p["su"]), (p["v"], p["sv"]))}
    cache = out.parent / "val_align.npz"
    if cache.exists():
        J = dict(np.load(cache))
    else:
        J = {f"{p['u']}|{p['su']}|{p['v']}|{p['sv']}": AL.align(AL.feat(T.FEAT / f"d/{p['u']}/{p['su']}.npz")[0], AL.feat(T.FEAT / f"d/{p['v']}/{p['sv']}.npz")[0]) for p in P}
        np.savez_compressed(cache, **J)
    tt = lambda u, k: torch.from_numpy(u[k])[None].to(DEV)
    for tag in tags:
        if tag in res["rows"] and opt.get("redo") != "1":
            continue
        name, _, ck = tag.partition("@")
        Pl = P3.Play(ROOT / "results" / name / f"{ck or 'last'}.pt").to(DEV).eval()
        mu = {}
        for v in voices:
            cs = torch.cat([Pl.ctx(tt(u, "y"), tt(u, "v"))[0].T[torch.from_numpy(u["v"] > 0).to(DEV)] for u in enr[v]])
            mu[v] = cs.mean(0)
        code = {k: torch.nn.functional.normalize(Pl.ctx(tt(u, "y"), tt(u, "v"))[0].T - mu[k[0]], dim=1) for k, u in du.items()}
        rows = []
        for p in P:
            s, t = du[(p["u"], p["su"])], du[(p["v"], p["sv"])]
            j = torch.from_numpy(J[f"{p['u']}|{p['su']}|{p['v']}|{p['sv']}"]).to(DEV)
            a, b = code[(p["u"], p["su"])], code[(p["v"], p["sv"])]
            n = min(len(a), len(j))
            tv = torch.from_numpy(t["v"][: len(b)] > 0).to(DEV)
            jj = j[:n].clamp(max=len(b) - 1)
            sel = torch.from_numpy(s["v"][:n] > 0).to(DEV) & tv[jj]
            nn = (a[:n][sel] @ b.T).masked_fill(~tv[None], -2.0).argmax(1)
            rows.append({"u": p["u"], "v": p["v"], "hit": round(float(((nn - jj[sel]).abs() <= TOL).float().mean()), 4)})
        res["rows"][tag] = rows
        del Pl
    base = tags[0] if tags else next(iter(res["rows"]))
    for tag, rows in res["rows"].items():
        o = {}
        for sx in ("ef", "em"):
            I = [i for i, r in enumerate(rows) if r["u"].startswith(sx)]
            o[sx] = {"n": len(I), "揃えた対応": round(float(np.mean([rows[i]["hit"] for i in I])), 4)}
            if tag != base and base in res["rows"]:
                us = sorted({rows[i]["u"] for i in I})
                d = np.array([np.mean([rows[i]["hit"] - res["rows"][base][i]["hit"] for i in I if rows[i]["u"] == u]) for u in us])
                g = np.random.default_rng(0)
                bs = [float(np.mean(g.choice(d, len(d)))) for _ in range(4000)]
                o[sx][f"対の差(− {base})"] = [round(float(d.mean()), 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]]
        res["summary"][tag] = o
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
