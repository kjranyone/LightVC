"""D332: diag_ppgvoc19 の対の損失(同じ文・同じ目標への 2 つの変換の WavLM-base-plus 全層の平均・標準偏差の L1)の床。
同じ文・同じ目標 b で 男 i ↔ 女 j(学習と同じ組)・男 i ↔ 女 i(同じ語の組)・女 i ↔ 女 j・男 i ↔ 男 j の距離を並べる。
女 ↔ 女 = 操作者の個人差だけの床・男 ↔ 女 − 女 ↔ 女 = 損失が詰められる男声の分。描画は学習の損失の塊と同じ(先頭 pair_sec 秒・表の経路・励起の種は固定)。
    uv run python d332_pairfloor.py --ckpt ../results/diag_ppgvoc18/last_ema.pt --xpair <dev の親>/xpair --out ../results/conv_p0/d332/ppgvoc18.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
import train_rvoc as TR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--xpair", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--sec", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=332)
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX["on"] = bool(st.get("ctx", False))
    P.CTX["ch"] = int(st.get("ctx_ch", 192))
    P.CTX["adapt"] = bool(st.get("adapt", False))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    P.TAB = P.ROOT / "data/rvoc_tables_real"
    _, spks, _ = P.load_rows({h[0] for h in P.e2e_held()}, True)
    zs = [np.load(P.TAB / f"{s}.npz") for s in spks]
    tabT = torch.from_numpy(np.stack([z["T"] for z in zs])).float().to(dev)
    tabP = torch.from_numpy(np.stack([z["per_p75"] for z in zs])).float().to(dev)
    tabMu = torch.from_numpy(np.array([float(z["mu"]) for z in zs])).float().to(dev)
    from slm_disc import SLMDisc
    slm = SLMDisc("microsoft/wavlm-base-plus", 256).to(dev).eval()
    pool = P.PairPool(a.xpair, 0)
    by_spk = {}
    for t, v in pool.by.items():
        for gk in ("m", "f"):
            for c, lg, spk in v[gk]:
                by_spk[(t, spk)] = (c, lg, pool.mu[spk])
    idx = sorted({int(s[2:]) for (_, s) in by_spk})
    rng = random.Random(a.seed)
    tmax = int(a.sec * N.SR / N.HOP)

    @torch.no_grad()
    def stat(t: str, spk: str, b: int) -> torch.Tensor:
        c, lg, mu = by_spk[(t, spk)]
        n = min(len(c), tmax)
        cc = torch.from_numpy(c[:n].T.copy()).float().to(dev)[None]
        pp = P.ppg_form(torch.from_numpy(lg[:n].astype(np.float32).T.copy()).float().to(dev)[None], P.PPG)
        vv = (cc[:, 26:27] > 0.5).float()
        lf = torch.where(vv > 0.5, cc[:, 25:26] + (tabMu[b] - mu), torch.zeros_like(vv))
        f0 = torch.where(vv[:, 0] > 0.5, 200.0 * torch.exp(lf[:, 0]), torch.zeros_like(lf[:, 0]))
        ex = TR.excitation(f0, cc.shape[-1] * N.HOP, torch.Generator().manual_seed(0)).to(dev)
        sp = (tabT[b:b + 1], tabP[b:b + 1])
        y = g(g.make(pp, cc[:, 0:1], lf, vv, sp), ex, sp)[:, TR.WARM:]
        h = slm.feats(y)
        return torch.cat([h.mean(-1), h.std(-1)], -1)[0]

    rows = []
    for k in range(a.n):
        t = rng.choice(pool.texts)
        i, j = rng.sample([q for q in idx if (t, f"pm{q:03d}") in by_spk and (t, f"pf{q:03d}") in by_spk], 2)
        b = rng.randrange(len(spks))
        s = {n_: stat(t, n_, b) for n_ in (f"pm{i:03d}", f"pm{j:03d}", f"pf{i:03d}", f"pf{j:03d}")}
        d = lambda x, y: float((s[x] - s[y]).abs().mean())
        rows.append({"t": t, "b": b, "i": i, "j": j, "MF_x": d(f"pm{i:03d}", f"pf{j:03d}"), "MF_same": d(f"pm{i:03d}", f"pf{i:03d}"),
                     "FF": d(f"pf{i:03d}", f"pf{j:03d}"), "MM": d(f"pm{i:03d}", f"pm{j:03d}")})
    out = {"ckpt": a.ckpt, "n": len(rows), "sec": a.sec}
    for key in ("MF_x", "MF_same", "FF", "MM"):
        v = np.array([r[key] for r in rows])
        out[key] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v)))}
    dv = np.array([r["MF_x"] - r["FF"] for r in rows])
    bs = np.random.default_rng(0).choice(dv, (2000, len(dv))).mean(1)
    out["MFx_minus_FF"] = {"mean": float(dv.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    out["male_share"] = float(dv.mean() / np.mean([r["MF_x"] for r in rows]))
    out["rows"] = rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
