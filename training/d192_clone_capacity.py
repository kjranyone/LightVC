"""D192(学習なし・diag_rr12c の判定): クローンの組(男声 TTS の読み → 目標 A のクローン・記憶 = 手本と同じ A の実録音)で、既知の話者(学習に 40 文を使った 24 人の保留 10 文)と
未知の話者(検証の 12 人 × 10 文)を分ける。記憶を別の話者の手本に差し替えた推論(表は正しい話者のまま)と並べる。
  gain = L1(差し替え) − L1(正しい記憶)(正 = 正しい記憶が効く)・dcorr = corr(R − 表, 目標 − 表) の正しい記憶 − 差し替え(残差の向きが本人に寄る分)。
話者単位のブートストラップ 5000 回。事前登録 = results/diag_rr12c/prereg.yaml。
    uv run python d192_clone_capacity.py --tag diag_rr12c --out ../results/diag_rr12c/verdict_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    dev = "cuda"
    ck = torch.load(ROOT / "results" / a.tag / "last.pt", map_location="cpu", weights_only=False)
    RRm.CFG["align"] = ck["args"].get("align", "rr")
    net = RRm.build(ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["net"])
    st = RRm.Store()
    cm = [it for it in json.loads((ROOT / "data/realx/items_clone_multi.json").read_text()) if ("realx_clone", "clone", it["clone"]) in st.f]
    rep: dict = {"n": a.n}
    rows: dict = {}
    for name, its in (("seen", [it for it in cm if it["group"] == "seen" and it["held"]]), ("unseen", [it for it in cm if it["group"] == "unseen"])):
        out = []
        for swap in (False, True):
            ds = RRm.Pairs(st, [], [], a.n, 0.0, 7, fixed=True, clonex=its, p_clonex=1.0, swap_mem=swap)
            dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4, collate_fn=RRm.collate)
            res = []
            with torch.no_grad():
                for b in dl:
                    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                    env, _, _ = net(b["q"], b["tab"], net.memory(b["rp"], b["rv"]), b["rm"])
                    v = b["yv"][0] & b["tm"][0]
                    e, y, t = env[0][:, v], b["y"][0][:, v], b["tab"][0][:, v]
                    l1 = float((e - y).abs().mean())
                    r1, r2 = (e - t).flatten(), (y - t).flatten()
                    dc = float(torch.corrcoef(torch.stack([r1, r2]))[0, 1])
                    res.append((b["A"][0], l1, float((t - y).abs().mean()), dc))
            out.append(res)
        rows[name] = [(r[0], r[1], r[2], r[3], s_[1], s_[3]) for r, s_ in zip(out[0], out[1])]
    rng = np.random.default_rng(0)
    for name, rr in rows.items():
        spk = np.array([r[0] for r in rr]); X = np.array([r[1:] for r in rr], float)
        us = sorted(set(spk)); idx = {u: np.flatnonzero(spk == u) for u in us}
        B = [np.concatenate([idx[s] for s in rng.choice(us, len(us))]) for _ in range(5000)]
        gain = X[:, 3] - X[:, 0]; dcor = X[:, 2] - X[:, 4]
        ci = lambda z: [round(float(q), 4) for q in np.percentile([z[ii].mean() for ii in B], [2.5, 97.5])]
        rep[name] = {"speakers": len(us), "l1_rr": round(float(X[:, 0].mean()), 4), "l1_tab": round(float(X[:, 1].mean()), 4), "l1_swap": round(float(X[:, 3].mean()), 4),
                     "dcorr_right": round(float(X[:, 2].mean()), 4), "dcorr_swap": round(float(X[:, 4].mean()), 4),
                     "gain_mem": round(float(gain.mean()), 4), "gain_mem_ci95": ci(gain), "dcorr_mem": round(float(dcor.mean()), 4), "dcorr_mem_ci95": ci(dcor)}
        rows[name] = (us, gain, dcor, spk)
    us_s, g_s, d_s, sp_s = rows["seen"]; us_u, g_u, d_u, sp_u = rows["unseen"]
    bs = []
    for _ in range(5000):
        ss = rng.choice(us_s, len(us_s)); su = rng.choice(us_u, len(us_u))
        i_s = np.concatenate([np.flatnonzero(sp_s == s) for s in ss]); i_u = np.concatenate([np.flatnonzero(sp_u == s) for s in su])
        bs.append((g_s[i_s].mean() - g_u[i_u].mean(), d_s[i_s].mean() - d_u[i_u].mean()))
    bs = np.array(bs)
    rep["seen_minus_unseen"] = {"gain_mem": round(float(g_s.mean() - g_u.mean()), 4), "gain_mem_ci95": np.round(np.percentile(bs[:, 0], [2.5, 97.5]), 4).tolist(),
                                "dcorr_mem": round(float(d_s.mean() - d_u.mean()), 4), "dcorr_mem_ci95": np.round(np.percentile(bs[:, 1], [2.5, 97.5]), 4).tolist()}
    sm = rep["seen_minus_unseen"]
    rep["verdict"] = "CAPACITY_OK" if sm["dcorr_mem_ci95"][0] > 0 and sm["gain_mem_ci95"][0] > 0 else ("CAPACITY_FAIL" if sm["dcorr_mem_ci95"][0] <= 0 and sm["gain_mem_ci95"][0] <= 0 else "MIXED")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
