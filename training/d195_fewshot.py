"""D195(診断・登録時の適応の量の検査): 学習済みの R(diag_rr12c)を未知の話者 1 人ずつ、その話者のクローンの組 5 文で短く追加学習し、残り 5 文(保留)で記憶の効果・残差の向きを測る。
対照 = 別の未知の話者の 5 文で同じ量だけ追加学習したもの(クローンの領域への一般の適応を差し引く)。評価の記憶はどちらも正しい話者の手本。
diag_rr12c は既知の話者 40 文で話者固有の実現を出した(CAPACITY_OK)= 話者あたり何文で出るかの下限の目安(20 秒の参照 ≈ 数文)。
    uv run python d195_fewshot.py --tag diag_rr12c --out ../results/conv_p0/d195.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent


def measure(net, st, its: list, mult: int, dev: str) -> np.ndarray:
    net.eval()
    ds = RRm.Pairs(st, [], [], mult * len(its), 0.0, 7, fixed=True, clonex=its, p_clonex=1.0)
    out = []
    with torch.no_grad():
        for i in range(len(ds)):
            b = RRm.collate([ds[i]])
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            env, _, _ = net(b["q"], b["tab"], net.memory(b["rp"], b["rv"]), b["rm"])
            v = b["yv"][0] & b["tm"][0]
            e, y, t = env[0][:, v], b["y"][0][:, v], b["tab"][0][:, v]
            dc = float(torch.corrcoef(torch.stack([(e - t).flatten(), (y - t).flatten()]))[0, 1])
            out.append((float((e - y).abs().mean()), dc))
    return np.array(out)


def adapt(net0, st, its: list, steps: int, lr: float, seed: int, dev: str, lam: float):
    net = copy.deepcopy(net0).train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ds = RRm.Pairs(st, [], [], steps * 8, 0.0, seed, fixed=True, crop=400, clonex=its, p_clonex=1.0)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=4, collate_fn=RRm.collate)
    for b in dl:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        env, dc0, _ = net(b["q"], b["tab"], net.memory(b["rp"], b["rv"]), b["rm"])
        loss, _, _ = RRm.lossf(env, dc0, b, lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="diag_rr12c")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mult", type=int, default=4)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    ck = torch.load(ROOT / "results" / a.tag / "last.pt", map_location="cpu", weights_only=False)
    RRm.CFG["align"] = ck["args"].get("align", "rr")
    net0 = RRm.build(ck["cfg"]).to(dev)
    net0.load_state_dict(ck["net"])
    lam = float(ck["args"].get("lam_c0", 1.0))
    st = RRm.Store()
    cm = [it for it in json.loads((ROOT / "data/realx/items_clone_multi.json").read_text()) if it["group"] == "unseen" and ("realx_clone", "clone", it["clone"]) in st.f]
    spk = sorted({it["target"] for it in cm})
    by = {s: [it for it in cm if it["target"] == s] for s in spk}
    rows = []
    for si, s in enumerate(spk):
        ad, te = by[s][: a.k], by[s][a.k:]
        other = spk[(si + 1) % len(spk)]
        z = measure(net0, st, te, a.mult, dev)
        own = measure(adapt(net0, st, ad, a.steps, a.lr, 100 + si, dev, lam), st, te, a.mult, dev)
        oth = measure(adapt(net0, st, by[other][: a.k], a.steps, a.lr, 100 + si, dev, lam), st, te, a.mult, dev)
        for r0, r1, r2 in zip(z, own, oth):
            rows.append((s, *r0, *r1, *r2))
        print(s, "zero", z.mean(0).round(4).tolist(), "own", own.mean(0).round(4).tolist(), "other", oth.mean(0).round(4).tolist(), flush=True)
    S = np.array([r[0] for r in rows]); X = np.array([r[1:] for r in rows], float)
    us = sorted(set(S)); idx = {u: np.flatnonzero(S == u) for u in us}
    rng = np.random.default_rng(0)
    d_own_oth_l1 = X[:, 4] - X[:, 2]
    d_own_oth_dc = X[:, 3] - X[:, 5]
    d_own_zero_dc = X[:, 3] - X[:, 1]
    d_oth_zero_dc = X[:, 5] - X[:, 1]
    zs = [d_own_oth_l1, d_own_oth_dc, d_own_zero_dc, d_oth_zero_dc]
    bs = np.array([[z[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for z in zs] for _ in range(5000)])
    ci = np.percentile(bs, [2.5, 97.5], axis=0).round(4).T.tolist()
    rep = {"tag": a.tag, "k": a.k, "steps": a.steps, "lr": a.lr, "speakers": len(us), "n": len(rows),
           "zero": {"l1": round(float(X[:, 0].mean()), 4), "dcorr": round(float(X[:, 1].mean()), 4)},
           "own": {"l1": round(float(X[:, 2].mean()), 4), "dcorr": round(float(X[:, 3].mean()), 4)},
           "other": {"l1": round(float(X[:, 4].mean()), 4), "dcorr": round(float(X[:, 5].mean()), 4)},
           "own_vs_other_l1_gain": [round(float(d_own_oth_l1.mean()), 4)] + ci[0], "own_vs_other_dcorr": [round(float(d_own_oth_dc.mean()), 4)] + ci[1],
           "own_vs_zero_dcorr": [round(float(d_own_zero_dc.mean()), 4)] + ci[2], "other_vs_zero_dcorr": [round(float(d_oth_zero_dc.mean()), 4)] + ci[3]}
    ok = ci[1][0] > 0 and ci[0][0] > 0 and d_own_oth_dc.mean() >= 0.04
    flat = ci[1][0] <= 0 and ci[0][0] <= 0
    rep["verdict"] = "FEWSHOT_OK" if ok else ("FEWSHOT_FAIL" if flat else "MIXED")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
