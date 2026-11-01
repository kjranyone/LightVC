"""D467 の評価(10-21 査読 R1 / R2 への対応): 同じ発話の版どうしを同じ判別器で比べる対の v1。組ごとに 本物 150(v1perp_basis の R)対 全ての版をまとめた偽物 で判別器を 1 つ学習し
(偽物は目標でまとめた 5 分割・本物は番号で 5 分割・分割の外の決定値)、版ごとの決定値の中央値と、版の対の発話ごとの差の平均を目標でまとめたブートストラップの 95% 区間で出す。
群ごとに判別器を学習し直す eval_v1perp(群の大きさに依存・D290)と違い、版の差は同じ判別器の中の対の差 = 部分集合の雑音を含まない。v1 と v1⊥ の両方。
    uv run python d467_paired_v1.py --dir ../results/conv_p0/d467/dev2 --ref OFF RND15C --out ../results/conv_p0/d467/dev2_paired.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--devset", default="", help="devset.json: ブートストラップを操作者の話者(spk)でまとめる(既定 = 目標)""")
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    z = np.load(ROOT / "results/conv_p0/v1perp_basis.npz")
    B, mu, scl, R = z["B"], z["mean"], z["scale"], z["R"]
    norm = lambda X: (X - mu) / scl
    proj = lambda X, Bs: X - (X @ Bs.T) @ Bs if len(Bs) else X
    import os
    dv_ = os.environ.get("V1_DEVICE", "cuda")
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dv_).eval().float()

    @torch.no_grad()
    def emb(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to(dv_), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    D = Path(a.dir)
    vars_ = sorted(d.name for d in D.iterdir() if d.is_dir())
    keys = sorted(set.intersection(*[{f.name[:10] for f in (D / v).glob("p*.wav")} for v in vars_]))
    X = {v: norm(np.array([emb(next((D / v).glob(f"{k}_*.wav"))) for k in keys])) for v in vars_}
    tg = np.array([int(k[7:10]) for k in keys])
    Rz = norm(R)
    rep: dict = {"n": len(keys), "vars": vars_}
    rng = np.random.default_rng(467)
    for key, Bs in (("v1", np.zeros((0, Rz.shape[1]))), ("v1perp", B)):
        Xr = proj(Rz, Bs)
        Xf = np.concatenate([proj(X[v], Bs) for v in vars_])
        tf = np.tile(tg, len(vars_))
        dv = np.zeros(len(Xf))
        for k5 in range(5):
            tr_r, tr_f = np.arange(len(Xr)) % 5 != k5, tf % 5 != k5
            Xtr = np.concatenate([Xr[tr_r], Xf[tr_f]])
            ytr = np.r_[np.zeros(tr_r.sum()), np.ones(tr_f.sum())]
            s = StandardScaler().fit(Xtr)
            c = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(s.transform(Xtr), ytr)
            dv[~tr_f] = c.decision_function(s.transform(Xf[~tr_f]))
        per = {v: dv[i * len(keys):(i + 1) * len(keys)] for i, v in enumerate(vars_)}
        cl = tg
        if a.devset:
            spk = {r["pi"]: r["spk"] for r in json.loads(Path(a.devset).read_text())["pairs"]}
            names = sorted(set(spk.values()))
            cl = np.array([names.index(spk[int(k[1:5])]) for k in keys])
        us = sorted(set(cl.tolist()))
        idx = {u: np.flatnonzero(cl == u) for u in us}
        out = {"median": {v: round(float(np.median(per[v])), 3) for v in vars_}, "diff": {}}
        for r_ in a.ref:
            for v in vars_:
                if v == r_:
                    continue
                d = per[v] - per[r_]
                bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(a.boot)]
                out["diff"][f"{v}-{r_}"] = {"mean": round(float(d.mean()), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
        rep[key] = out
        print(key, json.dumps(out), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
