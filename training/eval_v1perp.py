"""速い v1⊥ の評価(見当づけ用・10 分以内)。注意(D290): 決定値は群ごとに判別器を学習し直すので群の大きさに依存する(30 本と 90 本で約 1.1〜1.5 ずれる)。
大きさの違う群を比べるときは --one_per_target か同じ本数にそろえる。成績表(dev 60 どうし)は同じ大きさなので比べてよい。: results/conv_p0/v1perp_basis.npz(D267 の INLP の基底・標準化・本物 150 人の特徴)を読み、群ごとに v1 と v1⊥ の AUC・p_fake の中央値・決定値(対数オッズ)の中央値を出す。
    uv run python eval_v1perp.py --groups "A=<glob>" "B=<glob>" --limit 30 --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--one_per_target", action="store_true", help="目標ごとに 1 本(名前順の最初)= 群の大きさを目標の数にそろえる。群ごとの C2ST の決定値は群の大きさ(本物 150 との比と判別器の鋭さ)に依存するので、大きさの違う群を比べるときは必須(D290)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    z = np.load(ROOT / "results/conv_p0/v1perp_basis.npz")
    B, mu, scl, R = z["B"], z["mean"], z["scale"], z["R"]
    norm = lambda X: (X - mu) / scl
    proj = lambda X, Bs: X - (X @ Bs.T) @ Bs if len(Bs) else X
    Rz = norm(R)
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval().float()

    @torch.no_grad()
    def emb(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to("cuda"), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    rf = np.arange(len(Rz)) % 5

    def two(Xz: np.ndarray, tg: np.ndarray, Bs: np.ndarray) -> dict:
        Xa = np.concatenate([proj(Rz, Bs), proj(Xz, Bs)])
        ya = np.r_[np.zeros(len(Rz)), np.ones(len(Xz))]
        fa = np.r_[rf, tg % 5]
        pr, dv = np.zeros(len(ya)), np.zeros(len(ya))
        for k in range(5):
            te = fa == k
            s2 = StandardScaler().fit(Xa[~te])
            c2 = LogisticRegression(C=0.01, max_iter=2000).fit(s2.transform(Xa[~te]), ya[~te])
            pr[te] = c2.predict_proba(s2.transform(Xa[te]))[:, 1]
            dv[te] = c2.decision_function(s2.transform(Xa[te]))
        d = dv[len(Rz):]
        us = sorted(set(tg.tolist()))
        idx = {u: np.flatnonzero(tg == u) for u in us}
        rng = np.random.default_rng(0)
        bs = [np.median(d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])]) for _ in range(2000)]
        return {"auc": round(float(roc_auc_score(ya, pr)), 4), "p_fake": round(float(np.median(pr[len(Rz):])), 4), "logit": round(float(np.median(d)), 3),
                "logit_ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "per_utt": [round(float(v), 3) for v in d]}

    rep: dict = {}
    for g in a.groups:
        name, pat = g.split("=", 1)
        files = sorted(Path(p) for p in glob.glob(pat))
        if a.one_per_target:
            seen: set = set()
            files = [f for f in files if not (f.name.split("_t")[1][:3] in seen or seen.add(f.name.split("_t")[1][:3]))]
        if a.limit:
            files = files[: a.limit]
        Xz = norm(np.array([emb(f) for f in files]))
        tg = np.array([int(f.name.split("_t")[1][:3]) for f in files])
        rep[name] = {"n": len(files), "v1": two(Xz, tg, np.zeros((0, Xz.shape[1]))), "v1perp": two(Xz, tg, B)}
        print(name, {k: ({kk: vv for kk, vv in v.items() if kk != "per_utt"} if isinstance(v, dict) else v) for k, v in rep[name].items()}, flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
