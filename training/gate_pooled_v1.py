"""対にできない 2 つの群(例: 男声入力の変換 対 女声入力の変換・同じ目標の集合)を同じ判別器で比べる(gate_mf_v1 の N 軸)。本物 150(v1perp_basis の R)対 2 群をまとめた偽物で判別器を 1 つ学習し
(偽物は目標でまとめた 5 分割・本物は番号で 5 分割・分割の外の決定値)、群ごとの決定値の平均・中央値と、差(A − B の平均)を目標でまとめたブートストラップ(目標を引き直し、引いた目標の各群の発話を使う)の 95% 区間で出す。
    uv run python gate_pooled_v1.py --dir <dir with A/ and B/> --a MALE --b FEMALE --out <json>
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
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
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
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval().float()

    @torch.no_grad()
    def emb(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to("cuda"), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    D = Path(a.dir)
    fa, fb = sorted((D / a.a).glob("p*.wav")), sorted((D / a.b).glob("p*.wav"))
    Xa, Xb = norm(np.array([emb(f) for f in fa])), norm(np.array([emb(f) for f in fb]))
    ta = np.array([int(f.name[7:10]) for f in fa])
    tb = np.array([int(f.name[7:10]) for f in fb])
    Rz = norm(R)
    rng = np.random.default_rng(1)
    rep: dict = {"nA": len(fa), "nB": len(fb)}
    for key, Bs in (("v1", np.zeros((0, Rz.shape[1]))), ("v1perp", B)):
        Xr, Xf = proj(Rz, Bs), proj(np.concatenate([Xa, Xb]), Bs)
        tf = np.r_[ta, tb]
        dv = np.zeros(len(Xf))
        for k5 in range(5):
            tr_r, tr_f = np.arange(len(Xr)) % 5 != k5, tf % 5 != k5
            Xtr = np.concatenate([Xr[tr_r], Xf[tr_f]])
            ytr = np.r_[np.zeros(tr_r.sum()), np.ones(tr_f.sum())]
            s = StandardScaler().fit(Xtr)
            c = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(s.transform(Xtr), ytr)
            dv[~tr_f] = c.decision_function(s.transform(Xf[~tr_f]))
        da, db = dv[: len(fa)], dv[len(fa):]
        us = sorted(set(ta.tolist()) & set(tb.tolist()))
        ia = {u: np.flatnonzero(ta == u) for u in us}
        ib = {u: np.flatnonzero(tb == u) for u in us}
        bs = []
        for _ in range(a.boot):
            pick = rng.choice(us, len(us))
            bs.append(da[np.concatenate([ia[u] for u in pick])].mean() - db[np.concatenate([ib[u] for u in pick])].mean())
        rep[key] = {"mean": {a.a: round(float(da.mean()), 3), a.b: round(float(db.mean()), 3)}, "median": {a.a: round(float(np.median(da)), 3), a.b: round(float(np.median(db)), 3)},
                    "diff_A_minus_B": round(float(da.mean() - db.mean()), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "n_targets_shared": len(us)}
        print(key, json.dumps(rep[key], ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
