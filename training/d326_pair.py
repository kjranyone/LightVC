"""D326b: 同じ目標への変換で、女声の入力からのもの(A)と男声の入力からのもの(B)を直接見分けられるか(2 標本の判別)。本物の女声 150 との C2ST は英語・VCTK の領域で頭打ち
(入力そのものの男女差 0.02)になるので、男声の分を A 対 B の判別の AUC で測る。特徴 = eval_v1perp と同じ(WavLM-large 4/6/8 の平均と標準偏差・8 秒・v1⊥ の射影も)。
分割は目標でまとめた 5 分割(devset.json に block があれば同じ番号の女声と男声が同じ文の区間を読む組だけを使う・3 つ目に devset.json を与えると操作者の話者でまとめた 5 分割 = 同じ話者が学習と検査にまたがらない)・帰無 = 目標ごとに A / B の名札を入れ替える置換(既定 200 回)。
    uv run python d326_pair.py --pairs "VCTK_CV=<A の glob>|<B の glob>[|<devset.json>]" ... --out <json>
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
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    z = np.load(ROOT / "results/conv_p0/v1perp_basis.npz")
    B, mu, scl = z["B"], z["mean"], z["scale"]
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

    def cv_auc(X: np.ndarray, y: np.ndarray, fo: np.ndarray) -> float:
        pr = np.zeros(len(y))
        for k in range(5):
            te = fo % 5 == k
            s = StandardScaler().fit(X[~te])
            c = LogisticRegression(C=0.01, max_iter=2000).fit(s.transform(X[~te]), y[~te])
            pr[te] = c.predict_proba(s.transform(X[te]))[:, 1]
        return float(roc_auc_score(y, pr))

    rep: dict = {}
    rng = np.random.default_rng(326)
    for pr_ in a.pairs:
        name, pats = pr_.split("=", 1)
        parts = pats.split("|")
        pa, pb = parts[0], parts[1]
        fa, fb = sorted(Path(p) for p in glob.glob(pa)), sorted(Path(p) for p in glob.glob(pb))
        if len(parts) > 2:
            rows = json.loads(Path(parts[2]).read_text())["pairs"]
            if any(r.get("block") is not None for r in rows):
                bl: dict = {}
                for r in rows:
                    bl.setdefault(r["pi"] % 100, {})[r["kind"]] = r["block"]
                ok = {j for j, v in bl.items() if len(v) == 2 and len(set(v.values())) == 1}
                fa, fb = [f for f in fa if int(f.name[1:5]) % 100 in ok], [f for f in fb if int(f.name[1:5]) % 100 in ok]
        files = fa + fb
        X0 = norm(np.array([emb(f) for f in files]))
        y = np.r_[np.zeros(len(fa)), np.ones(len(fb))]
        tg = np.array([int(f.name.split("_t")[1][:3]) for f in files])
        fo = tg
        if len(parts) > 2:
            spk = {r["pi"]: r["kind"] + r["spk"] for r in json.loads(Path(parts[2]).read_text())["pairs"]}
            us = sorted({spk[int(f.name[1:5])] for f in files})
            fo = np.array([us.index(spk[int(f.name[1:5])]) for f in files])
        out = {"nA": len(fa), "nB": len(fb), "folds": "speaker" if len(parts) > 2 else "target"}
        for key, Bs in (("v1", np.zeros((0, X0.shape[1]))), ("v1perp", B)):
            X = proj(X0, Bs)
            obs = cv_auc(X, y, fo)
            null = []
            for _ in range(a.perm):
                yp = y.copy()
                for u in np.unique(tg):
                    i = np.flatnonzero(tg == u)
                    yp[i] = rng.permutation(yp[i])
                null.append(cv_auc(X, yp, fo))
            null = np.array(null)
            out[key] = {"auc": round(obs, 4), "null_mean": round(float(null.mean()), 4), "null_p95": round(float(np.percentile(null, 95)), 4), "p": round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4)}
        rep[name] = out
        print(name, out, flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
