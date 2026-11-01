"""v1'(別系統の物差し・wav2vec2-base-960h の 4/6/8 層の平均と標準偏差・8 秒): 学習の SLM 判別器(WavLM-base-plus)とも v1(WavLM-large)とも別の SSL で、
同じ本物の女声 150 本(results/conv_p0/d237.json の REAL_POOL・各話者の真ん中のファイル)と比べる C2ST。判別は eval_v1perp.py と同じ(5 分割・C 0.01・決定値の中央値・目標でまとめたブートストラップ)・INLP なし。
群の大きさに依存する(D290)= 同じ本数どうしで比べる。合わせ込み(rvslm2: WavLM 系だけ良くなる)の確認用。
    uv run python eval_v1p.py --groups "A=<glob>" "B=<glob>" --out <json>
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import Wav2Vec2Model
    net = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to("cuda").eval().float()

    @torch.no_grad()
    def emb(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to("cuda"), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    R = []
    for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]:
        if r["grp"] == "REAL_POOL":
            ws = sorted((ROOT / "female-dataset" / r["id"]).glob("*.wav"))
            R.append(emb(ws[len(ws) // 2]))
    R = np.array(R)
    rf = np.arange(len(R)) % 5
    rep: dict = {"n_real": len(R)}
    for g in a.groups:
        name, pat = g.split("=", 1)
        files = sorted(Path(p) for p in glob.glob(pat))
        X = np.array([emb(f) for f in files])
        tg = np.array([int(f.name.split("_t")[1][:3]) for f in files])
        Xa = np.concatenate([R, X])
        ya = np.r_[np.zeros(len(R)), np.ones(len(X))]
        fa = np.r_[rf, tg % 5]
        pr, dv = np.zeros(len(ya)), np.zeros(len(ya))
        for k in range(5):
            te = fa == k
            sc = StandardScaler().fit(Xa[~te])
            c = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Xa[~te]), ya[~te])
            pr[te] = c.predict_proba(sc.transform(Xa[te]))[:, 1]
            dv[te] = c.decision_function(sc.transform(Xa[te]))
        d = dv[len(R):]
        us = sorted(set(tg.tolist()))
        idx = {u: np.flatnonzero(tg == u) for u in us}
        rng = np.random.default_rng(0)
        bs = [np.median(d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])]) for _ in range(2000)]
        rep[name] = {"n": len(files), "auc": round(float(roc_auc_score(ya, pr)), 4), "logit": round(float(np.median(d)), 3),
                     "logit_ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
        print(name, rep[name], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
