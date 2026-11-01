"""D269(学習なし・見当づけ): v1 の特徴(WavLM-large 層 4/6/8 × 平均 / 標準偏差 = 6 束 × 1024)の束ごとに本物の女声との二標本の AUC と決定値の中央値を出し、男声 → 女声と女声 → 女声を本物から分ける束を比べる。
    uv run python d269_featgroups.py --groups "MF=<glob>" "FF=<glob>" --limit 30 --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["L4mean", "L4std", "L6mean", "L6std", "L8mean", "L8std"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    R = np.load(ROOT / "results/conv_p0/real_pool_wavlm_l468.npy")
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval().float()

    @torch.no_grad()
    def emb(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to("cuda"), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    rf = np.arange(len(R)) % 5
    rep: dict = {}
    for g in a.groups:
        name, pat = g.split("=", 1)
        files = sorted(Path(p) for p in glob.glob(pat))[: a.limit]
        X = np.array([emb(f) for f in files])
        tg = np.array([int(f.name.split("_t")[1][:3]) for f in files])
        rep[name] = {}
        for gi, gn in enumerate(NAMES):
            sl = slice(gi * 1024, (gi + 1) * 1024)
            Xa = np.concatenate([R[:, sl], X[:, sl]])
            ya = np.r_[np.zeros(len(R)), np.ones(len(X))]
            fa = np.r_[rf, tg % 5]
            pr, dv = np.zeros(len(ya)), np.zeros(len(ya))
            for k in range(5):
                te = fa == k
                s2 = StandardScaler().fit(Xa[~te])
                c2 = LogisticRegression(C=0.01, max_iter=2000).fit(s2.transform(Xa[~te]), ya[~te])
                pr[te] = c2.predict_proba(s2.transform(Xa[te]))[:, 1]
                dv[te] = c2.decision_function(s2.transform(Xa[te]))
            rep[name][gn] = {"auc": round(float(roc_auc_score(ya, pr)), 4), "logit": round(float(np.median(dv[len(R):])), 3)}
        print(name, {k: (v["auc"], v["logit"]) for k, v in rep[name].items()}, flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
