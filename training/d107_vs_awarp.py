"""D107(学習なし): 変換向けの関門の候補 = 『目標本人が元の話者の時間の流れで話したもの』と区別できないか。
基準 = AWARP(D106: A 自身の F3 の包絡・F2 の c0・教師 f0・F4 を M の時間へ伸縮し同じ出力部で鳴らす)・候補 = 同じ組の変換の各条件。
同じ出力部の音・同じ文・同じ時間の流れ = TTS のコーデックの指紋とリズムの違いでは分かれない。特徴 = WavLM-large 層 4/6/8 の平均 + 標準偏差 → 標準化 → L2 ロジスティック・男声で分けた 5 分割の AUC。
出力 = results/conv_p0/d107.json。
    uv run python d107_vs_awarp.py --dir <scratchpad>/d106 --conds CONV XC0 XDYN XF0 XPER XALL C0MAP ASELF
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--ref", default="AWARP")
    ap.add_argument("--tag", default="d107")
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    dev = "cuda"
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dev).eval().float()
    d = Path(a.dir)

    def load16(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        return resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)

    @torch.no_grad()
    def emb(x: np.ndarray) -> np.ndarray:
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to(dev), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    def load_set(c: str) -> tuple:
        fs = sorted(d.glob(f"m*_{c}.wav"))
        return np.array([emb(load16(f)) for f in fs]), np.array([int(f.name[1:4]) for f in fs])

    Xr, gr = load_set(a.ref)
    rep = {}
    for c in a.conds:
        X, g = load_set(c)
        Xa = np.concatenate([Xr, X]); ya = np.r_[np.zeros(len(Xr)), np.ones(len(X))]; ga = np.r_[gr, g]
        pr = np.zeros(len(ya))
        for k in range(5):
            te = (ga % 5) == k
            sc = StandardScaler().fit(Xa[~te])
            clf = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xa[~te]), ya[~te])
            pr[te] = clf.predict_proba(sc.transform(Xa[te]))[:, 1]
        rep[c] = round(float(roc_auc_score(ya, pr)), 3)
        print(c, "vs", a.ref, "AUC", rep[c], len(X), flush=True)
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
