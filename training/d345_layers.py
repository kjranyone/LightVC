"""D345: 層ごとの C2ST(WavLM-large の 1 層ずつ・平均と標準偏差・8 秒・本物 = d237 の REAL_POOL 150・5 分割・C 0.01)で、群と本物を見分ける運び手が浅い層(音響の質感)か深い層(音韻・韻律)かを見る。診断。
    uv run python d345_layers.py --groups "A=<glob>" ... --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
LAYERS = (1, 2, 4, 6, 8, 12, 16, 20, 24)


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
    from transformers import WavLMModel
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval().float()

    @torch.no_grad()
    def emb(p: Path) -> dict:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to("cuda"), output_hidden_states=True).hidden_states
        return {l: torch.cat([hs[l][0].mean(0), hs[l][0].std(0)]).cpu().numpy() for l in LAYERS}

    R = []
    for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]:
        if r["grp"] == "REAL_POOL":
            ws = sorted((ROOT / "female-dataset" / r["id"]).glob("*.wav"))
            R.append(emb(ws[len(ws) // 2]))
    rf = np.arange(len(R)) % 5
    rep: dict = {}
    for g in a.groups:
        name, pat = g.split("=", 1)
        files = sorted(Path(p) for p in glob.glob(pat))
        X = [emb(f) for f in files]
        tg = np.array([int(f.name.split("_t")[1][:3]) for f in files])
        row = {}
        for l in LAYERS:
            Xa = np.concatenate([np.array([r[l] for r in R]), np.array([x[l] for x in X])])
            ya = np.r_[np.zeros(len(R)), np.ones(len(X))]
            fa = np.r_[rf, tg % 5]
            pr = np.zeros(len(ya))
            for k in range(5):
                te = fa == k
                sc = StandardScaler().fit(Xa[~te])
                c = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Xa[~te]), ya[~te])
                pr[te] = c.predict_proba(sc.transform(Xa[te]))[:, 1]
            row[l] = round(float(roc_auc_score(ya, pr)), 3)
        rep[name] = row
        print(name, " ".join(f"L{l}:{v:.3f}" for l, v in row.items()), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
