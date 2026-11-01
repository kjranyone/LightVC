"""D241 の物差し(results/conv_p0/d241_prereg.yaml): D95 と同じ『本物の女声と区別できるか』の AUC を、男声(TTS)→ 女声・女声 → 女声・自己変換で並べる。
特徴 = 凍結 WavLM-large 層 4/6/8 の時間平均 + 標準偏差(8 秒)→ 標準化 → LogisticRegression(C=0.01)。本物 = D237 の REAL_POOL 150 人。5 分割(本物は話者・候補は目標で分ける)。
    uv run python d241_auc.py --s <dev の親> --out ../results/conv_p0/d241.json
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
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    S = Path(a.s)
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval().float()

    @torch.no_grad()
    def emb(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to("cuda"), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    base = ROOT / "female-dataset"
    real = []
    for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]:
        if r["grp"] == "REAL_POOL":
            ws = sorted((base / r["id"]).glob("*.wav"))
            real.append(emb(ws[len(ws) // 2]))
    R = np.array(real)
    ff = json.loads((S / "devff/devset.json").read_text())["pairs"]
    kind = {r["pi"]: r["kind"] for r in ff}
    G: dict = {}

    def add(g: str, files: list) -> None:
        X, tg = [], []
        for f in files:
            X.append(emb(f)); tg.append(int(f.name.split("_t")[1][:3]))
        G[g] = (np.array(X), np.array(tg))
        print(g, len(files), flush=True)

    dv, fv = S / "dev", S / "devff"
    add("MF_G", sorted((dv / "d114").glob("*_G03S10ZH.wav")))
    add("MF_K", sorted((dv / "d135_real").glob("*_KNN.wav")))
    add("MF_R", sorted((dv / "d139").glob("*_RRQSDC19LE.wav")))
    add("MF_SRC", sorted((dv / "d56").glob("*_SRC.wav")))
    sel = lambda d, c, k: [f for f in sorted((fv / d).glob(f"*_{c}.wav")) if kind.get(int(f.name[1:5])) == k]
    add("FF_G", sel("d114", "G03S10ZH", "FF"))
    add("FF_K", sel("d135_real", "KNN", "FF"))
    add("FF_SRC", sel("d56", "SRC", "FF"))
    add("AA_G", sel("d114", "G03S10ZH", "AA"))
    add("AA_SRC", sel("d56", "SRC", "AA"))
    rep: dict = {}
    rf = np.arange(len(R)) % 5
    for g, (X, tg) in G.items():
        Xa = np.concatenate([R, X])
        ya = np.r_[np.zeros(len(R)), np.ones(len(X))]
        fa = np.r_[rf, tg % 5]
        pr = np.zeros(len(ya))
        for k in range(5):
            te = fa == k
            sc = StandardScaler().fit(Xa[~te])
            clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Xa[~te]), ya[~te])
            pr[te] = clf.predict_proba(sc.transform(Xa[te]))[:, 1]
        rep[g] = {"auc": round(float(roc_auc_score(ya, pr)), 4), "n": len(X), "p_fake_median": round(float(np.median(pr[len(R):])), 4)}
        print(g, rep[g], flush=True)
    ok = rep["FF_SRC"]["auc"] <= 0.65
    v = "INVALID" if not ok else "SYSTEM_OK" if rep["FF_G"]["auc"] <= 0.95 else "CROSS_FAIL" if rep["AA_G"]["auc"] <= 0.95 else "SELF_FAIL"
    rep["verdict"] = v
    print("verdict", v, flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
