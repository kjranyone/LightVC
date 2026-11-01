"""D267(学習なし・results/conv_p0/d267_prereg.yaml): 目標関数 v1⊥ = 入力の TTS らしさを取り除いた WavLM の特徴の空間で本物の女声と区別できないこと。
TTS らしさ = 本物の女声(REAL_POOL 150)対 Irodori-TTS v4 Large の女声のクローン(data/realx/clone・当て 200 話者)の向きを反復的な零空間への射影(INLP)で取り除く(保留の 60 話者で AUC ≤ 0.60 まで)。
    uv run python d267_v1perp.py --s <dev の親> --out ../results/conv_p0/d267.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--basis_only", action="store_true")
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from scipy.stats import spearmanr
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

    R = np.load(ROOT / "results/conv_p0/real_pool_wavlm_l468.npy")
    spk: dict = {}
    for f in sorted((ROOT / "data/realx/clone").glob("*_CLN.wav")):
        spk.setdefault(f.name.split("_")[0], []).append(f)
    keys = sorted(spk)
    random.Random(267).shuffle(keys)
    Cfit = np.array([emb(sorted(spk[k])[0]) for k in keys[:200]])
    Chold = np.array([emb(sorted(spk[k])[0]) for k in keys[200:260]])
    print("clones", len(Cfit), len(Chold), flush=True)
    sc = StandardScaler().fit(np.concatenate([R, Cfit]))
    Rz, Cz, Hz = sc.transform(R), sc.transform(Cfit), sc.transform(Chold)
    B = np.zeros((0, Rz.shape[1]))
    proj = lambda X, B: X - (X @ B.T) @ B if len(B) else X
    hist = []
    for it in range(40):
        Xa = np.concatenate([proj(Rz, B), proj(Cz, B)])
        ya = np.r_[np.zeros(len(Rz)), np.ones(len(Cz))]
        clf = LogisticRegression(C=0.01, max_iter=3000).fit(Xa, ya)
        auc_h = roc_auc_score(np.r_[np.zeros(len(Rz)), np.ones(len(Hz))], clf.decision_function(np.concatenate([proj(Rz, B), proj(Hz, B)])))
        hist.append(round(float(auc_h), 4))
        if auc_h <= 0.60:
            break
        w = clf.coef_[0] - (B.T @ (B @ clf.coef_[0]) if len(B) else 0)
        B = np.vstack([B, w / np.linalg.norm(w)])
    print("INLP iterations", len(hist), "held-out clone AUC history", hist[:5], "...", hist[-3:], flush=True)
    np.savez(ROOT / "results/conv_p0/v1perp_basis.npz", B=B, mean=sc.mean_, scale=sc.scale_, R=R)
    if getattr(a, "basis_only", False):
        return 0
    groups = {
        "FF_GMFL": sorted((S / "devff/d263").glob("p00*_GMFL.wav")), "FF_T07F": sorted((S / "devff/d263").glob("p00*_T07F.wav")),
        "FF_X05F": sorted((S / "devff/d264").glob("p00*_X05F.wav")), "FF_PROD": sorted((S / "devff/d114").glob("p00*_G03S10ZH.wav")),
        "FF_KNN": sorted((S / "devff/d135_real").glob("p00*_KNN.wav")), "FF_SRC": sorted((S / "devff/d56").glob("p00*_SRC.wav")),
        "MF_PROD": sorted((S / "dev/d114").glob("p00*_G03S10ZH.wav")), "MF_RRQ": sorted((S / "dev/d139").glob("p00*_RRQSDC19LE.wav")),
        "MF_T07F": sorted((S / "dev/d263").glob("p00*_T07F.wav")), "MF_GMFL": sorted((S / "dev/d263").glob("p00*_GMFL.wav")),
        "MF_A100": sorted((S / "dev/d258").glob("p00*_A100.wav")), "MF_KNN": sorted((S / "dev/d135_real").glob("p00*_KNN.wav")),
        "MF_SRC": sorted((S / "dev/d56").glob("p00*_SRC.wav")),
    }
    rf = np.arange(len(Rz)) % 5
    rep: dict = {"inlp_iters": len(hist), "clone_auc_hist": hist}

    def two_sample(Xz: np.ndarray, tg: np.ndarray, Bs: np.ndarray) -> tuple:
        Xa = np.concatenate([proj(Rz, Bs), proj(Xz, Bs)])
        ya = np.r_[np.zeros(len(Rz)), np.ones(len(Xz))]
        fa = np.r_[rf, tg % 5]
        pr = np.zeros(len(ya))
        for k in range(5):
            te = fa == k
            s2 = StandardScaler().fit(Xa[~te])
            c2 = LogisticRegression(C=0.01, max_iter=2000).fit(s2.transform(Xa[~te]), ya[~te])
            pr[te] = c2.predict_proba(s2.transform(Xa[te]))[:, 1]
        return float(roc_auc_score(ya, pr)), float(np.median(pr[len(Rz):]))

    I = np.zeros((0, Rz.shape[1]))
    rep["clone_holdout"] = {"v1": two_sample(Hz, np.arange(len(Hz)), I), "v1perp": two_sample(Hz, np.arange(len(Hz)), B)}
    print("clone holdout", rep["clone_holdout"], flush=True)
    for g, files in groups.items():
        if not files:
            continue
        Xz = sc.transform(np.array([emb(f) for f in files]))
        tg = np.array([int(f.name.split("_t")[1][:3]) for f in files])
        v1 = two_sample(Xz, tg, I)
        vp = two_sample(Xz, tg, B)
        rep[g] = {"n": len(files), "v1_auc": round(v1[0], 4), "v1_pfake": round(v1[1], 4), "v1perp_auc": round(vp[0], 4), "v1perp_pfake": round(vp[1], 4)}
        print(g, rep[g], flush=True)
    ff = [g for g in ("FF_GMFL", "FF_T07F", "FF_X05F", "FF_PROD") if g in rep]
    rho = spearmanr([rep[g]["v1_pfake"] for g in ff], [rep[g]["v1perp_pfake"] for g in ff]).correlation
    rep["ff_order_spearman"] = round(float(rho), 3)
    v1ok = rep["clone_holdout"]["v1perp"][0] <= 0.65
    rep["verdict"] = "USABLE" if v1ok and rho >= 0.8 else "NOT_USABLE"
    print("verdict", rep["verdict"], "clone v1perp auc", round(rep["clone_holdout"]["v1perp"][0], 3), "ff rho", rep["ff_order_spearman"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
