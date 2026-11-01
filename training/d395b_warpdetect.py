"""D395b(設計三層則の検査 C・事後の入れ替えの学習の前): DTW で写した事後(SWAP_F・SWAP_M)が、自然な事後(F 自身)と窓の統計で見分けられるか。
見分けられれば生成器は「写した事後か」を手がかりに入れ替えの標本だけ扱いを変えられる(推論の自然な事後に効かないチート)。
窓 = 100 フレーム(0.5 s)・特徴 = エントロピーの平均と標準偏差・最大確率の平均・隣のフレームとの確率の L1 の平均と標準偏差・argmax が前と同じ割合・argmax の連の長さの平均。
比べる: 自然 対 写した事後 / 自然 対 往復で写した自然な事後(F → X → F・自己の写しの対照の候補)。目標でまとめた 5 分割のロジスティック回帰の AUC。
    uv run python d395b_warpdetect.py --set <dev の親>/d395 --out ../results/conv_p0/d395b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P


def path(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    C = 1.0 - np.sqrt(pa) @ np.sqrt(pb).T
    _, wp = librosa.sequence.dtw(C=C, backtrack=True)
    return wp[::-1]


def warp_to(pa: np.ndarray, pb: np.ndarray, wp: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(pa)
    cnt = np.zeros(len(pa))
    for i, j in wp:
        acc[i] += pb[j]
        cnt[i] += 1
    return acc / cnt[:, None]


def smooth_warp(pa: np.ndarray, pb: np.ndarray, wp: np.ndarray, win: int = 25) -> np.ndarray:
    """滑らかな時間写像: DTW の経路から a のフレームごとの b の平均の位置 → 移動平均(win フレーム)→ 単調化 → b の確率を線形補間で読む(フレームの重複と平均の均しを避ける)。"""
    s_ = np.zeros(len(pa))
    c_ = np.zeros(len(pa))
    for i, j in wp:
        s_[i] += j
        c_[i] += 1
    jb = s_ / c_
    k = np.ones(win) / win
    jp = np.convolve(np.pad(jb, (win // 2, win - 1 - win // 2), mode="edge"), k, "valid")
    jp = np.clip(np.maximum.accumulate(jp), 0, len(pb) - 1)
    j0 = np.floor(jp).astype(int)
    j1 = np.minimum(j0 + 1, len(pb) - 1)
    fr = (jp - j0)[:, None]
    return (1 - fr) * pb[j0] + fr * pb[j1]


def feats(p: np.ndarray, w: int = 100) -> np.ndarray:
    out = []
    for s in range(0, len(p) - w + 1, w):
        q = p[s:s + w]
        h = -(q * np.log(q + 1e-8)).sum(1)
        d = np.abs(np.diff(q, axis=0)).sum(1)
        am = q.argmax(1)
        same = (am[1:] == am[:-1]).mean()
        runs = np.diff(np.flatnonzero(np.r_[True, am[1:] != am[:-1], True]))
        out.append([h.mean(), h.std(), q.max(1).mean(), d.mean(), d.std(), same, runs.mean()])
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = Path(a.set)
    trip = json.loads((S / "set.json").read_text())["triples"]
    X: dict = {k: [] for k in ("NAT", "W_F2", "W_M", "RT_F2", "S_F2", "S_M", "SRT_F2")}
    G: dict = {k: [] for k in X}
    for t in trip:
        sm = lambda r: torch.from_numpy(P.load_dump(S / "dump", f"{t['key']}_{r}")[1]).softmax(1).numpy().astype(np.float64)
        pf, p2, pm = sm("F"), sm("F2"), sm("M")
        w2 = path(pf, p2)
        f_on_2 = warp_to(p2, pf, w2[:, ::-1])
        rt = warp_to(pf, f_on_2, w2)
        wm = path(pf, pm)
        srt = smooth_warp(pf, smooth_warp(p2, pf, w2[:, ::-1]), w2)
        for k, q in (("NAT", pf), ("W_F2", warp_to(pf, p2, w2)), ("W_M", warp_to(pf, pm, wm)), ("RT_F2", rt), ("S_F2", smooth_warp(pf, p2, w2)), ("S_M", smooth_warp(pf, pm, wm)), ("SRT_F2", srt)):
            f = feats(q)
            X[k].append(f)
            G[k].append(np.full(len(f), t["ti"]))
    rep: dict = {}
    for a_, b_ in (("NAT", "W_F2"), ("NAT", "W_M"), ("NAT", "RT_F2"), ("RT_F2", "W_F2"), ("RT_F2", "W_M"), ("NAT", "S_F2"), ("NAT", "S_M"), ("NAT", "SRT_F2"), ("SRT_F2", "S_F2"), ("SRT_F2", "S_M")):
        Xa, Xb = np.concatenate(X[a_]), np.concatenate(X[b_])
        ga, gb = np.concatenate(G[a_]), np.concatenate(G[b_])
        Xx, y, g = np.concatenate([Xa, Xb]), np.r_[np.zeros(len(Xa)), np.ones(len(Xb))], np.r_[ga, gb]
        pr = np.zeros(len(y))
        for k in range(5):
            te = g % 5 == k
            sc = StandardScaler().fit(Xx[~te])
            c = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xx[~te]), y[~te])
            pr[te] = c.predict_proba(sc.transform(Xx[te]))[:, 1]
        rep[f"{a_}_vs_{b_}"] = {"auc": round(float(roc_auc_score(y, pr)), 4), "n": [len(Xa), len(Xb)],
                                "mean_a": np.round(Xa.mean(0), 3).tolist(), "mean_b": np.round(Xb.mean(0), 3).tolist()}
        print(a_, "vs", b_, rep[f"{a_}_vs_{b_}"], flush=True)
    rep["feature_names"] = ["H_mean", "H_std", "maxp_mean", "dL1_mean", "dL1_std", "same_argmax", "run_len"]
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
