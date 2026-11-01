"""D118(学習なし・診断専用): 話し手のリズムを保つ変換の同一性の天井。D117: 自然さの 4 軸を通る構成は同一性 LDA top-1 0.22〜0.28(本物 0.53・合格線 0.42)。
リアルタイムの変換は操作者の時間の流れを保つ。目標本人の全部の特徴を男声の時間へ伸縮した AWARP(D106・DTW は診断専用)の同一性が合格線に届かなければ、
合格線は原理的に越えられない線 = 物差しの較正をやり直す。届くなら差は条件の作り方で詰められる(XALL / XDYN / XC0 / XF0 / XPER で帰属)。
組 = D106 のパラレル文(男声 M → 女声 TTS A・24 目標 × 5 文)。識別器 = 凍結 WavLM-large 層 4/6/8 の平均 + 標準偏差 → LDA(d4b と同じ)・学習話者 = 女声 TTS コーパスの目標以外(各 4 発話)= 目標と同じ領域。
目標の中心 = 目標の D106 で使っていない文の neutral(最大 8 本・8s)の LDA 空間の平均。本人の天井 = 目標の D106 の文の neutral の実音声(中心に入っていない)。24 目標中の順位。
出力 = results/conv_p0/d118.json。
    uv run python d118_identity_ceiling.py --dir <scratchpad>/d106
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
FEM = ROOT / "data/female_tts_corpus"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--conds", nargs="*", default=["ASELF", "AWARP", "XALL", "XDYN", "XC0", "XF0", "XPER", "C0MAP", "CONV"])
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--tag", default="d118")
    a = ap.parse_args()
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from transformers import WavLMModel
    from d4b_lda_verifier import load16
    d = Path(a.dir)
    meta = {}
    for f in d.glob("m*_*_t*_*.wav"):
        m = re.match(r"m(\d+)_([0-9a-f]+)_t(\d+)_(.+)\.wav", f.name)
        meta.setdefault(m.group(2), set()).add(int(m.group(3)))
    tg = sorted(meta)
    dev = "cuda"
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dev).eval().float()

    @torch.no_grad()
    def emb(x: np.ndarray) -> np.ndarray:
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to(dev), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    X, y = [], []
    others = sorted(p.name for p in FEM.iterdir() if p.is_dir() and p.name not in meta)
    for si, s in enumerate(others):
        ws = sorted(FEM / s / f for f in [w.name for w in (FEM / s).glob("t*_neutral.wav")])
        for w in ws[: a.per]:
            X.append(emb(load16(w, 8.0))); y.append(si)
        if si % 100 == 0:
            print("train", si, len(others), flush=True)
    X, y = np.stack(X), np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    lda = LDA(solver="eigen", shrinkage=0.3, n_components=128).fit((X - mu) / sd, y)
    tr = lambda E: lda.transform((E - mu) / sd)
    nz = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    cen = []
    for s in tg:
        ws = [w for w in sorted((FEM / s).glob("t*_neutral.wav")) if int(re.match(r"t(\d+)", w.name).group(1)) not in meta[s]][:8]
        cen.append(tr(np.stack([emb(load16(w, 8.0)) for w in ws])).mean(0))
    cen = np.stack(cen)
    mc = cen.mean(0, keepdims=True)
    C = nz(cen - mc)

    def rank(x: np.ndarray, ti: int) -> int:
        se = C @ nz(tr(emb(x)[None]) - mc)[0]
        return int((se > se[ti]).sum()) + 1

    rep = {"n_targets": len(tg), "n_train_spk": int(len(set(y))), "chance": round(1 / len(tg), 3)}
    rk = [rank(load16(FEM / s / f"t{k:02d}_neutral.wav", 8.0), ti) for ti, s in enumerate(tg) for k in sorted(meta[s])]
    rep["REAL_A"] = {"top1": round(float(np.mean(np.array(rk) == 1)), 3), "top5": round(float(np.mean(np.array(rk) <= 5)), 3), "mean_rank": round(float(np.mean(rk)), 2), "n": len(rk)}
    print("REAL_A", rep["REAL_A"], flush=True)
    for c in a.conds:
        rk, items = [], []
        for f in sorted(d.glob(f"m*_*_t*_{c}.wav")):
            m = re.match(r"m(\d+)_([0-9a-f]+)_t(\d+)_", f.name)
            rk.append(rank(load16(f, 8.0), tg.index(m.group(2))))
            items.append(f.name[: -len(f"_{c}.wav")])
        rk = np.array(rk)
        rep[c] = {"top1": round(float((rk == 1).mean()), 3), "top5": round(float((rk <= 5).mean()), 3), "mean_rank": round(float(rk.mean()), 2), "n": int(len(rk)),
                  "items": items, "ranks": [int(r) for r in rk]}
        print(c, rep[c], flush=True)
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
