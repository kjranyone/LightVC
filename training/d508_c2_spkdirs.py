"""D508: C2 の特徴(64・白色化した ContentVec 主成分の因果な推定)の話者の方向。学習の話者(実女声・TTS 女声・VCTK の評価外の男声・その日本語 TTS クローン)から
話者ごとに数ファイル、C2 を因果に流して音声区間のフレームを集め、話者の間の散らばり対 話者の中の散らばりの一般化固有問題(LDA)の上位 r 方向を求める。
射影 P⊥ = I − W (WᵀW)⁻¹ Wᵀ で外したとき: 性別の当て(発話の平均・話者で分けた半分で学習・残りで評価)・話者の当て・フレームの分散の残り割合を r = 0, 4, 8, 16, 24 で並べる。
出力 = results/conv_p0/c2_spkdirs.npz(W_r ごとの射影)・d508.json。評価の話者(外郎売・dev の操作者・devff の操作者・VCTK の評価男声)は使わない。
    uv run python d508_c2_spkdirs.py --c2 ../results/c2_1/last.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_c1 as TC
import train_ppgvoc as P
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2", required=True)
    ap.add_argument("--n_spk", type=int, default=120)
    ap.add_argument("--n_file", type=int, default=3)
    a = ap.parse_args()
    dev = "cuda"
    P.cvin_setup(a.c2, dev)
    rng = random.Random(508)
    fem = [r for r in json.loads((ROOT / "data/rvoc_f0hi/manifest.json").read_text())["rows"] if r["keep"]]
    mal = [r for r in json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())["rows"] if r["ok"] and r["split"] == "train" and r["src"] in ("vctk_m", "tts_m")]
    by: dict = {}
    for r in fem:
        by.setdefault(("F", r["spk"] + "_" + r["src"]), []).append(r["wav"])
    for r in mal:
        by.setdefault(("M", r["spk"] + "_" + r["src"]), []).append(r["wav"])
    fs = [k for k in by if k[0] == "F"]
    ms = [k for k in by if k[0] == "M"]
    rng.shuffle(fs)
    rng.shuffle(ms)
    spks = fs[: a.n_spk] + ms[: a.n_spk]
    utt, frames = [], []
    for si, key in enumerate(spks):
        ws = by[key][:]
        rng.shuffle(ws)
        for w in ws[: a.n_file]:
            x = load48(w).astype(np.float32)
            x = x[: len(x) // 240 * 240][: 48000 * 10]
            if len(x) < 48000:
                continue
            with torch.no_grad():
                z = P.cvin_frames(torch.from_numpy(TC.prime(x))[None].to(dev), len(x) // 240)[0].cpu().numpy()
            m = TC.speech_mask(x)[: z.shape[1]]
            if m.sum() < 50:
                continue
            zz = z[:, m].T
            utt.append((si, key[0], zz.mean(0)))
            frames.append((si, zz[:: 4]))
    S = np.array([u[0] for u in utt])
    G = np.array([1 if u[1] == "M" else 0 for u in utt])
    U = np.stack([u[2] for u in utt])
    Fr = np.concatenate([f[1] for f in frames])
    Fs = np.concatenate([np.full(len(f[1]), f[0]) for f in frames])
    print("speakers", len(set(S.tolist())), "utterances", len(U), "frames", len(Fr), flush=True)
    sp = sorted(set(Fs.tolist()))
    mu = {s: Fr[Fs == s].mean(0) for s in sp}
    Xw = np.concatenate([Fr[Fs == s] - mu[s] for s in sp])
    Sw = Xw.T @ Xw / len(Xw)
    M = np.stack([mu[s] for s in sp])
    Sb = np.cov(M.T)
    from scipy.linalg import eigh
    ev, V = eigh(Sb, Sw + 1e-6 * np.eye(len(Sw)))
    V = V[:, ::-1]
    tr = sorted(set(S.tolist()))
    rng2 = random.Random(5081)
    rng2.shuffle(tr)
    half = set(tr[: len(tr) // 2])
    a_idx = np.array([s in half for s in S])
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tot = float(((Fr - Fr.mean(0)) ** 2).sum())
    rep, Ws = {}, {}
    for r in (0, 4, 8, 16, 24):
        if r:
            W = V[:, :r]
            Pp = np.eye(len(W)) - W @ np.linalg.solve(W.T @ W, W.T)
        else:
            Pp = np.eye(V.shape[0])
        Ws[f"P{r}"] = Pp.astype(np.float32)
        Uz, Fz = U @ Pp.T, Fr @ Pp.T
        sc = StandardScaler().fit(Uz[a_idx])
        g = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Uz[a_idx]), G[a_idx])
        gacc = float((g.predict(sc.transform(Uz[~a_idx])) == G[~a_idx]).mean())
        keep = float(((Fz - Fz.mean(0)) ** 2).sum() / tot)
        w_ = np.concatenate([Fz[Fs == s] - Fz[Fs == s].mean(0) for s in sp])
        keep_within = float((w_ ** 2).sum() / (Xw ** 2).sum())
        rep[r] = {"gender_acc_heldspk": round(gacc, 3), "frame_var_kept": round(keep, 3), "within_spk_var_kept": round(keep_within, 3)}
        print("r", r, rep[r], flush=True)
    np.savez(ROOT / "results/conv_p0/c2_spkdirs.npz", V=V.astype(np.float32), ev=ev[::-1].astype(np.float32), **Ws)
    (ROOT / "results/conv_p0/d508.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
