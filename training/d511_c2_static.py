"""D511: C2 の特徴の男声の名残は静的(話者・発話の平均のずれ)か動的(フレームの形)か。同じ言語の実音声(VCTK の評価外の男声 対 女声)で、
フレームの性別の当て(話者で分けた半分で学習・残りで評価・ロジスティック)を 生の特徴・発話の平均を引いた特徴・因果な走行平均(時定数 4 s・最初のフレームで初期化)を引いた特徴 で比べる。
    uv run python d511_c2_static.py --c2 ../results/c2_1/last.pt
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


def ema_sub(z: np.ndarray, tau_fr: float) -> np.ndarray:
    a = 1.0 / tau_fr
    m = z[:, 0].copy()
    out = np.empty_like(z)
    for t in range(z.shape[1]):
        m = (1 - a) * m + a * z[:, t]
        out[:, t] = z[:, t] - m
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2", required=True)
    ap.add_argument("--n_spk", type=int, default=40)
    a = ap.parse_args()
    dev = "cuda"
    P.cvin_setup(a.c2, dev)
    rng = random.Random(511)
    rows = [r for r in json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())["rows"] if r["ok"] and r["split"] == "train" and r["src"] == "vctk_m"]
    by: dict = {}
    for r in rows:
        by.setdefault(("M", r["spk"]), []).append(r["wav"])
    for w, _, _, _ in TC.vctk_female_rows():
        by.setdefault(("F", Path(w).parent.name), []).append(w)
    keys = [k for k in by if k[0] == "M"][: a.n_spk] + [k for k in by if k[0] == "F"][: a.n_spk]
    data = []
    for si, k in enumerate(keys):
        ws = by[k][:]
        rng.shuffle(ws)
        zs = []
        for w in ws[:4]:
            x = load48(w).astype(np.float32)
            x = x[: len(x) // 240 * 240]
            with torch.no_grad():
                z = P.cvin_frames(torch.from_numpy(TC.prime(x))[None].to(dev), len(x) // 240)[0].cpu().numpy()
            m = TC.speech_mask(x)[: z.shape[1]]
            zs.append((z, m))
        z_all = np.concatenate([z for z, _ in zs], 1)
        m_all = np.concatenate([m for _, m in zs])
        data.append((si, k[0], z_all, m_all))
    print("speakers", len(data), flush=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sp = list(range(len(data)))
    rng.shuffle(sp)
    tr = set(sp[: len(sp) // 2])
    rep = {}
    for name, fn in (("raw", lambda z, m: z), ("utt_mean_sub", lambda z, m: z - z[:, m].mean(1, keepdims=True)), ("ema4s_sub", lambda z, m: ema_sub(z, 4.0 * 200)), ("ema1s_sub", lambda z, m: ema_sub(z, 1.0 * 200))):
        Xtr, ytr, Xte, yte = [], [], [], []
        for si, g, z, m in data:
            f = fn(z, m)[:, m][:, ::5].T
            (Xtr if si in tr else Xte).append(f)
            (ytr if si in tr else yte).append(np.full(len(f), 1 if g == "M" else 0))
        Xtr, ytr, Xte, yte = np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(Xte), np.concatenate(yte)
        sc = StandardScaler().fit(Xtr)
        c = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
        acc = float((c.predict(sc.transform(Xte)) == yte).mean())
        rep[name] = {"frame_gender_acc_heldspk": round(acc, 3), "frame_var": round(float(Xte.var(0).sum()), 2)}
        print(name, rep[name], flush=True)
    (ROOT / "results/conv_p0/d511.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
