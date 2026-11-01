"""D502: ContentVec(lengyue233/content-vec-best・最終層・L2 正規化)のフレームの主成分の基底(diag_ppgvoc49 の生成器の入力 = 連続の内容の特徴の上限の診断)。
学習の台帳(rvoc_f0hi・実女声)から無作為に 600 ファイル × 4 s を取り、フレームを集めて 64 成分。出力 = results/conv_p0/cv_pca64.npz(mean・comp [64, 768]・ev)。
    uv run python d502_cv_pca.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--k", type=int, default=64)
    a = ap.parse_args()
    from transformers import HubertModel
    m = HubertModel.from_pretrained("lengyue233/content-vec-best").to(a.device).eval()
    rows = [r for r in json.loads((ROOT / "data/rvoc_f0hi/manifest.json").read_text())["rows"] if "female-dataset" in r["wav"]]
    rng = random.Random(502)
    rng.shuffle(rows)
    H = []
    for r in rows[: a.n]:
        p = Path(r["wav"])
        if not p.exists():
            continue
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(1)
        if len(x) < 4 * sr:
            continue
        o = rng.randrange(0, len(x) - 4 * sr + 1)
        x16 = resample_poly(x[o:o + 4 * sr].astype(np.float64), 16000, sr).astype(np.float32)
        with torch.no_grad():
            h = F.normalize(m(torch.from_numpy(x16)[None].to(a.device)).last_hidden_state[0], dim=-1).cpu().numpy()
        H.append(h)
    X = np.concatenate(H)
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    ev = S ** 2 / (len(X) - 1)
    print("files", len(H), "frames", len(X), "explained", round(float(ev[: a.k].sum() / ev.sum()), 3), flush=True)
    np.savez(ROOT / "results/conv_p0/cv_pca64.npz", mean=mu.astype(np.float32), comp=Vt[: a.k].astype(np.float32), ev=ev[: a.k].astype(np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
