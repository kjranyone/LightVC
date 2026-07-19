from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import librosa
import torch

from build_formant_cache import formant_track, SR


def summary(fp: str):
    try:
        d = torch.load(fp, weights_only=False)
        x, _ = librosa.load(d["path"], sr=SR, mono=True)
        ft = formant_track(x.astype(np.float64))
        if len(ft) < 20:
            return None
        f1, f2 = ft[:, 0], ft[:, 1]
        m = np.isfinite(f1) & np.isfinite(f2)
        if m.sum() < 12:
            return None
        f1, f2 = f1[m], f2[m]
        df2 = np.abs(np.diff(f2))
        feats = np.array([
            np.median(f1), np.median(f2),
            np.percentile(f2, 90) - np.percentile(f2, 10),
            np.std(f1) * np.std(f2) / 1e4,
            np.percentile(df2, 90) if len(df2) else 0.0,
        ], dtype=np.float32)
        return (str(d["path"]), feats)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--out", default="../data/artic_feats.pt")
    args = ap.parse_args()
    files = [str(f) for f in Path(args.feat).rglob("*.pt")]
    with Pool(12) as p:
        res = [r for r in p.map(summary, files) if r is not None]
    torch.save({k: v for k, v in res}, args.out)
    print(f"artic feats for {len(res)} utts -> {args.out}")


if __name__ == "__main__":
    main()
